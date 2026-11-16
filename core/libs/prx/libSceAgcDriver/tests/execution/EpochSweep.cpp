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
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

namespace GuestMemory = AgcDriver::GuestMemory;

constexpr std::uint64_t Block = 65536;
constexpr std::uint64_t Page = 4096;
constexpr std::uint64_t Blocks = 32;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::atomic<std::uint64_t> osWalks{0};

void Hook(int point) {
    if (point != 1) return;
    osWalks.fetch_add(1);
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

GuestMemory::ContentToken Take(std::uint64_t address, std::size_t bytes) {
    Require(GuestMemory::CollectWrites(address, bytes) != 0, "a collect of the watched block failed");
    const auto token = GuestMemory::TokenFor(address, bytes);
    Require(token.content != 0, "a range collected in the thread's epoch got no content token");
    Require(GuestMemory::StillCurrent(address, bytes, token), "a token just taken does not hold");
    return token;
}

void SweepSeesAWriteMadeBeforeTheEpoch(std::uint64_t base) {
    const auto written = base + 2 * Block + 3 * Page;
    const auto quiet = base + 5 * Block + Page;
    GuestMemory::BumpCollectEpoch();
    const auto writtenToken = Take(written, 2 * Page);
    const auto quietToken = Take(quiet, 2 * Page);
    const auto generation = GuestMemory::CollectWrites(written, 2 * Page);
    Store(written + Page + 7);
    const auto walks = osWalks.load();
    GuestMemory::BumpCollectEpoch();
    const auto swept = osWalks.load();
    Require(swept > walks, "the epoch advance did not sweep the blocks asked in the epoch before");
    Require(!GuestMemory::StillCurrent(written, 2 * Page, writtenToken), "a token held across a store made before the epoch advance");
    Require(GuestMemory::StillCurrent(quiet, 2 * Page, quietToken), "a token broke although nothing wrote its block");
    Require(GuestMemory::CollectWrites(written, 2 * Page) != 0 && GuestMemory::CollectWrites(quiet, 2 * Page) != 0, "a collect of a swept block failed");
    Require(osWalks.load() == swept, "a collect of a block the sweep had walked in the epoch walked it again");
    Require(!GuestMemory::UnchangedSince(written, 2 * Page, generation), "the sweep did not stamp the store it collected");
    const auto retaken = Take(written, 2 * Page);
    GuestMemory::BumpCollectEpoch();
    Require(GuestMemory::StillCurrent(written, 2 * Page, retaken), "a token taken after the store broke with no further write");
    Require(GuestMemory::StillCurrent(quiet, 2 * Page, quietToken), "a quiet token broke at the second epoch advance");
}

void NoStoreIsMissed(std::uint64_t base) {
    const auto first = base + 8 * Block;
    constexpr std::uint64_t Ranges = 6;
    constexpr std::size_t Bytes = 3 * Page;
    std::array<GuestMemory::ContentToken, Ranges> tokens{};
    GuestMemory::BumpCollectEpoch();
    for (std::uint64_t index = 0; index < Ranges; ++index) tokens[index] = Take(first + index * Block + index * Page, Bytes);
    std::uint64_t seed = 0x9E3779B97F4A7C15ull;
    for (int round = 0; round < 200; ++round) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const auto target = (seed >> 33) % Ranges;
        const auto offset = (seed >> 13) % Bytes;
        const auto address = first + target * Block + target * Page;
        Store(address + offset);
        GuestMemory::BumpCollectEpoch();
        for (std::uint64_t index = 0; index < Ranges; ++index) {
            const auto range = first + index * Block + index * Page;
            const bool held = GuestMemory::StillCurrent(range, Bytes, tokens[index]);
            if (index == target) Require(!held, "a token held across a store into its range");
            else Require(held, "a token broke although the store went to another block");
        }
        tokens[target] = Take(address, Bytes);
    }
}

void DriverWritesBreakATokenAtOnce(std::uint64_t base) {
    const auto address = base + 15 * Block + 2 * Page;
    constexpr std::size_t Bytes = 2 * Page;
    GuestMemory::BumpCollectEpoch();
    auto token = Take(address, Bytes);
    const std::array<std::byte, 16> bytes{};
    GuestMemory::StoreOwnBytes(address + 64, bytes);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a token held across a store of the driver");
    token = Take(address, Bytes);
    Store(address + 5);
    GuestMemory::MarkWritten(address, 16);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a token held across a write the GPU reported");
    token = Take(address, Bytes);
    GuestMemory::Write(address + Page, bytes);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a token held across a DMA write");
    token = Take(address, Bytes);
    GuestMemory::NoteGpuPending(address, 32);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a token held across a recorded GPU write");
    token = Take(address, Bytes);
    GuestMemory::NoteRegistryChange(address, 32);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a token held across a registry change");
    token = Take(address, Bytes);
    GuestMemory::BreakProofs(GuestMemory::ProofBreak::Commit);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a token held across a commit that declared no ranges");
    token = Take(address, Bytes);
    GuestMemory::MarkWritten(address + Block, 16);
    Require(GuestMemory::StillCurrent(address, Bytes, token), "a token broke on a write to the next block");
    Require(!GuestMemory::StillCurrent(address, Bytes, GuestMemory::ContentToken{}), "an empty token held");
}

void WriteBeforeASubmissionIsSeenByTheDrawThatReads(std::uint64_t base) {
    const auto address = base + 18 * Block + Page;
    constexpr std::size_t Bytes = Page;
    GuestMemory::BumpCollectEpoch();
    const auto first = GuestMemory::HeldCollectEpoch();
    GuestMemory::ContentToken token;
    std::uint64_t generation = 0;
    std::thread([&] {
        GuestMemory::MarkProofThread();
        GuestMemory::AdoptCollectEpoch(first);
        generation = GuestMemory::CollectWrites(address, Bytes);
        token = Take(address, Bytes);
    }).join();
    std::thread([&] { Store(address + 9); }).join();
    GuestMemory::BumpCollectEpoch();
    const auto second = GuestMemory::HeldCollectEpoch();
    bool heldAtOld = true;
    bool heldAtNew = true;
    bool unchanged = true;
    bool tookWithoutEpoch = true;
    std::thread([&] {
        tookWithoutEpoch = GuestMemory::TokenFor(address, Bytes).content != 0;
        GuestMemory::MarkProofThread();
        GuestMemory::AdoptCollectEpoch(first);
        heldAtOld = GuestMemory::StillCurrent(address, Bytes, token);
        GuestMemory::AdoptCollectEpoch(second);
        heldAtNew = GuestMemory::StillCurrent(address, Bytes, token);
        unchanged = GuestMemory::CollectWrites(address, Bytes) != 0 && GuestMemory::UnchangedSince(address, Bytes, generation);
    }).join();
    Require(!tookWithoutEpoch, "a thread that holds no epoch got a content token");
    Require(!heldAtNew, "the draw of the next submission read a token that hid a store made before the submission");
    Require(!heldAtOld, "a store the sweep stamped left a token of an earlier epoch standing");
    Require(!unchanged, "the write tracker did not report the store made before the submission");
    Store(address + 11);
    std::thread([&] {
        GuestMemory::MarkProofThread();
        GuestMemory::AdoptCollectEpoch(second);
        token = Take(address, Bytes);
    }).join();
    Store(address + 13);
    GuestMemory::BumpCollectEpoch();
    const auto third = GuestMemory::HeldCollectEpoch();
    std::thread([&] {
        GuestMemory::MarkProofThread();
        GuestMemory::AdoptCollectEpoch(third);
        heldAtNew = GuestMemory::StillCurrent(address, Bytes, token);
    }).join();
    Require(!heldAtNew, "a store made after the token and before the next submission was hidden from that submission");
}

void ColdBlocksLeaveTheSweep(std::uint64_t base) {
    const auto address = base + 24 * Block + Page;
    GuestMemory::BumpCollectEpoch();
    const auto token = Take(address, Page);
    const auto before = osWalks.load();
    GuestMemory::BumpCollectEpoch();
    Require(osWalks.load() > before, "a block asked in the last epoch was not swept");
    std::uint64_t late = 0;
    for (int bump = 0; bump < 160; ++bump) {
        const auto walks = osWalks.load();
        GuestMemory::BumpCollectEpoch();
        if (bump >= 150) late += osWalks.load() - walks;
    }
    Require(late == 0, "blocks nobody asked about for 150 epochs are still swept");
    Require(!GuestMemory::StillCurrent(address, Page, token), "a token held over a block no walk observed in the epoch");
    const auto walks = osWalks.load();
    Require(GuestMemory::CollectWrites(address, Page) != 0 && osWalks.load() > walks, "a cold block was not walked on demand");
    Require(GuestMemory::StillCurrent(address, Page, token), "a token over a cold block nothing wrote did not hold once the block was observed again");
}

void NeighbourPagesLeaveATokenStanding(std::uint64_t base) {
    const auto block = base + 26 * Block;
    const auto inside = block + 2 * Page + 100;
    constexpr std::size_t InsideBytes = 2 * Page - 200;
    const auto wide = block + 9 * Page;
    constexpr std::size_t WideBytes = 2 * Block;
    GuestMemory::BumpCollectEpoch();
    auto token = Take(inside, InsideBytes);
    auto wideToken = Take(wide, WideBytes);
    Store(block + 6 * Page + 1);
    GuestMemory::BumpCollectEpoch();
    Require(GuestMemory::CollectWrites(inside, InsideBytes) != 0 && GuestMemory::CollectWrites(wide, WideBytes) != 0, "a collect of the test block failed");
    Require(GuestMemory::StillCurrent(inside, InsideBytes, token), "a store into another page of the block broke a token");
    Require(GuestMemory::StillCurrent(wide, WideBytes, wideToken), "a store before the first page of a range broke its token");
    const std::array<std::byte, 16> bytes{};
    GuestMemory::StoreOwnBytes(block + 5 * Page, bytes);
    GuestMemory::MarkWritten(block + 8 * Page - 16, 16);
    Require(GuestMemory::StillCurrent(inside, InsideBytes, token) && GuestMemory::StillCurrent(wide, WideBytes, wideToken), "a driver store into a neighbouring page broke a token");
    Store(block + 2 * Page + 3);
    Store(block + Block + 7 * Page);
    GuestMemory::BumpCollectEpoch();
    Require(GuestMemory::CollectWrites(inside, InsideBytes) != 0 && GuestMemory::CollectWrites(wide, WideBytes) != 0, "a collect of the test block failed");
    Require(!GuestMemory::StillCurrent(inside, InsideBytes, token), "a store into the first page of a range, before its first byte, left its token standing");
    Require(!GuestMemory::StillCurrent(wide, WideBytes, wideToken), "a store into a block wholly inside a range left its token standing");
    token = Take(inside, InsideBytes);
    wideToken = Take(wide, WideBytes);
    GuestMemory::MarkWritten(block + 4 * Page - 8, 8);
    GuestMemory::StoreOwnBytes(block + 2 * Block + 9 * Page - 16, bytes);
    Require(!GuestMemory::StillCurrent(inside, InsideBytes, token), "a GPU write into the last page of a range left its token standing");
    Require(!GuestMemory::StillCurrent(wide, WideBytes, wideToken), "a driver store into the last bytes of a range left its token standing");
    token = Take(inside, InsideBytes);
    wideToken = Take(wide, WideBytes);
    GuestMemory::NoteGpuPending(block + 2 * Block + 9 * Page, 64);
    Require(GuestMemory::StillCurrent(wide, WideBytes, wideToken), "a GPU write recorded just past a range broke its token");
    GuestMemory::Unwatch(block + 3 * Block, Page);
    GuestMemory::BumpCollectEpoch();
    Require(GuestMemory::CollectWrites(inside, InsideBytes) != 0 && GuestMemory::StillCurrent(inside, InsideBytes, token), "a page leaving the write watch three blocks away broke a token for good");
}

void ManyDriverStoresInABlockKeepPageGrain(std::uint64_t base) {
    const auto block = base + 23 * Block;
    const auto address = block + 10 * Page + 4;
    constexpr std::size_t Bytes = 0xacc;
    GuestMemory::BumpCollectEpoch();
    auto token = Take(address, Bytes);
    const auto generation = GuestMemory::CollectWrites(address, Bytes);
    const std::array<std::byte, 16> bytes{};
    for (std::uint64_t page = 0; page < 6; ++page) GuestMemory::StoreOwnBytes(block + page * Page + 32 * page, bytes);
    Require(!GuestMemory::UnchangedSince(address, Bytes, generation), "the tracker still places every driver store of a block that took six of them");
    for (int ask = 0; ask < 4; ++ask) {
        Require(GuestMemory::CollectWrites(address, Bytes) != 0, "a collect of the test block failed");
        Require(GuestMemory::StillCurrent(address, Bytes, token), "driver stores into other pages of the block broke a token");
    }
    GuestMemory::StoreOwnBytes(block + 10 * Page + 0x800, bytes);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a driver store into the range left its token standing after the block had taken many stores");
    token = Take(address, Bytes);
    GuestMemory::StoreOwnBytes(block + 10 * Page, std::span<const std::byte>(bytes).first(4));
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a driver store into the first page of a range, before its first byte, left its token standing");
    token = Take(address, Bytes);
    GuestMemory::NoteRegistryChange(block, Block);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a note over the whole block left a token of two of its pages standing");
    token = Take(address, Bytes);
    GuestMemory::NoteGpuPending(block - Block, 3 * Block);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a recorded GPU write over three blocks left a token inside the middle one standing");
    token = Take(address, Bytes);
    GuestMemory::NoteRegistryChange(block - 8 * Block, 80 * Block);
    Require(!GuestMemory::StillCurrent(address, Bytes, token), "a note over eighty blocks left a token inside them standing");
}

void PartlyWatchedBlockKeepsItsMemo(std::uint64_t base) {
    const auto block = base + 30 * Block;
    constexpr std::uint64_t Ranges = 13;
    GuestMemory::Unwatch(block + 15 * Page, Page);
    if (GuestMemory::Watched(block, Block) || !GuestMemory::Watched(block, 15 * Page)) {
        std::cout << "epoch sweep tests: the partly watched block case was skipped, one page could not leave the write watch\n";
        return;
    }
    const auto askAll = [&] {
        for (std::uint64_t index = 0; index < Ranges; ++index) {
            Require(GuestMemory::CollectWrites(block + index * Page, Page) != 0, "a collect inside a partly watched block failed");
            Require(GuestMemory::CollectWrites(block + index * Page, 2 * Page) != 0, "a collect of two pages inside a partly watched block failed");
        }
    };
    GuestMemory::BumpCollectEpoch();
    askAll();
    const auto walks = osWalks.load();
    askAll();
    Require(osWalks.load() == walks, "ranges of a partly watched block were walked again in the epoch that had observed them");
    Require(GuestMemory::CollectWrites(block + 14 * Page, 2 * Page) == 0, "a range that reaches an unwatched page was collected");
    const auto generation = GuestMemory::CollectWrites(block + 3 * Page, Page);
    const auto token = GuestMemory::TokenFor(block + 3 * Page, Page);
    Require(token.content != 0 && GuestMemory::StillCurrent(block + 3 * Page, Page, token), "a range of a partly watched block observed in the epoch got no token");
    Store(block + 3 * Page + 21);
    GuestMemory::BumpCollectEpoch();
    Require(!GuestMemory::StillCurrent(block + 3 * Page, Page, token), "a token over a partly watched block held in an epoch that had not observed it");
    const auto before = osWalks.load();
    Require(GuestMemory::CollectWrites(block + 3 * Page, Page) != 0 && osWalks.load() > before, "a range of a partly watched block was not walked in the next epoch");
    Require(!GuestMemory::UnchangedSince(block + 3 * Page, Page, generation), "a store into a partly watched block made before the epoch advance was not reported");
    Require(!GuestMemory::StillCurrent(block + 3 * Page, Page, token), "a token held across a store into a partly watched block");
    std::thread([&] { Store(block + 5 * Page + 2); }).join();
    bool unchanged = true;
    GuestMemory::BumpCollectEpoch();
    const auto epoch = GuestMemory::HeldCollectEpoch();
    Require(GuestMemory::CollectWrites(block + 5 * Page, Page) != 0, "a collect inside a partly watched block failed");
    std::thread([&] {
        GuestMemory::MarkProofThread();
        GuestMemory::AdoptCollectEpoch(epoch);
        const auto walked = osWalks.load();
        const bool collected = GuestMemory::CollectWrites(block + 5 * Page, Page) != 0;
        unchanged = !collected || osWalks.load() != walked || GuestMemory::UnchangedSince(block + 5 * Page, Page, generation);
    }).join();
    Require(!unchanged, "another thread of the epoch did not get the answer of the walk that observed a partly watched range, or missed its store");
}

void UnwatchedRangeHoldsNoToken(std::uint64_t base) {
    const auto address = base + 28 * Block;
    GuestMemory::BumpCollectEpoch();
    const auto token = Take(address, Page);
    GuestMemory::Unwatch(address, Block);
    Require(!GuestMemory::StillCurrent(address, Page, token), "a token held over a range that left the write watch");
    GuestMemory::BumpCollectEpoch();
    Require(!GuestMemory::StillCurrent(address, Page, token), "a token over an unwatched range held after an epoch advance");
    Require(GuestMemory::TokenFor(address, Page).content == 0, "an unwatched range got a content token");
}

void CountsAddUp() {
    const auto counts = GuestMemory::EpochSweepCounted();
    Require(counts.sweeps != 0 && counts.runs != 0 && counts.blocks >= counts.runs, "the sweep counts show no sweep");
    Require(counts.tokens != 0 && counts.held != 0, "the token counts show no token");
    Require(counts.missed[static_cast<std::size_t>(GuestMemory::TokenMiss::Written)] != 0, "the token counts show no token broken by a write");
}

}

int main(int argc, char** argv) {
    try {
#ifdef _WIN32
        _putenv_s("APS5_EPOCH_SWEEP", "1");
        _putenv_s("APS5_EPOCH_SWEEP_SPAN", "64");
        if (argc == 2 && std::string(argv[1]) == "--verify") _putenv_s("APS5_VERIFY_EPOCH_SWEEP", "1");
#else
        setenv("APS5_EPOCH_SWEEP", "1", 1);
        setenv("APS5_EPOCH_SWEEP_SPAN", "64", 1);
        if (argc == 2 && std::string(argv[1]) == "--verify") setenv("APS5_VERIFY_EPOCH_SWEEP", "1", 1);
#endif
        if (!GuestMemory::WriteWatched()) {
            std::cout << "epoch sweep tests skipped: guest memory is not write watched\n";
            return 77;
        }
        const auto base = AllocateWatched(static_cast<std::size_t>(Blocks * Block));
        if (!GuestMemory::Watched(base, static_cast<std::size_t>(Blocks * Block))) {
            std::cout << "epoch sweep tests skipped: the test block is not write watched\n";
            return 77;
        }
        Require(GuestMemory::EpochSweeps(), "the epoch sweep switch was not read");
        GuestMemory::SetObservationHookForTest(&Hook);
        if (argc == 2 && std::string(argv[1]) == "--verify") {
            NoStoreIsMissed(base);
            DriverWritesBreakATokenAtOnce(base);
            WriteBeforeASubmissionIsSeenByTheDrawThatReads(base);
            ManyDriverStoresInABlockKeepPageGrain(base);
        } else {
            SweepSeesAWriteMadeBeforeTheEpoch(base);
            NoStoreIsMissed(base);
            DriverWritesBreakATokenAtOnce(base);
            WriteBeforeASubmissionIsSeenByTheDrawThatReads(base);
            ManyDriverStoresInABlockKeepPageGrain(base);
            ColdBlocksLeaveTheSweep(base);
            NeighbourPagesLeaveATokenStanding(base);
            UnwatchedRangeHoldsNoToken(base);
            PartlyWatchedBlockKeepsItsMemo(base);
            CountsAddUp();
        }
        std::cout << "epoch sweep tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
