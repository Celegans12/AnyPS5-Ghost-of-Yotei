#define NOMINMAX
#include "prx/libc/include/WindowsMappings.hpp"
#include "prx/libc/include/WindowsPageTrack.hpp"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Mappings = GuestArena::WindowsMappings;
using Track = GuestArena::WindowsPageTrack;
using State = Track::State;
constexpr std::size_t PageBytes = 4096;
constexpr std::size_t BlockBytes = Track::BlockBytes;
constexpr std::size_t BlockPages = BlockBytes / PageBytes;
constexpr std::size_t RunBytes = 0x200000;
std::atomic<std::uint64_t> handled{0};
std::atomic<std::uint64_t> refused{0};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

LONG CALLBACK HandleFault(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    if (Track::Get().HandleWrite(record->ExceptionInformation[1])) {
        handled.fetch_add(1, std::memory_order_relaxed);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    refused.fetch_add(1, std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_SEARCH;
}

class Raw {
public:
    explicit Raw(std::size_t size) : bytes(size) {
        data = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
        Require(data != nullptr, "cannot allocate write-watched test memory");
        for (std::size_t at = 0; at < bytes; at += PageBytes) data[at] = 1;
        Require(ResetWriteWatch(data, bytes) == 0, "cannot reset the write watch of test memory");
    }
    ~Raw() {
        Track::Get().Forget(Address(), Address() + bytes);
        VirtualFree(data, 0, MEM_RELEASE);
    }
    Raw(const Raw&) = delete;
    Raw& operator=(const Raw&) = delete;

    std::uintptr_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::uintptr_t Block(std::size_t index) const { return Address() + index * BlockBytes; }
    void Write(std::size_t block, std::size_t page, unsigned char value) const { static_cast<volatile unsigned char*>(data)[block * BlockBytes + page * PageBytes + 3] = value; }
    DWORD Protection(std::size_t block) const {
        MEMORY_BASIC_INFORMATION memory{};
        Require(VirtualQuery(data + block * BlockBytes, &memory, sizeof(memory)) == sizeof(memory), "cannot query test memory");
        return memory.Protect;
    }
    std::set<std::size_t> Watched(std::size_t block, bool reset) const {
        std::array<void*, BlockPages> written{};
        ULONG_PTR count = written.size();
        DWORD granularity = 0;
        Require(GetWriteWatch(reset ? WRITE_WATCH_FLAG_RESET : 0, data + block * BlockBytes, BlockBytes, written.data(), &count, &granularity) == 0, "GetWriteWatch failed");
        std::set<std::size_t> pages;
        for (ULONG_PTR index = 0; index < count; ++index) pages.insert(static_cast<std::size_t>(static_cast<unsigned char*>(written[index]) - (data + block * BlockBytes)) / PageBytes);
        return pages;
    }

    unsigned char* data = nullptr;
    std::size_t bytes;
};

void CheckFaults() {
    auto& track = Track::Get();
    Raw memory(4 * BlockBytes);
    Require(track.StateOf(memory.Block(1)) == State::Unarmed && !track.Clean(memory.Block(1)), "a block starts armed");
    Require(track.Arm(memory.Block(1)), "cannot arm a block");
    Require(track.Clean(memory.Block(1)) && memory.Protection(1) == PAGE_READONLY && memory.Protection(0) == PAGE_READWRITE && memory.Protection(2) == PAGE_READWRITE, "arming did not write-protect exactly its block");
    Require(!track.Arm(memory.Block(1)), "an armed block was armed twice");
    Require(track.Counted().armed == 1, "the armed block is not counted");
    const auto before = handled.load();
    memory.Write(1, 5, 7);
    Require(handled.load() - before == 1 && memory.data[BlockBytes + 5 * PageBytes + 3] == 7, "the write to an armed block was not replayed after one fault");
    Require(track.StateOf(memory.Block(1)) == State::Written && track.Faults(memory.Block(1)) == 1 && memory.Protection(1) == PAGE_READWRITE, "the fault did not mark the block written and reopen it");
    Require(track.Counted().armed == 0 && track.Counted().faults >= 1, "the fault is not counted");
    memory.Write(1, 6, 8);
    Require(handled.load() - before == 1, "a reopened block faulted again");
    Require(memory.Watched(1, true) == (std::set<std::size_t>{5, 6}), "the write watch lost a write of a reopened block");
    Require(track.TakeWritten(memory.Block(1)) && !track.TakeWritten(memory.Block(1)) && track.StateOf(memory.Block(1)) == State::Unarmed, "a written block is not taken once");
    Require(track.HandleWrite(memory.Block(1) + 9), "a late fault on a reopened block was refused");
    DWORD previous;
    Require(VirtualProtect(memory.data + 2 * BlockBytes, BlockBytes, PAGE_READONLY, &previous) != 0, "cannot protect a block by hand");
    Require(!track.HandleWrite(memory.Block(2) + 9), "a fault on a read-only block that was never armed was handled");
    Require(VirtualProtect(memory.data + 2 * BlockBytes, BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block by hand");
    Require(track.Arm(memory.Block(2)), "cannot arm a second block");
    track.Open(memory.Block(2), memory.Block(2) + 1);
    Require(track.StateOf(memory.Block(2)) == State::Unarmed && memory.Protection(2) == PAGE_READWRITE && track.Faults(memory.Block(2)) == 0, "opening an armed block did not leave it clean and writable");
}

void CheckQuery() {
    auto& track = Track::Get();
    Raw memory(8 * BlockBytes);
    DWORD previous;
    Require(VirtualProtect(memory.data + 3 * BlockBytes, BlockBytes, PAGE_READONLY, &previous) != 0, "cannot protect a block by hand");
    Require(track.Arm(memory.Block(2)) && track.Arm(memory.Block(4)) && track.Arm(memory.Block(5)), "cannot arm blocks");
    MEMORY_BASIC_INFORMATION memoryInfo{};
    Require(track.Query(memory.Block(2) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READWRITE && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(2) && memoryInfo.RegionSize == BlockBytes, "an armed block is not reported read-write over exactly its extent");
    Require(track.Query(memory.Block(3) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READONLY && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(3) && memoryInfo.RegionSize == BlockBytes, "a read-only block between two armed blocks is not reported read-only over exactly its extent");
    Require(track.Query(memory.Block(5) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READWRITE && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(5) && memoryInfo.RegionSize == BlockBytes, "the second of two armed blocks is not reported alone");
    Require(track.Query(memory.Block(0) + 100, &memoryInfo) && memoryInfo.Protect == PAGE_READWRITE && reinterpret_cast<std::uintptr_t>(memoryInfo.BaseAddress) == memory.Block(0) && memoryInfo.RegionSize == 2 * BlockBytes, "open blocks are not reported as the host reports them");
    Require(track.Protected(memory.Block(4)) && !track.Protected(memory.Block(3)) && !track.Protected(memory.Block(6)), "the armed blocks are not the protected ones");
    Require(VirtualProtect(memory.data + 3 * BlockBytes, BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block by hand");
}

void CheckHolds() {
    auto& track = Track::Get();
    Raw memory(4 * BlockBytes);
    Require(track.Armable(memory.Block(0)), "a fresh block cannot be armed");
    Require(track.Arm(memory.Block(0)) && track.Arm(memory.Block(1)), "cannot arm blocks");
    track.Hold(memory.Block(0) + 100, memory.Block(0) + 200);
    Require(!track.Armable(memory.Block(0)) && track.StateOf(memory.Block(0)) == State::Unarmed && memory.Protection(0) == PAGE_READWRITE, "a hold did not reopen its block");
    Require(track.Clean(memory.Block(1)) && track.Armable(memory.Block(1)), "a hold touched a neighbour");
    track.Hold(memory.Block(0), memory.Block(2));
    Require(!track.Armable(memory.Block(1)) && memory.Protection(1) == PAGE_READWRITE, "a hold over two blocks left one armed");
    track.Release(memory.Block(0), memory.Block(2));
    Require(!track.Armable(memory.Block(0)) && track.Armable(memory.Block(1)), "releasing one of two holds released both, or kept the other block held");
    track.Release(memory.Block(0) + 100, memory.Block(0) + 200);
    Require(track.Armable(memory.Block(0)), "a block stays held after its last hold");
    Require(track.Arm(memory.Block(2)), "cannot arm a block");
    track.Import(memory.Block(2), memory.Block(2) + PageBytes);
    Require(!track.Armable(memory.Block(2)) && memory.Protection(2) == PAGE_READWRITE, "an import did not reopen its block for good");
    Require(track.Arm(memory.Block(3)), "cannot arm a block");
    track.Bar(memory.Block(3), memory.Block(3) + PageBytes);
    Require(!track.Armable(memory.Block(3)) && memory.Protection(3) == PAGE_READWRITE, "a barred block stays armed");
    track.Forget(memory.Block(2), memory.Block(3) + BlockBytes);
    Require(track.Armable(memory.Block(2)) && track.Armable(memory.Block(3)), "forgetting a range kept its imports or bars");
}

void CheckRace() {
    constexpr std::size_t Writers = 8;
    constexpr std::size_t Blocks = 32;
    constexpr std::size_t Pages = Blocks * BlockPages;
    constexpr std::size_t Trials = 1200;
    constexpr std::size_t Burst = 5;
    auto& track = Track::Get();
    Raw memory(Blocks * BlockBytes);
    const auto counter = [&](std::size_t page) { return reinterpret_cast<std::uint64_t*>(memory.data + page * PageBytes); };
    for (std::size_t page = 0; page < Pages; ++page) *counter(page) = 0;
    Require(ResetWriteWatch(memory.data, memory.bytes) == 0, "cannot reset the write watch");
    std::vector<std::uint64_t> shadow(Pages, 0);
    std::atomic<std::size_t> trial{0};
    std::atomic<std::size_t> finished{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> writers;
    for (std::size_t writer = 0; writer < Writers; ++writer) {
        writers.emplace_back([&, writer] {
            std::mt19937 random(static_cast<unsigned>(writer) * 7919u + 3u);
            for (std::size_t current = 1;; ++current) {
                while (trial.load(std::memory_order_acquire) != current) {
                    if (stop.load(std::memory_order_acquire)) return;
                    YieldProcessor();
                }
                for (std::size_t write = 0; write < Burst; ++write) {
                    const auto page = (random() % (Pages / Writers)) * Writers + writer;
                    auto* value = static_cast<volatile std::uint64_t*>(counter(page));
                    *value = *value + 1;
                }
                finished.fetch_add(1, std::memory_order_release);
            }
        });
    }
    std::uint64_t clean = 0;
    std::uint64_t collected = 0;
    const auto collect = [&] {
        for (std::size_t block = 0; block < Blocks; ++block) {
            if (track.Clean(memory.Block(block))) {
                ++clean;
                continue;
            }
            track.TakeWritten(memory.Block(block));
            track.Arm(memory.Block(block));
            ++collected;
            for (const auto page : memory.Watched(block, true)) shadow[block * BlockPages + page] = *static_cast<volatile std::uint64_t*>(counter(block * BlockPages + page));
        }
    };
    std::size_t lost = 0;
    const auto faultsBefore = track.Counted().faults;
    for (std::size_t current = 1; current <= Trials; ++current) {
        trial.store(current, std::memory_order_release);
        collect();
        while (finished.load(std::memory_order_acquire) != current * Writers) collect();
        collect();
        for (std::size_t page = 0; page < Pages; ++page) {
            if (shadow[page] != *counter(page)) ++lost;
        }
    }
    stop.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    std::printf("race of %zu writers against a collector that arms, resets the write watch, then reads: %zu trials, %llu faults, %llu collects answered by the armed state, %llu by GetWriteWatch, %zu last writes lost\n", Writers, Trials, static_cast<unsigned long long>(track.Counted().faults - faultsBefore), static_cast<unsigned long long>(clean), static_cast<unsigned long long>(collected), lost);
    Require(lost == 0, "a write to an armed block was lost");
    Require(refused.load() == 0, "a fault on tracked memory was refused");
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "";
    _putenv_s("APS5_PAGE_TRACK", "1");
    const auto handler = AddVectoredExceptionHandler(1, HandleFault);
    int result = 0;
    try {
        Require(Track::Get().Enabled(), "APS5_PAGE_TRACK=1 did not enable page tracking");
        if (mode == "table") {
            CheckFaults();
            CheckQuery();
            CheckHolds();
        } else if (mode == "race") {
            CheckRace();
        } else {
            std::fprintf(stderr, "usage: %s table|race\n", argv[0]);
            result = 2;
        }
        if (result == 0) std::printf("page tracking %.*s tests passed\n", static_cast<int>(mode.size()), mode.data());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    RemoveVectoredExceptionHandler(handler);
    return result;
}
