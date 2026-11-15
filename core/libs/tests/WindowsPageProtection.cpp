#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include "prx/libc/include/WindowsMappings.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {
using Mappings = GuestArena::WindowsMappings;
using Clock = std::chrono::steady_clock;
constexpr std::size_t PageBytes = 4096;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t BlockPages = BlockBytes / PageBytes;
constexpr std::size_t RunBytes = 0x200000;
constexpr std::uint32_t Open = 0;
constexpr std::uint32_t Arming = 1;
constexpr std::uint32_t Armed = 2;
constexpr std::uint32_t Opening = 3;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Tracked {
    std::uintptr_t base = 0;
    std::size_t blocks = 0;
    std::unique_ptr<std::atomic<std::uint32_t>[]> states;
};

std::atomic<Tracked*> tracked{nullptr};
std::atomic<std::uint64_t> faults{0};
std::atomic<std::uint64_t> waits{0};
std::atomic<std::uint64_t> retries{0};
std::atomic<std::uint64_t> watchedAtFault{0};
std::atomic<bool> probeWatchAtFault{false};

bool Writable(std::uintptr_t address) {
    MEMORY_BASIC_INFORMATION memory{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) != sizeof(memory)) return false;
    return memory.State == MEM_COMMIT && (memory.Protect & 0xffu) == PAGE_READWRITE;
}

LONG CALLBACK HandleFault(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    const auto* region = tracked.load(std::memory_order_acquire);
    const auto address = static_cast<std::uintptr_t>(record->ExceptionInformation[1]);
    if (region == nullptr || address < region->base || address >= region->base + region->blocks * BlockBytes) return EXCEPTION_CONTINUE_SEARCH;
    auto& state = region->states[(address - region->base) / BlockBytes];
    const auto block = address & ~(BlockBytes - 1);
    for (;;) {
        auto seen = state.load(std::memory_order_acquire);
        if (seen == Armed) {
            if (!state.compare_exchange_strong(seen, Opening, std::memory_order_acq_rel)) continue;
            if (probeWatchAtFault.load(std::memory_order_relaxed)) {
                std::array<void*, BlockPages> written{};
                ULONG_PTR count = written.size();
                DWORD granularity = 0;
                if (GetWriteWatch(0, reinterpret_cast<void*>(block), BlockBytes, written.data(), &count, &granularity) == 0) watchedAtFault.fetch_add(count, std::memory_order_relaxed);
            }
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(block), BlockBytes, PAGE_READWRITE, &previous)) return EXCEPTION_CONTINUE_SEARCH;
            faults.fetch_add(1, std::memory_order_relaxed);
            state.store(Open, std::memory_order_release);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (seen == Opening || seen == Arming) {
            waits.fetch_add(1, std::memory_order_relaxed);
            YieldProcessor();
            continue;
        }
        if (Writable(address)) {
            retries.fetch_add(1, std::memory_order_relaxed);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if (state.load(std::memory_order_acquire) != seen) continue;
        return EXCEPTION_CONTINUE_SEARCH;
    }
}

template <class TVisit>
void ForEachRun(unsigned char* base, std::size_t bytes, TVisit&& visit) {
    const auto end = reinterpret_cast<std::uintptr_t>(base) + bytes;
    for (auto cursor = reinterpret_cast<std::uintptr_t>(base); cursor < end;) {
        const auto stop = std::min(end, (cursor & ~(RunBytes - 1)) + RunBytes);
        visit(reinterpret_cast<unsigned char*>(cursor), stop - cursor);
        cursor = stop;
    }
}

void ResetWatch(unsigned char* base, std::size_t bytes) {
    ForEachRun(base, bytes, [](unsigned char* run, std::size_t size) { Require(ResetWriteWatch(run, size) == 0, "ResetWriteWatch failed: the memory is not write-watched"); });
}

class Memory {
public:
    Memory(bool resident, std::size_t size) : bytes(size), resident(resident) {
        auto& mappings = Mappings::Get();
        reservation = static_cast<unsigned char*>(mappings.Reserve(nullptr, bytes + RunBytes));
        Require(reservation != nullptr, "cannot reserve test memory");
        data = reinterpret_cast<unsigned char*>((reinterpret_cast<std::uintptr_t>(reservation) + RunBytes - 1) & ~(RunBytes - 1));
        if (resident) {
            section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, static_cast<DWORD>(bytes), nullptr);
            Require(section != nullptr, "cannot reserve the backing section of resident test memory");
            mappings.ForgetSection(section);
            mappings.Map(data, bytes, section, 0, PAGE_READWRITE, true, true);
        } else {
            mappings.Commit(data, bytes, PAGE_READWRITE, bytes, true);
        }
        for (std::size_t at = 0; at < bytes; at += PageBytes) static_cast<volatile unsigned char*>(data)[at] = 1;
        ResetWatch(data, bytes);
        region.base = reinterpret_cast<std::uintptr_t>(data);
        region.blocks = bytes / BlockBytes;
        region.states = std::make_unique<std::atomic<std::uint32_t>[]>(region.blocks);
        for (std::size_t block = 0; block < region.blocks; ++block) region.states[block].store(Open);
        tracked.store(&region, std::memory_order_release);
    }

    ~Memory() {
        tracked.store(nullptr, std::memory_order_release);
        DWORD previous;
        VirtualProtect(data, bytes, PAGE_READWRITE, &previous);
        Mappings::Get().Reset(data, bytes);
        if (section != nullptr) {
            Mappings::Get().ForgetSection(section);
            CloseHandle(section);
        }
        const auto end = reinterpret_cast<std::uintptr_t>(reservation) + bytes + RunBytes;
        for (auto cursor = reinterpret_cast<std::uintptr_t>(reservation); cursor < end;) {
            MEMORY_BASIC_INFORMATION memory{};
            if (VirtualQuery(reinterpret_cast<const void*>(cursor), &memory, sizeof(memory)) != sizeof(memory)) break;
            const auto next = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
            if (memory.State == MEM_RESERVE) VirtualFree(memory.BaseAddress, 0, MEM_RELEASE);
            cursor = next;
        }
    }

    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;

    const char* Kind() const { return resident ? "resident" : "private"; }
    unsigned char* Block(std::size_t index) const { return data + index * BlockBytes; }
    std::atomic<std::uint32_t>& State(std::size_t index) { return region.states[index]; }

    void Arm(std::size_t index) {
        State(index).store(Arming, std::memory_order_release);
        DWORD previous;
        Require(VirtualProtect(Block(index), BlockBytes, PAGE_READONLY, &previous) != 0, "cannot write-protect a block");
        State(index).store(Armed, std::memory_order_release);
    }

    void Describe() const {
        MEMORY_BASIC_INFORMATION memory{};
        Require(VirtualQuery(data, &memory, sizeof(memory)) == sizeof(memory), "cannot query test memory");
        std::array<void*, 1> page{};
        ULONG_PTR count = page.size();
        DWORD granularity = 0;
        const bool watched = GetWriteWatch(0, data, PageBytes, page.data(), &count, &granularity) == 0;
        std::printf("[memory %s] type 0x%lx state 0x%lx protect 0x%lx allocation protect 0x%lx region 0x%llx bytes, write watch %s\n", Kind(), memory.Type, memory.State, memory.Protect, memory.AllocationProtect, static_cast<unsigned long long>(memory.RegionSize), watched ? "yes" : "no");
    }

    unsigned char* data = nullptr;
    std::size_t bytes;

private:
    unsigned char* reservation = nullptr;
    HANDLE section = nullptr;
    bool resident;
    Tracked region;
};

struct Summary {
    double median = 0;
    double p99 = 0;
    double mean = 0;
    std::size_t count = 0;
};

Summary Summarize(std::vector<double> samples) {
    Summary summary;
    if (samples.empty()) return summary;
    std::sort(samples.begin(), samples.end());
    summary.count = samples.size();
    summary.median = samples[samples.size() / 2];
    summary.p99 = samples[std::min(samples.size() - 1, samples.size() * 99 / 100)];
    double total = 0;
    for (const auto sample : samples) total += sample;
    summary.mean = total / static_cast<double>(samples.size());
    return summary;
}

double Microseconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::micro>(to - from).count();
}

void Print(const char* kind, const char* what, const Summary& summary) {
    std::printf("[fault-cost %s] %s: median %.2f us, p99 %.2f us, mean %.2f us over %zu samples\n", kind, what, summary.median, summary.p99, summary.mean, summary.count);
}

std::set<std::size_t> Watched(unsigned char* base, std::size_t bytes, bool reset, ULONG_PTR capacity = 0) {
    std::set<std::size_t> pages;
    ForEachRun(base, bytes, [&](unsigned char* run, std::size_t size) {
        std::vector<void*> written(capacity != 0 ? capacity : size / PageBytes);
        ULONG_PTR count = written.size();
        DWORD granularity = 0;
        Require(GetWriteWatch(reset ? WRITE_WATCH_FLAG_RESET : 0, run, size, written.data(), &count, &granularity) == 0, "GetWriteWatch failed");
        for (ULONG_PTR index = 0; index < count; ++index) pages.insert(static_cast<std::size_t>(static_cast<unsigned char*>(written[index]) - base) / PageBytes);
    });
    return pages;
}

struct FaultCost {
    Summary single;
    Summary distinct;
    Summary same;
    Summary armBlock;
    Summary armRun;
    Summary openRun;
    Summary watchBlock;
    Summary watchRun;
};

FaultCost MeasureFaultCost(bool resident) {
    constexpr std::size_t Blocks = 256;
    constexpr std::size_t Faults = 10000;
    constexpr std::size_t Threads = 8;
    Memory memory(resident, Blocks * BlockBytes);
    memory.Describe();
    FaultCost cost;
    {
        std::vector<double> arm(Faults);
        std::vector<double> fault(Faults);
        const auto before = faults.load();
        for (std::size_t index = 0; index < Faults; ++index) {
            const auto block = index % Blocks;
            auto* target = static_cast<volatile unsigned char*>(memory.Block(block) + (index * PageBytes) % BlockBytes);
            const auto armStart = Clock::now();
            memory.Arm(block);
            const auto start = Clock::now();
            *target = static_cast<unsigned char>(index);
            const auto end = Clock::now();
            arm[index] = Microseconds(armStart, start);
            fault[index] = Microseconds(start, end);
        }
        Require(faults.load() - before == Faults, "one thread: the number of faults differs from the number of armed writes");
        cost.armBlock = Summarize(arm);
        cost.single = Summarize(fault);
        Print(memory.Kind(), "fault round trip, 1 thread (vectored handler + VirtualProtect of the 64 KiB block)", cost.single);
        Print(memory.Kind(), "arming one 64 KiB block (VirtualProtect to read-only)", cost.armBlock);
    }
    {
        std::vector<double> arm(1000);
        std::vector<double> open(1000);
        for (std::size_t index = 0; index < arm.size(); ++index) {
            auto* run = memory.data + (index % (Blocks * BlockBytes / RunBytes)) * RunBytes;
            DWORD previous;
            const auto start = Clock::now();
            Require(VirtualProtect(run, RunBytes, PAGE_READONLY, &previous) != 0, "cannot write-protect a 2 MiB run");
            const auto middle = Clock::now();
            Require(VirtualProtect(run, RunBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a 2 MiB run");
            const auto end = Clock::now();
            arm[index] = Microseconds(start, middle);
            open[index] = Microseconds(middle, end);
        }
        cost.armRun = Summarize(arm);
        cost.openRun = Summarize(open);
        Print(memory.Kind(), "arming one 2 MiB run (VirtualProtect to read-only)", cost.armRun);
        Print(memory.Kind(), "reopening one 2 MiB run (VirtualProtect to read-write)", cost.openRun);
    }
    {
        std::vector<double> block(Faults);
        std::vector<double> run(1000);
        std::array<void*, RunBytes / PageBytes> written{};
        for (std::size_t index = 0; index < block.size(); ++index) {
            ULONG_PTR count = written.size();
            DWORD granularity = 0;
            const auto start = Clock::now();
            Require(GetWriteWatch(WRITE_WATCH_FLAG_RESET, memory.Block(index % Blocks), BlockBytes, written.data(), &count, &granularity) == 0, "GetWriteWatch of a block failed");
            block[index] = Microseconds(start, Clock::now());
        }
        for (std::size_t index = 0; index < run.size(); ++index) {
            ULONG_PTR count = written.size();
            DWORD granularity = 0;
            const auto start = Clock::now();
            Require(GetWriteWatch(WRITE_WATCH_FLAG_RESET, memory.data + (index % (Blocks * BlockBytes / RunBytes)) * RunBytes, RunBytes, written.data(), &count, &granularity) == 0, "GetWriteWatch of a run failed");
            run[index] = Microseconds(start, Clock::now());
        }
        cost.watchBlock = Summarize(block);
        cost.watchRun = Summarize(run);
        Print(memory.Kind(), "today's GetWriteWatch with reset over one clean 64 KiB block", cost.watchBlock);
        Print(memory.Kind(), "today's GetWriteWatch with reset over one clean 2 MiB run", cost.watchRun);
    }
    {
        std::vector<std::vector<double>> samples(Threads);
        std::atomic<bool> go{false};
        const auto before = faults.load();
        std::vector<std::thread> threads;
        for (std::size_t thread = 0; thread < Threads; ++thread) {
            threads.emplace_back([&, thread] {
                auto& mine = samples[thread];
                mine.reserve(Faults / Threads);
                while (!go.load(std::memory_order_acquire)) YieldProcessor();
                for (std::size_t index = 0; index < Faults / Threads; ++index) {
                    const auto block = thread * (Blocks / Threads) + index % (Blocks / Threads);
                    auto* target = static_cast<volatile unsigned char*>(memory.Block(block) + (index * PageBytes) % BlockBytes);
                    memory.Arm(block);
                    const auto start = Clock::now();
                    *target = static_cast<unsigned char>(index);
                    mine.push_back(Microseconds(start, Clock::now()));
                }
            });
        }
        const auto start = Clock::now();
        go.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
        const auto wall = Microseconds(start, Clock::now());
        Require(faults.load() - before == Faults, "8 threads on distinct pages: the number of faults differs from the number of armed writes");
        std::vector<double> all;
        for (const auto& mine : samples) all.insert(all.end(), mine.begin(), mine.end());
        cost.distinct = Summarize(all);
        Print(memory.Kind(), "fault round trip, 8 threads on distinct pages", cost.distinct);
        std::printf("[fault-cost %s] 8 threads on distinct pages: %zu faults and arms in %.1f ms of wall time\n", memory.Kind(), Faults, wall / 1000.0);
    }
    {
        constexpr std::size_t Rounds = Faults / Threads;
        std::vector<std::vector<double>> samples(Threads);
        std::atomic<std::size_t> round{0};
        std::atomic<std::size_t> done{0};
        const auto before = faults.load();
        const auto waitsBefore = waits.load();
        const auto retriesBefore = retries.load();
        auto* target = static_cast<volatile unsigned char*>(memory.Block(3) + 5 * PageBytes);
        std::vector<std::thread> threads;
        for (std::size_t thread = 0; thread < Threads; ++thread) {
            threads.emplace_back([&, thread] {
                auto& mine = samples[thread];
                mine.reserve(Rounds);
                for (std::size_t current = 1; current <= Rounds; ++current) {
                    while (round.load(std::memory_order_acquire) != current) YieldProcessor();
                    const auto start = Clock::now();
                    *target = static_cast<unsigned char>(thread);
                    mine.push_back(Microseconds(start, Clock::now()));
                    done.fetch_add(1, std::memory_order_release);
                }
            });
        }
        for (std::size_t current = 1; current <= Rounds; ++current) {
            memory.Arm(3);
            round.store(current, std::memory_order_release);
            while (done.load(std::memory_order_acquire) != current * Threads) YieldProcessor();
        }
        for (auto& thread : threads) thread.join();
        std::vector<double> all;
        for (const auto& mine : samples) all.insert(all.end(), mine.begin(), mine.end());
        cost.same = Summarize(all);
        Print(memory.Kind(), "write to one armed page raced by 8 threads", cost.same);
        std::printf("[fault-cost %s] 8 threads on the same page: %zu writes over %zu arms, %llu reopenings, %llu handler waits, %llu faults on a block another thread had reopened\n", memory.Kind(), all.size(), Rounds, static_cast<unsigned long long>(faults.load() - before), static_cast<unsigned long long>(waits.load() - waitsBefore), static_cast<unsigned long long>(retries.load() - retriesBefore));
        Require(faults.load() - before == Rounds, "8 threads on the same page: a block was reopened more or less than once per arm");
    }
    return cost;
}

int FaultCostExperiment() {
    bool viable = true;
    for (const bool resident : {false, true}) {
        const auto cost = MeasureFaultCost(resident);
        const double cycle = cost.single.median + cost.armBlock.median + cost.watchBlock.median;
        const double breakEven = cost.watchBlock.median > 0 ? cycle / cost.watchBlock.median : 0;
        std::printf("[fault-cost %s] one dirty cycle (fault + rearm + GetWriteWatch) costs %.2f us against %.2f us for one GetWriteWatch: a block pays for itself when it stays clean for more than %.1f collects between two writes\n", resident ? "resident" : "private", cycle, cost.watchBlock.median, breakEven);
        if (cost.single.median > 50 || cost.distinct.median > 500 || cost.same.median > 5000) viable = false;
    }
    std::printf("verdict fault-cost: %s\n", viable ? "safe under a condition: only blocks that stay clean for several collects may be armed, a block written every frame must stay on the write watch" : "impossible: the median fault costs more than 50 us on one thread, 500 us on 8 threads or 5 ms when 8 threads race for one page");
    return 0;
}

int WriteWatchExperiment() {
    bool consistent = true;
    const auto check = [&](bool condition, const char* kind, const char* what) {
        std::printf("[write-watch %s] %s: %s\n", kind, what, condition ? "yes" : "NO");
        if (!condition) consistent = false;
    };
    for (const bool resident : {false, true}) {
        Memory memory(resident, 64 * BlockBytes);
        memory.Describe();
        const auto kind = memory.Kind();
        auto* block = memory.Block(1);
        const auto write = [&](std::size_t page, unsigned char value) { static_cast<volatile unsigned char*>(block)[page * PageBytes + 7] = value; };
        const std::set<std::size_t> none;
        memory.Arm(1);
        check(Watched(block, BlockBytes, false) == none, kind, "arming a clean block reports no written page");
        auto before = faults.load();
        probeWatchAtFault.store(true);
        const auto watchedBefore = watchedAtFault.load();
        write(5, 1);
        probeWatchAtFault.store(false);
        check(faults.load() - before == 1 && block[5 * PageBytes + 7] == 1, kind, "the write to an armed page faulted once and was replayed");
        std::printf("[write-watch %s] pages the write watch held inside the handler, before the block was reopened: %llu\n", kind, static_cast<unsigned long long>(watchedAtFault.load() - watchedBefore));
        check(Watched(block, BlockBytes, false) == std::set<std::size_t>{5}, kind, "after the replayed write GetWriteWatch reports exactly the written page");
        check(Watched(block, BlockBytes, true) == std::set<std::size_t>{5} && Watched(block, BlockBytes, false) == none, kind, "a reset collect returns the page once");
        write(9, 2);
        check(Watched(block, BlockBytes, false) == std::set<std::size_t>{9}, kind, "a write to the reopened block is watched without a fault");
        memory.Arm(1);
        check(Watched(block, BlockBytes, false) == std::set<std::size_t>{9}, kind, "arming keeps the pages already written");
        DWORD previous;
        Require(VirtualProtect(block, BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block");
        memory.State(1).store(Open);
        check(Watched(block, BlockBytes, false) == std::set<std::size_t>{9}, kind, "reopening without a write adds no page");
        memory.Arm(1);
        Require(ResetWriteWatch(block, BlockBytes) == 0, "ResetWriteWatch failed on an armed block");
        check(Watched(block, BlockBytes, false) == none, kind, "ResetWriteWatch works on an armed block");
        before = faults.load();
        write(0, 3);
        write(15, 4);
        check(faults.load() - before == 1 && Watched(block, BlockBytes, true) == (std::set<std::size_t>{0, 15}), kind, "two pages written after one fault are both reported");
        for (std::size_t page = 0; page < BlockPages; ++page) write(page, 5);
        const auto partial = Watched(block, BlockBytes, true, 4);
        const auto rest = Watched(block, BlockBytes, true);
        check(partial == (std::set<std::size_t>{0, 1, 2, 3}) && rest.size() == BlockPages - 4 && *rest.begin() == 4, kind, "a reset collect into a short buffer resets only the pages it returned");
        std::mt19937 random(12345);
        std::size_t mismatches = 0;
        for (std::size_t cycle = 0; cycle < 2000; ++cycle) {
            const auto index = 2 + random() % 60;
            const auto page = random() % BlockPages;
            memory.Arm(index);
            static_cast<volatile unsigned char*>(memory.Block(index))[page * PageBytes] = static_cast<unsigned char>(cycle);
            if (Watched(memory.Block(index), BlockBytes, true) != std::set<std::size_t>{page}) ++mismatches;
        }
        std::printf("[write-watch %s] 2000 arm, fault, collect cycles: %zu collects that differ from the written page\n", kind, mismatches);
        if (mismatches != 0) consistent = false;

        constexpr std::size_t Writers = 4;
        constexpr std::size_t Trials = 1500;
        constexpr std::size_t Burst = 6;
        constexpr std::size_t Blocks = 32;
        constexpr std::size_t Pages = Blocks * BlockPages;
        auto* base = memory.Block(8);
        const auto counter = [&](std::size_t page) { return reinterpret_cast<std::uint64_t*>(base + page * PageBytes); };
        for (std::size_t page = 0; page < Pages; ++page) *counter(page) = 0;
        ResetWatch(base, Blocks * BlockBytes);
        std::vector<std::uint64_t> shadow(Pages, 0);
        std::atomic<std::size_t> trial{0};
        std::atomic<std::size_t> finished{0};
        std::atomic<bool> stop{false};
        std::vector<std::thread> writers;
        for (std::size_t writer = 0; writer < Writers; ++writer) {
            writers.emplace_back([&, writer] {
                std::mt19937 own(static_cast<unsigned>(writer) * 7919u + 1u);
                for (std::size_t current = 1;; ++current) {
                    while (trial.load(std::memory_order_acquire) != current) {
                        if (stop.load(std::memory_order_acquire)) return;
                        YieldProcessor();
                    }
                    for (std::size_t write = 0; write < Burst; ++write) {
                        const auto page = (own() % (Pages / Writers)) * Writers + writer;
                        auto* value = static_cast<volatile std::uint64_t*>(counter(page));
                        *value = *value + 1;
                    }
                    finished.fetch_add(1, std::memory_order_release);
                }
            });
        }
        std::uint64_t clean = 0;
        std::uint64_t collected = 0;
        std::uint64_t busy = 0;
        const auto collect = [&] {
            for (std::size_t index = 0; index < Blocks; ++index) {
                auto& state = memory.State(8 + index);
                const auto seen = state.load(std::memory_order_acquire);
                if (seen == Armed) {
                    ++clean;
                    continue;
                }
                if (seen == Opening) {
                    ++busy;
                    for (std::size_t page = 0; page < BlockPages; ++page) shadow[index * BlockPages + page] = *static_cast<volatile std::uint64_t*>(counter(index * BlockPages + page));
                    continue;
                }
                ++collected;
                memory.Arm(8 + index);
                for (const auto page : Watched(base + index * BlockBytes, BlockBytes, true)) shadow[index * BlockPages + page] = *static_cast<volatile std::uint64_t*>(counter(index * BlockPages + page));
            }
        };
        std::size_t lost = 0;
        const auto faultsBefore = faults.load();
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
        std::printf("[write-watch %s] race of %zu writers against a collector that arms, resets, then reads: %zu trials, %llu faults, %llu collects answered by the armed state alone, %llu by GetWriteWatch, %llu met a block being reopened; pages whose last write was lost: %zu\n", kind, Writers, Trials, static_cast<unsigned long long>(faults.load() - faultsBefore), static_cast<unsigned long long>(clean), static_cast<unsigned long long>(collected), static_cast<unsigned long long>(busy), lost);
        if (lost != 0) consistent = false;
    }
    std::printf("verdict write-watch: %s\n", consistent ? "safe: a page can be write-watched and write-protected at once, the replayed write is watched, and arming before the reset collect loses no write" : "impossible: the write watch and the page protection disagree");
    return consistent ? 0 : 1;
}

struct TemporaryFile {
    std::wstring path;
    HANDLE handle = INVALID_HANDLE_VALUE;

    TemporaryFile(std::size_t bytes, DWORD flags) {
        wchar_t directory[MAX_PATH];
        wchar_t name[MAX_PATH];
        Require(GetTempPathW(MAX_PATH, directory) != 0 && GetTempFileNameW(directory, L"aps", 0, name) != 0, "cannot name a temporary file");
        path = name;
        const HANDLE writer = CreateFileW(name, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        Require(writer != INVALID_HANDLE_VALUE, "cannot create a temporary file");
        std::vector<unsigned char> content(1u << 20, 0xa5);
        for (std::size_t done = 0; done < bytes; done += content.size()) {
            DWORD written = 0;
            Require(WriteFile(writer, content.data(), static_cast<DWORD>(content.size()), &written, nullptr) != 0 && written == content.size(), "cannot fill a temporary file");
        }
        FlushFileBuffers(writer);
        CloseHandle(writer);
        handle = CreateFileW(name, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, flags, nullptr);
        Require(handle != INVALID_HANDLE_VALUE, "cannot reopen a temporary file");
    }

    ~TemporaryFile() {
        CloseHandle(handle);
        DeleteFileW(path.c_str());
    }
};

bool Filled(const unsigned char* bytes, std::size_t count, unsigned char value) {
    return std::all_of(bytes, bytes + count, [value](unsigned char byte) { return byte == value; });
}

int KernelWriteExperiment() {
    bool silent = false;
    bool lateSilent = false;
    Memory memory(false, 576 * BlockBytes);
    memory.Describe();
    const auto prepare = [&](std::size_t index) {
        DWORD previous;
        Require(VirtualProtect(memory.Block(index), BlockBytes, PAGE_READWRITE, &previous) != 0, "cannot reopen a block");
        memory.State(index).store(Open);
        std::memset(memory.Block(index), 0x11, BlockBytes);
        Require(ResetWriteWatch(memory.Block(index), BlockBytes) == 0, "ResetWriteWatch failed");
    };
    const auto report = [&](const char* what, bool succeeded, DWORD error, std::uint64_t transferred, std::uint64_t handled, bool changed, std::size_t watchedPages) {
        std::printf("[kernel-writes] %s: %s, error %lu, %llu bytes transferred, %llu faults handled, destination bytes %s, %zu pages on the write watch\n", what, succeeded ? "succeeded" : "failed", error, static_cast<unsigned long long>(transferred), static_cast<unsigned long long>(handled), changed ? "CHANGED" : "unchanged", watchedPages);
    };
    {
        TemporaryFile file(1u << 20, FILE_ATTRIBUTE_NORMAL);
        prepare(1);
        DWORD read = 0;
        SetLastError(0);
        const bool open = ReadFile(file.handle, memory.Block(1), PageBytes, &read, nullptr) != 0;
        report("ReadFile into an open page (today's behaviour)", open, open ? 0 : GetLastError(), read, 0, !Filled(memory.Block(1), PageBytes, 0x11), Watched(memory.Block(1), BlockBytes, false).size());
        prepare(1);
        memory.Arm(1);
        auto before = faults.load();
        read = 0;
        SetLastError(0);
        const bool armed = ReadFile(file.handle, memory.Block(1), PageBytes, &read, nullptr) != 0;
        const auto error = armed ? 0 : GetLastError();
        const bool changed = !Filled(memory.Block(1), PageBytes, 0x11);
        report("ReadFile into an armed page", armed, error, read, faults.load() - before, changed, Watched(memory.Block(1), BlockBytes, false).size());
        if (armed || changed) silent = true;
        prepare(2);
        prepare(3);
        memory.Arm(3);
        before = faults.load();
        read = 0;
        SetLastError(0);
        SetFilePointer(file.handle, 0, nullptr, FILE_BEGIN);
        const bool straddling = ReadFile(file.handle, memory.Block(3) - PageBytes, 2 * PageBytes, &read, nullptr) != 0;
        const auto straddlingError = straddling ? 0 : GetLastError();
        const bool openPart = !Filled(memory.Block(3) - PageBytes, PageBytes, 0x11);
        const bool armedPart = !Filled(memory.Block(3), PageBytes, 0x11);
        report("ReadFile across an open page then an armed page", straddling, straddlingError, read, faults.load() - before, armedPart, Watched(memory.Block(2), 2 * BlockBytes, false).size());
        std::printf("[kernel-writes] ReadFile across an open page then an armed page: the open page was %s\n", openPart ? "written although the call failed" : "left unchanged");
        if (straddling || armedPart) silent = true;
    }
    {
        prepare(4);
        memory.Arm(4);
        std::array<unsigned char, PageBytes> source;
        source.fill(0x77);
        const auto before = faults.load();
        SIZE_T copied = 0;
        SetLastError(0);
        const bool succeeded = ReadProcessMemory(GetCurrentProcess(), source.data(), memory.Block(4), source.size(), &copied) != 0;
        const auto error = succeeded ? 0 : GetLastError();
        const bool changed = !Filled(memory.Block(4), PageBytes, 0x11);
        report("ReadProcessMemory into an armed page", succeeded, error, copied, faults.load() - before, changed, Watched(memory.Block(4), BlockBytes, false).size());
        if (succeeded || changed) silent = true;
    }
    {
        WSADATA data{};
        Require(WSAStartup(MAKEWORD(2, 2), &data) == 0, "cannot start Winsock");
        const SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        int length = sizeof(address);
        Require(listener != INVALID_SOCKET && bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0 && listen(listener, 1) == 0 && getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) == 0, "cannot listen on the loopback interface");
        const SOCKET client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        Require(client != INVALID_SOCKET && connect(client, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0, "cannot connect on the loopback interface");
        const SOCKET server = accept(listener, nullptr, nullptr);
        Require(server != INVALID_SOCKET, "cannot accept on the loopback interface");
        std::array<char, 512> message;
        message.fill(0x33);
        Require(send(client, message.data(), static_cast<int>(message.size()), 0) == static_cast<int>(message.size()), "cannot send on the loopback interface");
        Sleep(50);
        prepare(5);
        memory.Arm(5);
        const auto before = faults.load();
        WSASetLastError(0);
        const int received = recv(server, reinterpret_cast<char*>(memory.Block(5)), static_cast<int>(message.size()), 0);
        const auto error = received < 0 ? WSAGetLastError() : 0;
        const bool changed = !Filled(memory.Block(5), PageBytes, 0x11);
        report("recv into an armed page", received >= 0, static_cast<DWORD>(error), received > 0 ? static_cast<std::uint64_t>(received) : 0, faults.load() - before, changed, Watched(memory.Block(5), BlockBytes, false).size());
        if (received >= 0 || changed) silent = true;
        std::array<char, 512> again{};
        u_long nonBlocking = 1;
        ioctlsocket(server, FIONBIO, &nonBlocking);
        const int retried = recv(server, again.data(), static_cast<int>(again.size()), 0);
        std::printf("[kernel-writes] recv into an armed page: a second recv into host memory returned %d bytes (%s)\n", retried, retried == static_cast<int>(message.size()) ? "the failed call consumed nothing" : "the failed call consumed the data");
        closesocket(server);
        closesocket(client);
        closesocket(listener);
        WSACleanup();
    }
    {
        const auto name = L"\\\\.\\pipe\\aps5-page-protection-" + std::to_wstring(GetCurrentProcessId());
        const HANDLE server = CreateNamedPipeW(name.c_str(), PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED, PIPE_TYPE_BYTE | PIPE_WAIT, 1, 65536, 65536, 0, nullptr);
        Require(server != INVALID_HANDLE_VALUE, "cannot create a named pipe");
        const HANDLE client = CreateFileW(name.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        Require(client != INVALID_HANDLE_VALUE, "cannot open the named pipe");
        prepare(6);
        OVERLAPPED overlapped{};
        overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const bool immediate = ReadFile(server, memory.Block(6), 512, nullptr, &overlapped) != 0;
        Require(!immediate && GetLastError() == ERROR_IO_PENDING, "the overlapped pipe read did not stay pending");
        memory.Arm(6);
        const auto before = faults.load();
        std::array<unsigned char, 512> message;
        message.fill(0x44);
        DWORD written = 0;
        Require(WriteFile(client, message.data(), static_cast<DWORD>(message.size()), &written, nullptr) != 0, "cannot write to the named pipe");
        DWORD transferred = 0;
        SetLastError(0);
        const bool completed = GetOverlappedResult(server, &overlapped, &transferred, TRUE) != 0;
        const auto error = completed ? 0 : GetLastError();
        const bool changed = !Filled(memory.Block(6), PageBytes, 0x11);
        report("overlapped pipe read issued on an open page, armed before the bytes arrive", completed, error, transferred, faults.load() - before, changed, Watched(memory.Block(6), BlockBytes, false).size());
        if (changed) lateSilent = true;
        if (completed && !changed) silent = true;
        CloseHandle(overlapped.hEvent);
        CloseHandle(client);
        CloseHandle(server);
    }
    {
        constexpr std::size_t First = 64;
        constexpr std::size_t Count = 512;
        constexpr std::size_t Bytes = Count * BlockBytes;
        TemporaryFile file(Bytes, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED);
        DWORD previous;
        std::memset(memory.Block(First), 0x11, Bytes);
        ResetWatch(memory.Block(First), Bytes);
        OVERLAPPED overlapped{};
        overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        const bool immediate = ReadFile(file.handle, memory.Block(First), static_cast<DWORD>(Bytes), nullptr, &overlapped) != 0;
        const bool pending = !immediate && GetLastError() == ERROR_IO_PENDING;
        bool protectedAll = false;
        bool stillPending = false;
        if (pending) {
            for (std::size_t index = First; index < First + Count; ++index) memory.State(index).store(Arming);
            protectedAll = VirtualProtect(memory.Block(First), Bytes, PAGE_READONLY, &previous) != 0;
            stillPending = !HasOverlappedIoCompleted(&overlapped);
            for (std::size_t index = First; index < First + Count; ++index) memory.State(index).store(protectedAll ? Armed : Open);
        }
        const auto before = faults.load();
        DWORD transferred = 0;
        SetLastError(0);
        const bool completed = GetOverlappedResult(file.handle, &overlapped, &transferred, TRUE) != 0;
        const auto error = completed ? 0 : GetLastError();
        const bool changed = !Filled(memory.Block(First) + Bytes - BlockBytes, BlockBytes, 0x11);
        if (!pending || !stillPending) std::printf("[kernel-writes] unbuffered overlapped file read: %s, the case could not be exercised\n", !pending ? "it did not stay pending" : "it completed before the pages were armed");
        else {
            report("unbuffered overlapped file read issued on open pages, armed while it was pending", completed, error, transferred, faults.load() - before, changed, Watched(memory.Block(First), Bytes, false).size());
            std::printf("[kernel-writes] unbuffered overlapped file read: VirtualProtect of the pages locked by the pending read %s\n", protectedAll ? "succeeded" : "failed");
            if (changed) lateSilent = true;
        }
        CloseHandle(overlapped.hEvent);
    }
    {
        prepare(8);
        memory.Arm(8);
        auto before = faults.load();
        GetSystemTimeAsFileTime(reinterpret_cast<FILETIME*>(memory.Block(8)));
        const auto first = faults.load() - before;
        std::printf("[kernel-writes] GetSystemTimeAsFileTime into an armed page (a user-mode store): %llu fault handled, value %s\n", static_cast<unsigned long long>(first), Filled(memory.Block(8), sizeof(FILETIME), 0x11) ? "NOT written" : "written");
        if (first != 1 || Filled(memory.Block(8), sizeof(FILETIME), 0x11)) silent = true;
        prepare(9);
        memory.Arm(9);
        before = faults.load();
        const bool counted = QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(memory.Block(9))) != 0;
        const auto second = faults.load() - before;
        std::printf("[kernel-writes] QueryPerformanceCounter into an armed page: %s, %llu fault handled, value %s\n", counted ? "succeeded" : "failed", static_cast<unsigned long long>(second), Filled(memory.Block(9), sizeof(LARGE_INTEGER), 0x11) ? "NOT written" : "written");
        if (!counted || second != 1) silent = true;
    }
    if (silent) std::printf("verdict kernel-writes: impossible: a system call wrote into an armed page, or failed, without a visible error\n");
    else if (lateSilent) std::printf("verdict kernel-writes: safe under a condition: a synchronous system call into an armed page fails with a visible error and writes nothing, but a read that is already pending when its destination is armed lands without a fault, so a block must never be armed while a host write is open on it\n");
    else std::printf("verdict kernel-writes: safe: every system call into an armed page fails with a visible error and writes nothing\n");
    return silent ? 1 : 0;
}

}

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "";
    const auto handler = AddVectoredExceptionHandler(1, HandleFault);
    int result = 2;
    try {
        if (mode == "fault-cost") result = FaultCostExperiment();
        else if (mode == "write-watch") result = WriteWatchExperiment();
        else if (mode == "kernel-writes") result = KernelWriteExperiment();
        else std::fprintf(stderr, "usage: %s fault-cost|write-watch|kernel-writes\n", argv[0]);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    RemoveVectoredExceptionHandler(handler);
    return result;
}
