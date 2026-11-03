#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/TracyZone.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include "prx/libc/include/general/ThreadName.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#include <tlhelp32.h>
#endif

namespace AgcDriver::DriverDetail {

std::uint64_t WorkerAffinityMask() {
    static const std::uint64_t mask = [] {
        if (std::getenv("APS5_NO_WORKER_AFFINITY") != nullptr) return std::uint64_t{0};
        const auto requested = CpuTopology::MaskFromEnvironment("APS5_WORKER_AFFINITY_MASK");
        if (requested != 0) return requested;
#ifndef __linux__
        if (std::getenv("APS5_WORKER_AFFINITY") == nullptr) return std::uint64_t{0};
#endif
        const auto& layout = CpuTopology::Get();
        return layout.hybrid ? layout.performant : std::uint64_t{0};
    }();
    return mask;
}

void PinWorkerThread(const char* role) {
    TracyZone::NameThread(role);
#ifdef _WIN32
    const HRESULT nameResult = ThreadName::SetCurrent(role);
    if (FAILED(nameResult)) std::fprintf(stderr, "[thread-name] could not name %s (0x%08lx)\n", role, static_cast<unsigned long>(nameResult));
#endif
    const auto mask = WorkerAffinityMask();
    if (mask == 0) return;
    static std::once_flag summary;
    std::call_once(summary, [mask] {
        const auto& layout = CpuTopology::Get();
        std::fprintf(stderr, "[affinity] queue workers and presenter -> 0x%llx (hybrid=%d efficient=0x%llx performant=0x%llx process=0x%llx)\n", static_cast<unsigned long long>(mask), layout.hybrid ? 1 : 0, static_cast<unsigned long long>(layout.efficient), static_cast<unsigned long long>(layout.performant), static_cast<unsigned long long>(layout.process));
    });
    CpuTopology::PinTraced(role, nullptr, mask);
}

namespace {

std::array<std::atomic<unsigned long>, 3> criticalThreads{};

#ifdef _WIN32
struct Placement {
    std::uint64_t process = 0;
    std::array<std::uint64_t, 3> critical{};
    std::uint64_t rest = 0;
};

// The physical cores of processor group 0 (each a mask of its logical CPUs within the process mask), in
// order of their lowest CPU.
std::vector<std::uint64_t> physicalCores(std::uint64_t process) {
    std::vector<std::uint64_t> cores;
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || length == 0) return cores;
    std::vector<unsigned char> buffer(length);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()), &length)) return cores;
    for (DWORD offset = 0; offset + 8 <= length;) {
        const auto* entry = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
        if (entry->Size == 0 || offset + entry->Size > length) break;
        if (entry->Relationship == RelationProcessorCore && entry->Processor.GroupCount >= 1 && entry->Processor.GroupMask[0].Group == 0) {
            const auto mask = static_cast<std::uint64_t>(entry->Processor.GroupMask[0].Mask) & process;
            if (mask != 0) cores.push_back(mask);
        }
        offset += entry->Size;
    }
    std::sort(cores.begin(), cores.end(), [](std::uint64_t a, std::uint64_t b) { return (a & (~a + 1)) < (b & (~b + 1)); });
    return cores;
}

// The placement, computed once; process == 0 when pinning does not apply on this machine.
const Placement& placement() {
    static const Placement result = [] {
        Placement placed{};
        const auto& layout = CpuTopology::Get();
        const auto cores = physicalCores(layout.process);
        if (layout.process == 0 || layout.hybrid || cores.size() < 6) {
            std::fprintf(stderr, "[affinity] thread pinning off: %zu physical cores%s\n", cores.size(), layout.hybrid ? ", hybrid CPU" : "");
            return placed;
        }
        std::uint64_t taken = 0;
        for (std::size_t i = 0; i < placed.critical.size(); ++i) {
            const auto core = cores[cores.size() * (i + 1) / 4];
            placed.critical[i] = core & (~core + 1);
            taken |= core;
        }
        placed.process = layout.process;
        placed.rest = layout.process & ~taken;
        std::fprintf(stderr, "[affinity] thread pinning placement: queue-0 worker 0x%llx, committer 0x%llx, preparer 0x%llx, other threads 0x%llx (process 0x%llx, %zu physical cores)\n", static_cast<unsigned long long>(placed.critical[0]), static_cast<unsigned long long>(placed.critical[1]), static_cast<unsigned long long>(placed.critical[2]), static_cast<unsigned long long>(placed.rest), static_cast<unsigned long long>(placed.process), cores.size());
        return placed;
    }();
    return result;
}

// Each placed thread's mask from before its first placement, restored when pinning turns off. Only the
// queue-0 worker (FollowThreadPinning) touches it.
std::unordered_map<DWORD, DWORD_PTR> originalMasks;

// Pinning: every thread of the process on its CPUs (the critical ones alone, the rest on the remaining
// cores). Not pinning: every thread placed before back on its original mask. Returns the threads changed.
unsigned placeThreads(const Placement& placed, bool pinning) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 0;
    std::array<unsigned long, 3> critical{};
    for (std::size_t i = 0; i < critical.size(); ++i) critical[i] = criticalThreads[i].load(std::memory_order_acquire);
    const DWORD process = GetCurrentProcessId();
    unsigned changed = 0;
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != process) continue;
        const DWORD id = entry.th32ThreadID;
        const auto original = originalMasks.find(id);
        if (!pinning && original == originalMasks.end()) continue;
        std::uint64_t mask = placed.rest;
        for (std::size_t i = 0; i < critical.size(); ++i) {
            if (critical[i] == id) mask = placed.critical[i];
        }
        const auto target = pinning ? static_cast<DWORD_PTR>(mask) : original->second;
        HANDLE thread = OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, FALSE, id);
        if (thread == nullptr) continue;
        const auto previous = SetThreadAffinityMask(thread, target);
        CloseHandle(thread);
        if (previous == 0) continue;
        if (pinning && original == originalMasks.end()) originalMasks.emplace(id, previous);
        if (previous != target) ++changed;
    }
    CloseHandle(snapshot);
    if (!pinning) originalMasks.clear();
    return changed;
}
#endif

}

void NoteCriticalThread(CriticalThread role) {
#ifdef _WIN32
    criticalThreads[static_cast<std::size_t>(role)].store(GetCurrentThreadId(), std::memory_order_release);
#else
    static_cast<void>(role);
#endif
}

void FollowThreadPinning() {
#ifdef _WIN32
    const bool pinning = Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::ThreadPinning);
    static bool pinned = false;
    static std::chrono::steady_clock::time_point lastSweep{};
    // Unset and never applied: every thread keeps the mask the title and the driver gave it.
    if (!pinning && !pinned) return;
    const auto now = std::chrono::steady_clock::now();
    if (pinning && pinned && now - lastSweep < std::chrono::seconds(1)) return;
    const auto& placed = placement();
    if (placed.process == 0) return;
    const auto threads = placeThreads(placed, pinning);
    if (pinning != pinned) std::fprintf(stderr, "[affinity] thread pinning %s (%u threads changed)\n", pinning ? "on" : "off", threads);
    pinned = pinning;
    lastSweep = now;
#endif
}

}
