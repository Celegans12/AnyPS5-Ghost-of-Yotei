#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#include "prx/libc/include/FaultStorm.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace FaultStorm {

namespace {

constexpr DWORD TrapFlag = 0x100;
constexpr DWORD64 SingleStepStatus = 0x4000;

std::atomic<bool> g_active{false};
std::atomic<DWORD> g_pending{0};
std::atomic<unsigned> g_remaining{0};
unsigned g_burst = 1;
bool g_clobber = true;
bool g_clobberGuestOnly = false;
constexpr std::size_t RedZoneBytes = 0x80;
std::atomic<unsigned long long> g_delivered{0};
std::atomic<unsigned long long> g_stray{0};

std::wstring g_clobberImage;
std::uintptr_t g_clobberBegin = 0;
std::uintptr_t g_clobberEnd = 0;
std::atomic<std::uintptr_t> g_clobberLow{0};
std::atomic<std::uintptr_t> g_clobberHigh{0};

struct Range {
    std::uintptr_t Begin;
    std::uintptr_t End;
};

std::vector<Range> GuestCode() {
    std::vector<Range> ranges;
    HMODULE modules[2048];
    DWORD needed = 0;
    const auto process = GetCurrentProcess();
    if (!K32EnumProcessModules(process, modules, sizeof(modules), &needed)) return ranges;
    const auto count = std::min<DWORD>(needed / sizeof(HMODULE), static_cast<DWORD>(std::size(modules)));
    const auto main = GetModuleHandleW(nullptr);
    constexpr wchar_t Suffix[] = L".guest.prx";
    constexpr std::size_t SuffixLength = std::size(Suffix) - 1;
    for (DWORD index = 0; index < count; ++index) {
        wchar_t name[MAX_PATH];
        const auto length = GetModuleFileNameW(modules[index], name, MAX_PATH);
        const bool guest = modules[index] == main || (length > SuffixLength && _wcsicmp(name + length - SuffixLength, Suffix) == 0);
        if (!guest) continue;
        MODULEINFO info{};
        if (!K32GetModuleInformation(process, modules[index], &info, sizeof(info))) continue;
        const auto base = reinterpret_cast<std::uintptr_t>(info.lpBaseOfDll);
        ranges.push_back({base, base + info.SizeOfImage});
        if (!g_clobberImage.empty()) {
            const wchar_t* file = name;
            for (DWORD at = 0; at < length; ++at) if (name[at] == L'\\' || name[at] == L'/') file = name + at + 1;
            if (_wcsicmp(file, g_clobberImage.c_str()) == 0) {
                g_clobberLow.store(base + g_clobberBegin, std::memory_order_relaxed);
                g_clobberHigh.store(base + g_clobberEnd, std::memory_order_release);
            }
        }
    }
    return ranges;
}

bool Inside(const std::vector<Range>& ranges, const std::uintptr_t address) {
    return std::any_of(ranges.begin(), ranges.end(), [&](const Range& range) { return address >= range.Begin && address < range.End; });
}

std::atomic<const std::vector<Range>*> g_guestCode{nullptr};

long long Now() {
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return value.QuadPart;
}

DWORD WINAPI Run(LPVOID parameter) {
    const auto rate = static_cast<long long>(reinterpret_cast<std::uintptr_t>(parameter));
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    const auto second = frequency.QuadPart;
    const auto period = std::max<long long>(1, second / rate);
    const auto self = GetCurrentThreadId();
    const auto process = GetCurrentProcessId();
    std::mt19937_64 random(static_cast<std::uint64_t>(Now()));
    std::map<DWORD, HANDLE> threads;
    std::map<DWORD, ULONG64> cycles;
    std::vector<std::pair<DWORD, HANDLE>> list;
    std::vector<ULONG64> weights;
    ULONG64 totalWeight = 0;
    std::vector<Range> code;
    unsigned long long tries = 0, injected = 0, outside = 0, slow = 0, misses = 0;
    long long nextModules = 0, nextThreads = 0, nextReport = Now() + 10 * second, nextStep = Now();
    for (;;) {
        const auto now = Now();
        if (now >= nextModules) {
            const auto before = code.size();
            code = GuestCode();
            if (code.size() != before) {
                std::fprintf(stderr, "[fault-storm] %zu guest code images\n", code.size());
                std::fflush(stderr);
                g_guestCode.store(new std::vector<Range>(code), std::memory_order_release);
            }
            nextModules = now + 2 * second;
        }
        if (now >= nextThreads) {
            std::map<DWORD, HANDLE> seen;
            const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if (snapshot != INVALID_HANDLE_VALUE) {
                THREADENTRY32 entry{};
                entry.dwSize = sizeof(entry);
                for (auto ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
                    if (entry.th32OwnerProcessID != process || entry.th32ThreadID == self) continue;
                    if (const auto known = threads.find(entry.th32ThreadID); known != threads.end()) {
                        seen.emplace(*known);
                        threads.erase(known);
                        continue;
                    }
                    if (const auto handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID)) seen.emplace(entry.th32ThreadID, handle);
                }
                CloseHandle(snapshot);
            }
            for (const auto& [id, handle] : threads) CloseHandle(handle);
            threads = std::move(seen);
            list.clear();
            weights.clear();
            totalWeight = 0;
            std::map<DWORD, ULONG64> nextCycles;
            for (const auto& [id, handle] : threads) {
                ULONG64 used = 0;
                if (!QueryThreadCycleTime(handle, &used)) continue;
                const auto last = cycles.find(id);
                const auto delta = last == cycles.end() || used < last->second ? 0 : used - last->second;
                nextCycles.emplace(id, used);
                if (delta == 0) continue;
                list.emplace_back(id, handle);
                weights.push_back(delta);
                totalWeight += delta;
            }
            cycles = std::move(nextCycles);
            nextThreads = now + second / 5;
        }
        if (now >= nextReport) {
            std::fprintf(stderr, "[fault-storm] 10 s: %llu tries, %llu landings in guest code (%llu tries elsewhere), %llu steps taken, %llu stray steps, %llu slow bursts; %zu running threads\n",
                tries, injected, outside, g_delivered.exchange(0), g_stray.exchange(0), slow, list.size());
            std::fflush(stderr);
            tries = injected = outside = slow = 0;
            nextReport += 10 * second;
        }
        if (now < nextStep) {
            SwitchToThread();
            continue;
        }
        if (list.empty() || code.empty() || totalWeight == 0) {
            Sleep(10);
            continue;
        }
        auto pick = random() % totalWeight;
        std::size_t chosen = 0;
        while (chosen + 1 < weights.size() && pick >= weights[chosen]) pick -= weights[chosen++];
        const auto [id, handle] = list[chosen];
        ++tries;
        if (SuspendThread(handle) == static_cast<DWORD>(-1)) continue;
        alignas(16) CONTEXT context{};
        context.ContextFlags = CONTEXT_CONTROL;
        bool stepped = false;
        if (GetThreadContext(handle, &context) && Inside(code, static_cast<std::uintptr_t>(context.Rip)) && (context.EFlags & TrapFlag) == 0) {
            g_remaining.store(g_burst, std::memory_order_relaxed);
            g_pending.store(id, std::memory_order_release);
            context.EFlags |= TrapFlag;
            stepped = SetThreadContext(handle, &context) != 0;
            if (!stepped) g_pending.store(0, std::memory_order_release);
        }
        ResumeThread(handle);
        if (!stepped) {
            ++outside;
            if (++misses % 256 == 0) Sleep(1);
            continue;
        }
        misses = 0;
        ++injected;
        nextStep = std::max(nextStep + period, Now() - second);
        const auto deadline = Now() + second;
        bool counted = false;
        while (g_pending.load(std::memory_order_acquire) != 0) {
            if (!counted && Now() > deadline) {
                ++slow;
                counted = true;
            }
            SwitchToThread();
        }
    }
}

}

bool Handle(EXCEPTION_POINTERS* info) {
    if (!g_active.load(std::memory_order_relaxed) || info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) return false;
    auto* context = info->ContextRecord;
    if (g_pending.load(std::memory_order_acquire) == GetCurrentThreadId()) {
        g_delivered.fetch_add(1, std::memory_order_relaxed);
        const auto* guest = g_guestCode.load(std::memory_order_acquire);
        const auto rip = static_cast<std::uintptr_t>(context->Rip);
        const bool inRange = g_clobberImage.empty() || (rip >= g_clobberLow.load(std::memory_order_relaxed) && rip < g_clobberHigh.load(std::memory_order_acquire));
        if (g_clobber && inRange && (!g_clobberGuestOnly || (guest != nullptr && Inside(*guest, rip)))) std::memset(reinterpret_cast<void*>(context->Rsp - RedZoneBytes), 0xa5, RedZoneBytes);
        if (g_remaining.fetch_sub(1, std::memory_order_relaxed) <= 1) {
            context->EFlags &= ~TrapFlag;
            g_pending.store(0, std::memory_order_release);
        } else {
            context->EFlags |= TrapFlag;
        }
        return true;
    }
    if ((context->Dr6 & SingleStepStatus) != 0 && (context->Dr7 & 0xff) == 0) {
        context->EFlags &= ~TrapFlag;
        g_stray.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    return false;
}

void Start() {
    const char* value = std::getenv("APS5_FAULT_STORM");
    if (value == nullptr || value[0] < '0' || value[0] > '9') return;
    auto rate = std::strtoull(value, nullptr, 10);
    if (rate == 0) return;
    if (rate == 1) rate = 2000;
    if (const char* burst = std::getenv("APS5_FAULT_STORM_BURST"); burst != nullptr && burst[0] >= '0' && burst[0] <= '9') g_burst = std::max(1u, static_cast<unsigned>(std::strtoul(burst, nullptr, 10)));
    else g_burst = 256;
    g_clobber = std::getenv("APS5_FAULT_STORM_NO_CLOBBER") == nullptr;
    g_clobberGuestOnly = std::getenv("APS5_FAULT_STORM_GUEST_CLOBBER") != nullptr;
    if (const char* range = std::getenv("APS5_FAULT_STORM_CLOBBER_RANGE")) {
        const char* colon = std::strchr(range, ':');
        const char* dash = colon != nullptr ? std::strchr(colon, '-') : nullptr;
        if (colon != nullptr && dash != nullptr) {
            for (const char* at = range; at < colon; ++at) g_clobberImage.push_back(static_cast<wchar_t>(*at));
            g_clobberBegin = static_cast<std::uintptr_t>(std::strtoull(colon + 1, nullptr, 16));
            g_clobberEnd = static_cast<std::uintptr_t>(std::strtoull(dash + 1, nullptr, 16));
            std::fprintf(stderr, "[fault-storm] red zone overwritten only in %.*s+0x%llx..0x%llx\n", static_cast<int>(colon - range), range, static_cast<unsigned long long>(g_clobberBegin), static_cast<unsigned long long>(g_clobberEnd));
        }
    }
    g_active.store(true, std::memory_order_relaxed);
    std::fprintf(stderr, "[fault-storm] on: up to %llu landings per second in guest code, %u steps each, red zone %s at every step\n", rate, g_burst, g_clobber ? "overwritten" : "left to the real frame");
    std::fflush(stderr);
    if (const auto thread = CreateThread(nullptr, 0, Run, reinterpret_cast<LPVOID>(static_cast<std::uintptr_t>(rate)), 0, nullptr)) CloseHandle(thread);
}

}
