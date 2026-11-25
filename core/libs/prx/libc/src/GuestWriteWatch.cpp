#include "prx/libc/include/GuestWriteWatch.hpp"
#if defined(__linux__)
#include "prx/libc/include/specifics/linux/WriteWatchOracle.hpp"
#endif
#if defined(__linux__) && defined(__x86_64__)
#include "prx/libc/include/specifics/linux/GuardedCopy.hpp"
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include "prx/libc/include/specifics/linux/HostMapping.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/futex.h>
#include <linux/userfaultfd.h>
#include <pthread.h>
#include <sched.h>
#include <ucontext.h>
#include <semaphore.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifndef UFFD_FEATURE_WP_UNPOPULATED
#define UFFD_FEATURE_WP_UNPOPULATED (1 << 13)
#endif
#ifndef UFFD_FEATURE_WP_ASYNC
#define UFFD_FEATURE_WP_ASYNC (1 << 15)
#endif
#ifndef PAGEMAP_SCAN
struct page_region {
    __u64 start;
    __u64 end;
    __u64 categories;
};
struct pm_scan_arg {
    __u64 size;
    __u64 flags;
    __u64 start;
    __u64 end;
    __u64 walk_end;
    __u64 vec;
    __u64 vec_len;
    __u64 max_pages;
    __u64 category_inverted;
    __u64 category_mask;
    __u64 category_anyof_mask;
    __u64 return_mask;
};
#define PAGE_IS_WRITTEN (1 << 1)
#define PM_SCAN_WP_MATCHING (1 << 0)
#define PM_SCAN_CHECK_WPASYNC (1 << 1)
#define PAGEMAP_SCAN _IOWR('f', 16, struct pm_scan_arg)
#endif
#endif

#if defined(__APPLE__)
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/semaphore.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>
#endif

#if (defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC)) || defined(__APPLE__)
#define APS5_GUEST_WRITE_WATCH 1
#endif

#if defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC) && !defined(ANYPS5_KYTY)
#define APS5_LINUX_READ_TRAPS 1
#endif

namespace GuestWriteWatch {
namespace {

#if defined(__linux__)
#if defined(ANYPS5_KYTY)
constexpr bool GpuOwnsFaults = true;
#else
constexpr bool GpuOwnsFaults = false;
#endif

class ReadTrapTable {
public:
    static ReadTrapTable& Get() {
        static auto* table = new ReadTrapTable;
        return *table;
    }

    void Register(std::uintptr_t begin, std::uintptr_t end) {
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin) return;
        const Guard lock(_lock);
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> gaps;
        auto cursor = begin;
        auto it = _ranges.upper_bound(begin);
        if (it != _ranges.begin()) --it;
        for (; it != _ranges.end() && it->first < end; ++it) {
            if (it->second.end <= cursor) continue;
            if (it->first > cursor) gaps.emplace_back(cursor, it->first);
            cursor = std::max(cursor, it->second.end);
        }
        if (cursor < end) gaps.emplace_back(cursor, end);
        for (const auto& [from, to] : gaps) _ranges.emplace(from, Range{to, std::vector<std::uint8_t>(static_cast<std::size_t>((to - from) / PageSize), 0)});
        _registered.store(true, std::memory_order_release);
    }

    void Unregister(std::uintptr_t begin, std::uintptr_t end) {
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin || !_registered.load(std::memory_order_acquire)) return;
        const Guard lock(_lock);
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> dropped;
        for (auto it = _holds.begin(); it != _holds.end();) {
            if (pageStart(it->begin) < end && begin < pageEnd(it->end)) {
                dropped.emplace_back(it->begin, it->end);
                it = _holds.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto& [from, to] : dropped) reconcile(pageStart(from), pageEnd(to), false);
        reconcile(begin, end, false);
        erase(begin, end);
        noteHolds();
    }

    bool SetResolver(void (*resolver)(std::uintptr_t, std::uintptr_t)) {
        if (resolver == nullptr) {
            _resolver.store(nullptr, std::memory_order_release);
            return true;
        }
        std::call_once(_resolverStarted, [this] { startResolver(); });
        if (!_resolverRunning.load(std::memory_order_acquire)) return false;
        _resolver.store(resolver, std::memory_order_release);
        return true;
    }

    void SetBypass(bool (*bypass)()) {
        _bypass.store(bypass, std::memory_order_release);
    }

    bool Trap(std::uintptr_t begin, std::uintptr_t end, bool writes) {
        if (end <= begin || _resolver.load(std::memory_order_acquire) == nullptr) return false;
        const Guard lock(_lock);
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        const auto found = std::find_if(_holds.begin(), _holds.end(), [&](const Hold& hold) { return hold.begin == begin && hold.end == end && hold.writes == writes; });
        if (found != _holds.end() && found->state != HoldResolved && found->state != HoldDormant) {
            ++found->count;
            ++found->generation;
            reconcile(first, last, true);
            return true;
        }
        if (found != _holds.end() && found->state == HoldDormant) {
            bool armed = true;
            forEachPage(first, last, [&](std::uintptr_t, std::uint8_t& state) { armed = armed && (state & (Armed | ArmedWrites)) != 0; });
            if (armed) {
                found->count = 1;
                ++found->generation;
                found->state = HoldArmed;
                reconcile(first, last, true);
                _counts.revived.fetch_add(1, std::memory_order_relaxed);
                _counts.trapped.fetch_add(1, std::memory_order_relaxed);
                noteHolds();
                return true;
            }
            found->count = 0;
        }
        if (!prepare(first, last)) {
            _counts.refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (found != _holds.end()) {
            ++found->count;
            ++found->generation;
            found->state = HoldArmed;
            found->dormantSince = 0;
        } else {
            _holds.push_back({begin, end, 1, HoldArmed, 0, 0, writes, 0});
        }
        if (!reconcile(first, last, true)) {
            const auto added = std::find_if(_holds.begin(), _holds.end(), [&](const Hold& hold) { return hold.begin == begin && hold.end == end && hold.writes == writes; });
            if (added != _holds.end() && --added->count == 0) _holds.erase(added);
            else if (added != _holds.end()) added->state = HoldResolved;
            reconcile(first, last, false);
            _counts.refused.fetch_add(1, std::memory_order_relaxed);
            noteHolds();
            return false;
        }
        std::uint64_t pages = 0;
        forEachPage(first, last, [&](std::uintptr_t, std::uint8_t&) { ++pages; });
        _counts.trappedPages.fetch_add(pages, std::memory_order_relaxed);
        _counts.trapped.fetch_add(1, std::memory_order_relaxed);
        noteHolds();
        return true;
    }

    void Release(std::uintptr_t begin, std::uintptr_t end, bool writes) {
        if (end <= begin || !_registered.load(std::memory_order_acquire)) return;
        const Guard lock(_lock);
        const auto found = std::find_if(_holds.begin(), _holds.end(), [&](const Hold& hold) { return hold.begin == begin && hold.end == end && hold.writes == writes; });
        if (found == _holds.end() || found->state == HoldDormant || --found->count != 0) return;
        _counts.released.fetch_add(1, std::memory_order_relaxed);
        if (found->state == HoldArmed) {
            found->state = HoldDormant;
            found->dormantSince = _sweep;
            reconcile(pageStart(begin), pageEnd(end), false);
            noteHolds();
            return;
        }
        _holds.erase(found);
        reconcile(pageStart(begin), pageEnd(end), false);
        noteHolds();
    }

    void Sweep() {
        if (!_registered.load(std::memory_order_acquire)) return;
        const Guard lock(_lock);
        ++_sweep;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> dropped;
        for (auto it = _holds.begin(); it != _holds.end();) {
            if (it->state == HoldDormant && it->dormantSince + DormantSweeps <= _sweep) {
                dropped.emplace_back(pageStart(it->begin), pageEnd(it->end));
                it = _holds.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto& [from, to] : dropped) reconcile(from, to, false);
        if (!dropped.empty()) {
            _counts.swept.fetch_add(dropped.size(), std::memory_order_relaxed);
            noteHolds();
        }
    }

    bool Active() const {
        return _active.load(std::memory_order_acquire) != 0 || _dormant.load(std::memory_order_acquire) != 0;
    }

    std::uint64_t Epoch() const {
        return _epoch.load(std::memory_order_acquire);
    }

    int Protection(std::uintptr_t address, std::uintptr_t limit, std::uintptr_t* runEnd) {
        *runEnd = limit;
        if (!_registered.load(std::memory_order_acquire) || limit <= address) return -1;
        const Guard lock(_lock);
        const auto page = pageStart(address);
        auto* state = pageState(page);
        if (state != nullptr && (*state & Held) != 0) {
            const int protection = *state & ProtectionBits;
            auto end = page + PageSize;
            while (end < limit) {
                const auto* next = pageState(end);
                if (next == nullptr || (*next & Held) == 0 || (*next & ProtectionBits) != protection) break;
                end += PageSize;
            }
            *runEnd = std::min(end, limit);
            return protection;
        }
        auto end = page + PageSize;
        while (end < limit) {
            const auto* next = pageState(end);
            if (next != nullptr && (*next & Held) != 0) break;
            end += PageSize;
        }
        *runEnd = std::min(end, limit);
        return -1;
    }

    void Settle(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin || !Active()) return;
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        const auto self = currentThread();
        const auto bypass = _bypass.load(std::memory_order_acquire);
        {
            const Guard lock(_lock);
            if (!activeOverlap(first, last)) {
                openDormant(first, last);
                return;
            }
            _counts.settles.fetch_add(1, std::memory_order_relaxed);
            if (self == _resolverThread.load(std::memory_order_acquire) || resolvingBy(first, last, self) || (bypass != nullptr && bypass())) {
                forEachPage(first, last, [&](std::uintptr_t page, std::uint8_t& state) {
                    if ((state & (Armed | ArmedWrites)) == 0) return;
                    openPage(page, state);
                    _counts.settleOpens.fetch_add(1, std::memory_order_relaxed);
                });
                return;
            }
        }
        waitForResolution(first, last);
    }

    void BeginHostWrite(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin || _resolver.load(std::memory_order_acquire) == nullptr) return;
        Settle(begin, end);
        const Guard lock(_lock);
        _hostWrites.emplace_back(pageStart(begin), pageEnd(end));
    }

    void EndHostWrite(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin || _resolver.load(std::memory_order_acquire) == nullptr) return;
        const Guard lock(_lock);
        const auto found = std::find(_hostWrites.begin(), _hostWrites.end(), std::pair{pageStart(begin), pageEnd(end)});
        if (found != _hostWrites.end()) _hostWrites.erase(found);
    }

    int OpenIdle(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin || !Active()) return 0;
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        const Guard lock(_lock);
        if (activeOverlap(first, last)) return -1;
        int opened = 0;
        forEachPage(first, last, [&](std::uintptr_t page, std::uint8_t& state) {
            if ((state & (Armed | ArmedWrites)) == 0) return;
            openPage(page, state);
            _counts.dormantOpens.fetch_add(1, std::memory_order_relaxed);
            ++opened;
        });
        return opened;
    }

    void Open(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin || !Active()) return;
        const Guard lock(_lock);
        forEachPage(pageStart(begin), pageEnd(end), [&](std::uintptr_t page, std::uint8_t& state) {
            if ((state & (Armed | ArmedWrites)) == 0) return;
            openPage(page, state);
            _counts.settleOpens.fetch_add(1, std::memory_order_relaxed);
        });
    }

    int Protect(std::uintptr_t begin, std::uintptr_t end, int protection) {
        const Guard lock(_lock);
        forEachPage(pageStart(begin), pageEnd(end), [&](std::uintptr_t page, std::uint8_t& state) {
            if ((state & (Armed | ArmedWrites)) == 0) return;
            openPage(page, state);
            _counts.settleOpens.fetch_add(1, std::memory_order_relaxed);
        });
        if (::mprotect(reinterpret_cast<void*>(begin), static_cast<std::size_t>(end - begin), protection) != 0) return errno;
        forEachPage(pageStart(begin), pageEnd(end), [&](std::uintptr_t, std::uint8_t& state) { state = static_cast<std::uint8_t>((state & ~ProtectionBits) | Known | (protection & ProtectionBits)); });
        return 0;
    }

    bool HandleFault(int signal, siginfo_t* info, void* context) {
        if (signal != SIGSEGV || info == nullptr || info->si_code != SEGV_ACCERR || context == nullptr || !_registered.load(std::memory_order_acquire)) return false;
        const int savedErrno = errno;
        const auto page = pageStart(reinterpret_cast<std::uintptr_t>(info->si_addr));
        const bool write = (static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 2) != 0;
        const auto self = currentThread();
        {
            const Guard lock(_lock);
            auto* state = pageState(page);
            if (state == nullptr) {
                errno = savedErrno;
                return false;
            }
            if ((*state & Armed) == 0 && ((*state & ArmedWrites) == 0 || !write)) {
                const bool retry = retryOpen(page, *state, write, self);
                errno = savedErrno;
                return retry;
            }
            if (!activeOverlap(page, page + PageSize)) {
                openPage(page, *state);
                _counts.dormantOpens.fetch_add(1, std::memory_order_relaxed);
                errno = savedErrno;
                return true;
            }
            const bool own = self == _resolverThread.load(std::memory_order_acquire) || resolvingBy(page, page + PageSize, self);
            const auto bypass = _bypass.load(std::memory_order_acquire);
            if (own || (bypass != nullptr && bypass())) {
                openPage(page, *state);
                (own ? _counts.ownOpens : _counts.bypassOpens).fetch_add(1, std::memory_order_relaxed);
                errno = savedErrno;
                return true;
            }
        }
        _counts.faults.fetch_add(1, std::memory_order_relaxed);
        noteFault(page, static_cast<std::uintptr_t>(static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP]), self, write);
        waitForResolution(page, page + PageSize);
        errno = savedErrno;
        return true;
    }

    std::size_t Faults(ReadTrapFault* faults, std::size_t capacity) const {
        const auto newest = _faultIndex.load(std::memory_order_acquire);
        std::size_t count = 0;
        for (std::uint64_t index = newest; index > 0 && count < capacity && newest - index < FaultRing; --index) {
            const auto& entry = _faultRing[(index - 1) % FaultRing];
            faults[count++] = {entry.page.load(std::memory_order_relaxed), entry.instruction.load(std::memory_order_relaxed), entry.thread.load(std::memory_order_relaxed), entry.write.load(std::memory_order_relaxed)};
        }
        return count;
    }

    void Counts(ReadTrapCounts& counts) const {
        counts.trapped = _counts.trapped.load(std::memory_order_relaxed);
        counts.trappedPages = _counts.trappedPages.load(std::memory_order_relaxed);
        counts.refused = _counts.refused.load(std::memory_order_relaxed);
        counts.released = _counts.released.load(std::memory_order_relaxed);
        counts.faults = _counts.faults.load(std::memory_order_relaxed);
        counts.resolved = _counts.resolved.load(std::memory_order_relaxed);
        counts.resolveNs = _counts.resolveNs.load(std::memory_order_relaxed);
        counts.waitNs = _counts.waitNs.load(std::memory_order_relaxed);
        counts.bypassOpens = _counts.bypassOpens.load(std::memory_order_relaxed);
        counts.ownOpens = _counts.ownOpens.load(std::memory_order_relaxed);
        counts.forcedOpens = _counts.forcedOpens.load(std::memory_order_relaxed);
        counts.retries = _counts.retries.load(std::memory_order_relaxed);
        counts.settles = _counts.settles.load(std::memory_order_relaxed);
        counts.settleOpens = _counts.settleOpens.load(std::memory_order_relaxed);
        counts.holds = _active.load(std::memory_order_relaxed);
        counts.dormant = _dormant.load(std::memory_order_relaxed);
        counts.revived = _counts.revived.load(std::memory_order_relaxed);
        counts.swept = _counts.swept.load(std::memory_order_relaxed);
        counts.dormantOpens = _counts.dormantOpens.load(std::memory_order_relaxed);
    }

private:
    static constexpr std::uintptr_t PageSize = 4096;
    static constexpr std::uint8_t ProtectionBits = PROT_READ | PROT_WRITE | PROT_EXEC;
    static constexpr std::uint8_t Known = 0x08;
    static constexpr std::uint8_t Held = 0x10;
    static constexpr std::uint8_t Armed = 0x20;
    static constexpr std::uint8_t ArmedWrites = 0x40;
    static constexpr std::uint8_t Opened = 0x80;
    static constexpr std::size_t SlotCount = 64;
    static constexpr std::size_t RetryCount = 64;
    static constexpr unsigned RetryLimit = 16;
    static constexpr std::int64_t WaitSliceNs = 50'000'000;
    static constexpr std::int64_t ForcedOpenNs = 5'000'000'000;

    enum SlotState : unsigned { Free, Claimed, Queued, Running, Done, Abandoned };
    enum HoldState : std::uint8_t { HoldArmed, HoldResolving, HoldResolved, HoldDormant };
    static constexpr std::uint64_t DormantSweeps = 4;

    struct Hold {
        std::uintptr_t begin;
        std::uintptr_t end;
        std::uint32_t count;
        HoldState state;
        pid_t owner;
        std::uint64_t generation;
        bool writes;
        std::uint64_t dormantSince;
    };

    struct Range {
        std::uintptr_t end;
        std::vector<std::uint8_t> pages;
    };

    struct Slot {
        std::atomic<unsigned> state{Free};
        std::uintptr_t begin = 0;
        std::uintptr_t end = 0;
    };

    class Guard {
    public:
        explicit Guard(std::mutex& mutex) : _mutex(mutex) {
            sigset_t blocked;
            sigfillset(&blocked);
            for (const int synchronous : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP, SIGABRT}) sigdelset(&blocked, synchronous);
            pthread_sigmask(SIG_BLOCK, &blocked, &_saved);
            _mutex.lock();
        }
        ~Guard() {
            _mutex.unlock();
            pthread_sigmask(SIG_SETMASK, &_saved, nullptr);
        }
        Guard(const Guard&) = delete;
        Guard& operator=(const Guard&) = delete;

    private:
        std::mutex& _mutex;
        sigset_t _saved;
    };

    struct Retry {
        std::uintptr_t page = 0;
        pid_t thread = 0;
        std::uint64_t epoch = 0;
        unsigned count = 0;
        std::uint64_t used = 0;
    };

    struct FaultEntry {
        std::atomic<std::uintptr_t> page{0};
        std::atomic<std::uintptr_t> instruction{0};
        std::atomic<int> thread{0};
        std::atomic<bool> write{false};
    };

    static constexpr std::size_t FaultRing = 64;

    void noteFault(std::uintptr_t page, std::uintptr_t instruction, pid_t thread, bool write) {
        const auto index = _faultIndex.load(std::memory_order_relaxed);
        auto& entry = _faultRing[index % FaultRing];
        entry.page.store(page, std::memory_order_relaxed);
        entry.instruction.store(instruction, std::memory_order_relaxed);
        entry.thread.store(thread, std::memory_order_relaxed);
        entry.write.store(write, std::memory_order_relaxed);
        _faultIndex.fetch_add(1, std::memory_order_release);
    }

    struct Taken {
        std::uintptr_t begin;
        std::uintptr_t end;
        std::uint64_t generation;
    };

    struct Counters {
        std::atomic<std::uint64_t> trapped{0}, trappedPages{0}, refused{0}, released{0}, faults{0}, resolved{0}, resolveNs{0}, waitNs{0}, bypassOpens{0}, ownOpens{0}, forcedOpens{0}, retries{0}, settles{0}, settleOpens{0}, revived{0}, swept{0}, dormantOpens{0};
    };

    static_assert(sizeof(std::atomic<unsigned>) == sizeof(unsigned));

    ReadTrapTable() {
        _holds.reserve(4096);
        _hostWrites.reserve(256);
    }

    static pid_t currentThread() {
        return static_cast<pid_t>(syscall(SYS_gettid));
    }

    static std::int64_t now() {
        timespec time{};
        clock_gettime(CLOCK_MONOTONIC, &time);
        return static_cast<std::int64_t>(time.tv_sec) * 1'000'000'000 + time.tv_nsec;
    }

    static void futexWait(std::atomic<unsigned>& word, unsigned expected, std::int64_t timeoutNs) {
        timespec timeout{static_cast<time_t>(timeoutNs / 1'000'000'000), static_cast<long>(timeoutNs % 1'000'000'000)};
        syscall(SYS_futex, reinterpret_cast<unsigned*>(&word), FUTEX_WAIT_PRIVATE, expected, timeoutNs > 0 ? &timeout : nullptr, nullptr, 0);
    }

    static void futexWake(std::atomic<unsigned>& word) {
        syscall(SYS_futex, reinterpret_cast<unsigned*>(&word), FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
    }

    void noteHolds() {
        const auto active = static_cast<std::size_t>(std::count_if(_holds.begin(), _holds.end(), [](const Hold& hold) { return hold.state == HoldArmed || hold.state == HoldResolving; }));
        const auto dormant = static_cast<std::size_t>(std::count_if(_holds.begin(), _holds.end(), [](const Hold& hold) { return hold.state == HoldDormant; }));
        _active.store(active, std::memory_order_release);
        _dormant.store(dormant, std::memory_order_release);
        _epoch.fetch_add(1, std::memory_order_acq_rel);
    }

    bool prepare(std::uintptr_t first, std::uintptr_t last) {
        if (!covers(first, last)) return false;
        for (const auto& [hostBegin, hostEnd] : _hostWrites)
            if (hostBegin < last && first < hostEnd) return false;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> unknown;
        forEachPage(first, last, [&](std::uintptr_t page, std::uint8_t& state) {
            if ((state & (Armed | ArmedWrites)) != 0) return;
            if (!unknown.empty() && unknown.back().second == page) unknown.back().second = page + PageSize;
            else unknown.emplace_back(page, page + PageSize);
        });
        std::vector<std::tuple<std::uintptr_t, std::uintptr_t, int>> protections;
        for (const auto& [from, to] : unknown) {
            if (!queryProtections(from, to, protections)) return false;
        }
        for (const auto& [from, to, protection] : protections)
            if ((protection & PROT_READ) == 0) return false;
        for (const auto& [from, to, protection] : protections) {
            forEachPage(from, to, [&](std::uintptr_t, std::uint8_t& state) { state = static_cast<std::uint8_t>((state & (Held | Opened)) | Known | (protection & ProtectionBits)); });
        }
        return true;
    }

    bool reconcile(std::uintptr_t first, std::uintptr_t last, bool reopen) {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> reads;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> writes;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> dormant;
        for (const auto& hold : _holds) {
            if (hold.state == HoldResolved || pageStart(hold.begin) >= last || first >= pageEnd(hold.end)) continue;
            if (hold.state == HoldDormant) dormant.emplace_back(pageStart(hold.begin), pageEnd(hold.end));
            else (hold.writes ? writes : reads).emplace_back(pageStart(hold.begin), pageEnd(hold.end));
        }
        const auto merge = [](std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& spans) {
            std::sort(spans.begin(), spans.end());
            std::size_t merged = 0;
            for (const auto& span : spans) {
                if (merged != 0 && span.first <= spans[merged - 1].second) spans[merged - 1].second = std::max(spans[merged - 1].second, span.second);
                else spans[merged++] = span;
            }
            spans.resize(merged);
        };
        merge(reads);
        merge(writes);
        merge(dormant);
        std::size_t readAt = 0;
        std::size_t writeAt = 0;
        std::size_t dormantAt = 0;
        const auto covered = [](const std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& spans, std::size_t& at, std::uintptr_t page) {
            while (at < spans.size() && spans[at].second <= page) ++at;
            return at < spans.size() && spans[at].first <= page;
        };
        bool applied = true;
        bool changed = false;
        std::uintptr_t runStart = 0;
        std::uintptr_t runEnd = 0;
        int runProtection = -1;
        const auto flush = [&] {
            if (runProtection >= 0 && ::mprotect(reinterpret_cast<void*>(runStart), static_cast<std::size_t>(runEnd - runStart), runProtection) != 0) applied = false;
            runProtection = -1;
        };
        forEachPage(first, last, [&](std::uintptr_t page, std::uint8_t& state) {
            const bool read = covered(reads, readAt, page);
            const bool write = covered(writes, writeAt, page);
            const bool asleep = covered(dormant, dormantAt, page) && (state & (Armed | ArmedWrites)) != 0;
            if (!read && !write && asleep) return;
            const bool held = read || write;
            const int protection = state & ProtectionBits;
            std::uint8_t next = state;
            if (!held) next = static_cast<std::uint8_t>(next & ~(Held | Opened));
            else next = static_cast<std::uint8_t>(next | Held);
            if (held && reopen) next = static_cast<std::uint8_t>(next & ~Opened);
            std::uint8_t level = 0;
            if (held && (next & Opened) == 0) level = read ? Armed : (protection & PROT_WRITE) != 0 ? ArmedWrites : 0;
            if ((state & (Armed | ArmedWrites)) != level) {
                const int wanted = level == Armed ? PROT_NONE : level == ArmedWrites ? (protection & ~PROT_WRITE) : protection;
                if (runProtection != wanted || runEnd != page) {
                    flush();
                    runStart = page;
                    runProtection = wanted;
                }
                runEnd = page + PageSize;
                changed = true;
            }
            state = static_cast<std::uint8_t>((next & ~(Armed | ArmedWrites)) | level);
        });
        flush();
        if (changed) _epoch.fetch_add(1, std::memory_order_acq_rel);
        return applied;
    }

    void startResolver() {
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
        pthread_t thread{};
        const bool created = pthread_create(&thread, &attributes, [](void* table) -> void* {
            static_cast<ReadTrapTable*>(table)->resolveLoop();
            return nullptr;
        }, this) == 0;
        pthread_attr_destroy(&attributes);
        if (!created) return;
        while (_resolverThread.load(std::memory_order_acquire) == 0) sched_yield();
        _resolverRunning.store(true, std::memory_order_release);
    }

    void resolveLoop() {
        pthread_setname_np(pthread_self(), "read traps");
        _resolverThread.store(currentThread(), std::memory_order_release);
        const auto self = _resolverThread.load(std::memory_order_relaxed);
        for (;;) {
            const auto seen = _wake.load(std::memory_order_acquire);
            bool served = false;
            for (auto& slot : _slots) {
                unsigned expected = Queued;
                if (!slot.state.compare_exchange_strong(expected, Running, std::memory_order_acq_rel)) continue;
                served = true;
                const auto start = now();
                const auto first = slot.begin;
                const auto last = slot.end;
                for (;;) {
                    if (resolve(first, last, self)) continue;
                    {
                        const Guard lock(_lock);
                        if (!resolvingOverlap(first, last)) {
                            reconcile(first, last, false);
                            if (!activeOverlap(first, last)) openDormant(first, last);
                            break;
                        }
                    }
                    sched_yield();
                }
                _counts.resolved.fetch_add(1, std::memory_order_relaxed);
                _counts.resolveNs.fetch_add(static_cast<std::uint64_t>(now() - start), std::memory_order_relaxed);
                expected = Running;
                if (slot.state.compare_exchange_strong(expected, Done, std::memory_order_acq_rel)) futexWake(slot.state);
                else slot.state.store(Free, std::memory_order_release);
            }
            if (!served) futexWait(_wake, seen, 0);
        }
    }

    bool resolve(std::uintptr_t first, std::uintptr_t last, pid_t self) {
        std::vector<Taken> taken;
        {
            const Guard lock(_lock);
            for (auto& hold : _holds) {
                if (hold.state != HoldArmed || pageStart(hold.begin) >= last || first >= pageEnd(hold.end)) continue;
                hold.state = HoldResolving;
                hold.owner = self;
                taken.push_back({hold.begin, hold.end, hold.generation});
            }
            if (taken.empty()) return false;
            noteHolds();
        }
        const auto resolver = _resolver.load(std::memory_order_acquire);
        for (const auto& hold : taken)
            if (resolver != nullptr) resolver(hold.begin, hold.end);
        const Guard lock(_lock);
        for (const auto& entry : taken) {
            const auto found = std::find_if(_holds.begin(), _holds.end(), [&](const Hold& hold) { return hold.begin == entry.begin && hold.end == entry.end; });
            if (found == _holds.end() || found->state != HoldResolving || found->owner != self) continue;
            found->state = found->generation == entry.generation ? HoldResolved : HoldArmed;
            found->owner = 0;
        }
        for (const auto& entry : taken) reconcile(pageStart(entry.begin), pageEnd(entry.end), false);
        noteHolds();
        return true;
    }

    void waitForResolution(std::uintptr_t first, std::uintptr_t last) {
        const auto start = now();
        Slot* slot = nullptr;
        while (slot == nullptr) {
            for (auto& candidate : _slots) {
                unsigned expected = Free;
                if (candidate.state.compare_exchange_strong(expected, Claimed, std::memory_order_acq_rel)) {
                    slot = &candidate;
                    break;
                }
            }
            if (slot == nullptr) sched_yield();
        }
        slot->begin = first;
        slot->end = last;
        slot->state.store(Queued, std::memory_order_release);
        _wake.fetch_add(1, std::memory_order_acq_rel);
        futexWake(_wake);
        for (;;) {
            const auto state = slot->state.load(std::memory_order_acquire);
            if (state == Done) break;
            if (now() - start >= ForcedOpenNs) {
                unsigned expected = Queued;
                if (!slot->state.compare_exchange_strong(expected, Free, std::memory_order_acq_rel)) {
                    expected = Running;
                    if (!slot->state.compare_exchange_strong(expected, Abandoned, std::memory_order_acq_rel)) continue;
                }
                {
                    const Guard lock(_lock);
                    forEachPage(first, last, [&](std::uintptr_t page, std::uint8_t& state) {
                        if ((state & (Armed | ArmedWrites)) != 0) openPage(page, state);
                    });
                }
                _counts.forcedOpens.fetch_add(1, std::memory_order_relaxed);
                constexpr char message[] = "[read-trap] a trapped page was not resolved within 5 s; it is opened without waiting for the GPU\n";
                static_cast<void>(::write(STDERR_FILENO, message, sizeof(message) - 1));
                _counts.waitNs.fetch_add(static_cast<std::uint64_t>(now() - start), std::memory_order_relaxed);
                return;
            }
            futexWait(slot->state, state, WaitSliceNs);
        }
        slot->state.store(Free, std::memory_order_release);
        _counts.waitNs.fetch_add(static_cast<std::uint64_t>(now() - start), std::memory_order_relaxed);
    }

    bool retryOpen(std::uintptr_t page, std::uint8_t state, bool write, pid_t self) {
        if ((state & Known) == 0 || (state & (write ? PROT_WRITE : PROT_READ)) == 0) return false;
        const auto epoch = _epoch.load(std::memory_order_acquire);
        auto found = std::find_if(_retries.begin(), _retries.end(), [&](const Retry& retry) { return retry.used != 0 && retry.page == page && retry.thread == self; });
        if (found == _retries.end()) found = std::min_element(_retries.begin(), _retries.end(), [](const Retry& left, const Retry& right) { return left.used < right.used; });
        if (found->used == 0 || found->page != page || found->thread != self || found->epoch != epoch) *found = {page, self, epoch, 0, 0};
        found->used = ++_retryUses;
        if (++found->count > RetryLimit) return false;
        _counts.retries.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    void openPage(std::uintptr_t page, std::uint8_t& state) {
        ::mprotect(reinterpret_cast<void*>(page), PageSize, state & ProtectionBits);
        state = static_cast<std::uint8_t>((state & ~(Armed | ArmedWrites)) | Opened);
        _epoch.fetch_add(1, std::memory_order_acq_rel);
    }

    bool activeOverlap(std::uintptr_t first, std::uintptr_t last) const {
        return std::any_of(_holds.begin(), _holds.end(), [&](const Hold& hold) { return (hold.state == HoldArmed || hold.state == HoldResolving) && pageStart(hold.begin) < last && first < pageEnd(hold.end); });
    }

    void openDormant(std::uintptr_t first, std::uintptr_t last) {
        forEachPage(first, last, [&](std::uintptr_t page, std::uint8_t& state) {
            if ((state & (Armed | ArmedWrites)) == 0) return;
            openPage(page, state);
            _counts.dormantOpens.fetch_add(1, std::memory_order_relaxed);
        });
    }

    bool resolvingOverlap(std::uintptr_t first, std::uintptr_t last) const {
        return std::any_of(_holds.begin(), _holds.end(), [&](const Hold& hold) { return hold.state == HoldResolving && pageStart(hold.begin) < last && first < pageEnd(hold.end); });
    }

    bool resolvingBy(std::uintptr_t first, std::uintptr_t last, pid_t self) const {
        return std::any_of(_holds.begin(), _holds.end(), [&](const Hold& hold) { return hold.state == HoldResolving && hold.owner == self && pageStart(hold.begin) < last && first < pageEnd(hold.end); });
    }

    bool queryProtections(std::uintptr_t first, std::uintptr_t last, std::vector<std::tuple<std::uintptr_t, std::uintptr_t, int>>& protections) const {
        const int fd = HostMapping::QueryFd();
        if (fd < 0) return false;
        for (auto cursor = first; cursor < last;) {
            procmap_query query{};
            query.size = sizeof(query);
            query.query_flags = PROCMAP_QUERY_COVERING_OR_NEXT_VMA;
            query.query_addr = cursor;
            if (ioctl(fd, PROCMAP_QUERY, &query) != 0 || query.vma_start > cursor || query.vma_end <= cursor) return false;
            const int protection = ((query.vma_flags & PROCMAP_QUERY_VMA_READABLE) != 0 ? PROT_READ : 0) | ((query.vma_flags & PROCMAP_QUERY_VMA_WRITABLE) != 0 ? PROT_WRITE : 0) | ((query.vma_flags & PROCMAP_QUERY_VMA_EXECUTABLE) != 0 ? PROT_EXEC : 0);
            const auto next = std::min<std::uintptr_t>(last, query.vma_end);
            protections.emplace_back(cursor, next, protection);
            cursor = next;
        }
        return true;
    }

    std::uint8_t* pageState(std::uintptr_t page) {
        auto it = _ranges.upper_bound(page);
        if (it == _ranges.begin()) return nullptr;
        --it;
        if (page >= it->second.end) return nullptr;
        return &it->second.pages[static_cast<std::size_t>((page - it->first) / PageSize)];
    }

    template <typename TFunction>
    void forEachPage(std::uintptr_t begin, std::uintptr_t end, TFunction function) {
        auto it = _ranges.upper_bound(begin);
        if (it != _ranges.begin()) --it;
        for (; it != _ranges.end() && it->first < end; ++it) {
            const auto first = std::max(begin, it->first);
            const auto last = std::min(end, it->second.end);
            for (auto page = first; page < last; page += PageSize) function(page, it->second.pages[static_cast<std::size_t>((page - it->first) / PageSize)]);
        }
    }

    bool covers(std::uintptr_t begin, std::uintptr_t end) const {
        auto it = _ranges.upper_bound(begin);
        if (it == _ranges.begin()) return false;
        --it;
        auto cursor = begin;
        for (; it != _ranges.end() && it->first <= cursor; ++it) {
            if (it->second.end > cursor) cursor = it->second.end;
            if (cursor >= end) return true;
        }
        return false;
    }

    void erase(std::uintptr_t begin, std::uintptr_t end) {
        auto it = _ranges.upper_bound(begin);
        if (it != _ranges.begin()) --it;
        while (it != _ranges.end() && it->first < end) {
            const auto rangeBegin = it->first;
            const auto rangeEnd = it->second.end;
            if (rangeEnd <= begin) {
                ++it;
                continue;
            }
            auto pages = std::move(it->second.pages);
            it = _ranges.erase(it);
            const auto cutBegin = std::max(begin, rangeBegin);
            const auto cutEnd = std::min(end, rangeEnd);
            if (rangeBegin < cutBegin) _ranges.emplace(rangeBegin, Range{cutBegin, std::vector<std::uint8_t>(pages.begin(), pages.begin() + static_cast<std::ptrdiff_t>((cutBegin - rangeBegin) / PageSize))});
            if (cutEnd < rangeEnd) it = _ranges.emplace(cutEnd, Range{rangeEnd, std::vector<std::uint8_t>(pages.begin() + static_cast<std::ptrdiff_t>((cutEnd - rangeBegin) / PageSize), pages.end())}).first;
        }
    }

    static std::uintptr_t pageStart(std::uintptr_t address) {
        return address & ~(PageSize - 1);
    }

    static std::uintptr_t pageEnd(std::uintptr_t address) {
        return (address + PageSize - 1) & ~(PageSize - 1);
    }

    std::mutex _lock;
    std::map<std::uintptr_t, Range> _ranges;
    std::vector<Hold> _holds;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> _hostWrites;
    std::array<Retry, RetryCount> _retries{};
    std::uint64_t _retryUses = 0;
    std::array<Slot, SlotCount> _slots;
    std::atomic<unsigned> _wake{0};
    std::atomic<bool> _registered{false};
    std::atomic<std::size_t> _active{0};
    std::atomic<std::size_t> _dormant{0};
    std::uint64_t _sweep = 0;
    std::atomic<std::uint64_t> _epoch{0};
    std::atomic<void (*)(std::uintptr_t, std::uintptr_t)> _resolver{nullptr};
    std::atomic<bool (*)()> _bypass{nullptr};
    std::once_flag _resolverStarted;
    std::atomic<bool> _resolverRunning{false};
    std::atomic<pid_t> _resolverThread{0};
    Counters _counts;
    std::array<FaultEntry, FaultRing> _faultRing;
    std::atomic<std::uint64_t> _faultIndex{0};
};

const bool g_readTrapsReady = (ReadTrapTable::Get(), true);
#endif

#if defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC)
constexpr std::uintptr_t PageBytes = 4096;

char g_unavailableReason[192];
thread_local char t_registrationFailure[192];

class Watch {
public:
    static Watch& Get() {
        static auto* watch = new Watch;
        return *watch;
    }

    bool Available() const {
        return _pagemap >= 0 || _reported;
    }

    void Note(std::uintptr_t begin, std::uintptr_t end) {
        if ((!_reported && !_verified) || end <= begin) return;
        begin &= ~(PageBytes - 1);
        end = (end + PageBytes - 1) & ~(PageBytes - 1);
        const std::lock_guard lock(_notedLock);
        remove(_noted, begin, end);
        insert(_noted, begin, end);
    }

    void Register(std::uintptr_t begin, std::uintptr_t end) {
        if (!Available() || end <= begin) return;
        t_registrationFailure[0] = '\0';
        std::unique_lock lock(_lock);
        remove(_ranges, begin, end);
        remove(_fresh, begin, end);
        cut(begin, end);
        uffdio_register registration{};
        registration.range.start = begin;
        registration.range.len = end - begin;
        registration.mode = UFFDIO_REGISTER_MODE_WP;
        if (!_reported && ioctl(_uffd, UFFDIO_REGISTER, &registration) != 0) {
            const int error = errno;
            std::snprintf(t_registrationFailure, sizeof(t_registrationFailure), "UFFDIO_REGISTER of 0x%llx+0x%llx failed (%s, errno %d)", static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin), std::strerror(error), error);
            static bool reported = false;
            if (!reported) {
                reported = true;
                std::fprintf(stderr, "[memory] write watch: cannot register 0x%llx+0x%llx (%s); the range stays unwatched\n", static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end - begin), std::strerror(error));
            }
            return;
        }
        insert(_ranges, begin, end);
        insert(_fresh, begin, end);
    }

    static const char* RegistrationFailure() {
        return t_registrationFailure;
    }

    bool Unregister(std::uintptr_t begin, std::uintptr_t end) {
        if (!Available() || end <= begin) return false;
        std::unique_lock lock(_lock);
        const bool watched = remove(_ranges, begin, end);
        remove(_fresh, begin, end);
        cut(begin, end);
        return watched;
    }

    bool Alias(std::uintptr_t first, std::uintptr_t second, std::uintptr_t bytes) {
        if (!Available() || bytes == 0 || first == second || (first | second | bytes) % PageBytes != 0) return false;
        std::unique_lock lock(_lock);
        if (!covers(first, first + bytes) || !covers(second, second + bytes)) return false;
        link(first, first + bytes, second);
        link(second, second + bytes, first);
        return true;
    }

    bool Covers(std::uintptr_t begin, std::uintptr_t end) {
        if (!Available()) return false;
        std::shared_lock lock(_lock);
        return covers(begin, end);
    }

    bool Overlaps(std::uintptr_t begin, std::uintptr_t end) {
        if (!Available()) return false;
        std::shared_lock lock(_lock);
        const auto next = _ranges.upper_bound(begin);
        return (next != _ranges.begin() && std::prev(next)->second > begin) || (next != _ranges.end() && next->first < end);
    }

    bool Collect(std::uintptr_t begin, std::uintptr_t end, void (*written)(void*, std::uintptr_t, std::uintptr_t), void* context) {
        if (!Available()) return false;
        begin &= ~(PageBytes - 1);
        end = (end + PageBytes - 1) & ~(PageBytes - 1);
        if (end <= begin) return true;
        const auto plan = aliasPlan(begin, end);
        Fanout fanout{&plan.own, written, context};
        if (!takeFresh(begin, end, true, &Fanout::Report, &fanout) || !scanWritten(begin, end, &Fanout::Report, &fanout)) return false;
        for (const auto& alias : plan.aliases) {
            Fanout through{&alias.translations, written, context};
            if (!takeFresh(alias.begin, alias.end, false, &Fanout::Report, &through) || !scanWritten(alias.begin, alias.end, &Fanout::Report, &through)) return false;
        }
        return true;
    }

    bool CollectFresh(std::uintptr_t begin, std::uintptr_t end, void (*written)(void*, std::uintptr_t, std::uintptr_t), void* context) {
        if (!Available()) return false;
        begin &= ~(PageBytes - 1);
        end = (end + PageBytes - 1) & ~(PageBytes - 1);
        if (end <= begin) return true;
        const auto plan = aliasPlan(begin, end);
        if (plan.aliases.empty()) {
            std::shared_lock lock(_lock);
            auto it = _fresh.upper_bound(begin);
            if (it != _fresh.begin()) --it;
            while (it != _fresh.end() && it->first < end && it->second <= begin) ++it;
            if (it == _fresh.end() || it->first >= end) return true;
        }
        Fanout fanout{&plan.own, written, context};
        if (!takeFresh(begin, end, false, &Fanout::Report, &fanout)) return false;
        for (const auto& alias : plan.aliases) {
            Fanout through{&alias.translations, written, context};
            if (!takeFresh(alias.begin, alias.end, false, &Fanout::Report, &through)) return false;
        }
        return true;
    }

    bool CollectArmed(std::uintptr_t begin, std::uintptr_t end, void (*written)(void*, std::uintptr_t, std::uintptr_t), void* context) {
        if (!Available()) return false;
        begin &= ~(PageBytes - 1);
        end = (end + PageBytes - 1) & ~(PageBytes - 1);
        if (end <= begin) return true;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> armed;
        AliasPlan plan;
        std::vector<std::vector<std::pair<std::uintptr_t, std::uintptr_t>>> aliasArmed;
        {
            std::shared_lock lock(_lock);
            if (!covers(begin, end)) return false;
            armed = armedParts(begin, end);
            plan = aliasPlanLocked(begin, end);
            for (const auto& alias : plan.aliases) aliasArmed.push_back(armedParts(alias.begin, alias.end));
        }
        Fanout fanout{&plan.own, written, context};
        for (const auto& [from, to] : armed) {
            if (!scanWritten(from, to, &Fanout::Report, &fanout)) return false;
        }
        for (std::size_t index = 0; index < plan.aliases.size(); ++index) {
            Fanout through{&plan.aliases[index].translations, written, context};
            for (const auto& [from, to] : aliasArmed[index]) {
                if (!scanWritten(from, to, &Fanout::Report, &through)) return false;
            }
        }
        return true;
    }

private:
    struct Translation {
        std::uintptr_t begin;
        std::uintptr_t end;
        std::uintptr_t partner;
    };

    struct AliasRange {
        std::uintptr_t begin;
        std::uintptr_t end;
        std::vector<Translation> translations;
    };

    struct AliasPlan {
        std::vector<Translation> own;
        std::vector<AliasRange> aliases;
    };

    struct Fanout {
        const std::vector<Translation>* translations;
        void (*written)(void*, std::uintptr_t, std::uintptr_t);
        void* context;

        static void Report(void* self, std::uintptr_t begin, std::uintptr_t end) {
            auto& fanout = *static_cast<Fanout*>(self);
            fanout.written(fanout.context, begin, end);
            for (const auto& [from, to, partner] : *fanout.translations) {
                const auto left = std::max(from, begin);
                const auto right = std::min(to, end);
                if (left < right) fanout.written(fanout.context, partner + (left - from), partner + (right - from));
            }
        }
    };

    struct Link {
        std::uintptr_t end;
        std::uintptr_t partner;
    };

    AliasPlan aliasPlan(std::uintptr_t begin, std::uintptr_t end) {
        std::shared_lock lock(_lock);
        return aliasPlanLocked(begin, end);
    }

    AliasPlan aliasPlanLocked(std::uintptr_t begin, std::uintptr_t end) const {
        AliasPlan plan;
        plan.own = translations(begin, end);
        for (const auto& [from, to, partner] : plan.own) {
            const auto stop = partner + (to - from);
            plan.aliases.push_back({partner, stop, translations(partner, stop)});
        }
        return plan;
    }

    std::vector<Translation> translations(std::uintptr_t begin, std::uintptr_t end) const {
        std::vector<Translation> found;
        if (_links.empty()) return found;
        for (auto it = _links.lower_bound(begin > _linkBytes ? begin - _linkBytes : 0); it != _links.end() && it->first < end; ++it) {
            const auto from = std::max(it->first, begin);
            const auto to = std::min(it->second.end, end);
            if (from < to) found.push_back({from, to, it->second.partner + (from - it->first)});
        }
        return found;
    }

    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> armedParts(std::uintptr_t begin, std::uintptr_t end) const {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> armed;
        auto cursor = begin;
        auto it = _fresh.upper_bound(begin);
        if (it != _fresh.begin()) --it;
        for (; it != _fresh.end() && it->first < end; ++it) {
            if (it->second <= cursor) continue;
            if (it->first > cursor) armed.emplace_back(cursor, it->first);
            cursor = std::max(cursor, it->second);
        }
        if (cursor < end) armed.emplace_back(cursor, end);
        return armed;
    }

    void link(std::uintptr_t begin, std::uintptr_t end, std::uintptr_t partner) {
        _links.emplace(begin, Link{end, partner});
        _linkBytes = std::max(_linkBytes, end - begin);
    }

    void cut(std::uintptr_t begin, std::uintptr_t end) {
        std::vector<std::pair<std::uintptr_t, Link>> kept;
        for (auto it = _links.begin(); it != _links.end();) {
            const auto from = it->first;
            const auto bytes = it->second.end - from;
            const auto partner = it->second.partner;
            if (!(from < end && begin < from + bytes) && !(partner < end && begin < partner + bytes)) {
                ++it;
                continue;
            }
            it = _links.erase(it);
            std::vector<std::pair<std::uintptr_t, std::uintptr_t>> pieces{{0, bytes}};
            for (const auto base : {from, partner}) {
                const auto low = std::min(bytes, begin > base ? begin - base : 0);
                const auto high = std::min(bytes, end > base ? end - base : 0);
                if (low >= high) continue;
                std::vector<std::pair<std::uintptr_t, std::uintptr_t>> remaining;
                for (const auto& [left, right] : pieces) {
                    if (left < low) remaining.emplace_back(left, std::min(right, low));
                    if (right > high) remaining.emplace_back(std::max(left, high), right);
                }
                pieces = std::move(remaining);
            }
            for (const auto& [left, right] : pieces) kept.emplace_back(from + left, Link{from + right, partner + left});
        }
        for (const auto& [from, piece] : kept) _links.emplace(from, piece);
    }

    bool takeFresh(std::uintptr_t begin, std::uintptr_t end, bool whole, void (*written)(void*, std::uintptr_t, std::uintptr_t), void* context) {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> fresh;
        {
            std::unique_lock lock(_lock);
            if (whole && !covers(begin, end)) return false;
            auto it = _fresh.upper_bound(begin);
            if (it != _fresh.begin()) --it;
            for (; it != _fresh.end() && it->first < end; ++it) {
                const auto from = std::max(it->first, begin);
                const auto to = std::min(it->second, end);
                if (from < to) fresh.emplace_back(from, to);
            }
            for (const auto& [from, to] : fresh) remove(_fresh, from, to);
        }
        for (const auto& [from, to] : fresh) {
            written(context, from, to);
            if (protect(from, to)) continue;
            std::unique_lock lock(_lock);
            for (const auto& [left, right] : fresh) {
                if (!covers(left, right)) continue;
                remove(_fresh, left, right);
                insert(_fresh, left, right);
            }
            if (whole) written(context, begin, end);
            else for (const auto& [left, right] : fresh) written(context, left, right);
            return false;
        }
        return true;
    }

    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> takeNoted(std::uintptr_t begin, std::uintptr_t end) const {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> taken;
        const std::lock_guard lock(_notedLock);
        auto it = _noted.upper_bound(begin);
        if (it != _noted.begin()) --it;
        for (; it != _noted.end() && it->first < end; ++it) {
            const auto from = std::max(it->first, begin);
            const auto to = std::min(it->second, end);
            if (from < to) taken.emplace_back(from, to);
        }
        for (const auto& [from, to] : taken) remove(_noted, from, to);
        return taken;
    }

    void verifyNoted(std::uintptr_t begin, std::uintptr_t end, const page_region* regions, long count) const {
        const auto noted = takeNoted(begin, end);
        for (long i = 0; i < count; ++i) {
            for (auto page = static_cast<std::uintptr_t>(regions[i].start); page < regions[i].end; page += PageBytes) {
                if (std::any_of(noted.begin(), noted.end(), [&](const auto& range) { return range.first <= page && page < range.second; })) continue;
                const auto missed = _unreported.fetch_add(1, std::memory_order_relaxed) + 1;
                if (missed <= 32 || (missed & (missed - 1)) == 0) std::fprintf(stderr, "[memory] write watch: page 0x%llx was written without a report from the host (%llu so far)\n", static_cast<unsigned long long>(page), static_cast<unsigned long long>(missed));
            }
        }
    }

    bool scanWritten(std::uintptr_t begin, std::uintptr_t end, void (*written)(void*, std::uintptr_t, std::uintptr_t), void* context) const {
        if (_reported) {
            for (const auto& [from, to] : takeNoted(begin, end)) written(context, from, to);
            return true;
        }
        std::array<page_region, 256> regions;
        auto cursor = begin;
        while (cursor < end) {
            pm_scan_arg scan{};
            scan.size = sizeof(scan);
            scan.flags = PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC;
            scan.start = cursor;
            scan.end = end;
            scan.vec = reinterpret_cast<std::uintptr_t>(regions.data());
            scan.vec_len = regions.size();
            scan.category_mask = PAGE_IS_WRITTEN;
            scan.return_mask = PAGE_IS_WRITTEN;
            const auto count = ioctl(_pagemap, PAGEMAP_SCAN, &scan);
            if (count < 0) {
                written(context, cursor, end);
                return false;
            }
            for (long i = 0; i < count; ++i) written(context, regions[i].start, regions[i].end);
            if (_verified) verifyNoted(cursor, scan.walk_end > cursor ? std::min<std::uintptr_t>(scan.walk_end, end) : end, regions.data(), count);
            if (scan.walk_end <= cursor || scan.walk_end >= end) break;
            cursor = scan.walk_end;
        }
        return true;
    }

private:
    Watch() {
        if (GpuOwnsFaults || std::getenv("APS5_NO_WRITE_WATCH") != nullptr) {
            std::snprintf(g_unavailableReason, sizeof(g_unavailableReason), GpuOwnsFaults ? "the Kyty backend owns guest write faults" : "disabled by APS5_NO_WRITE_WATCH");
            return;
        }
        const char* oracle = std::getenv("APS5_WRITE_WATCH_ORACLE");
        if (oracle != nullptr && std::strcmp(oracle, "verify") != 0) {
            _reported = true;
            std::fprintf(stderr, "[memory] write watch: writes are reported by the host (APS5_WRITE_WATCH_ORACLE); the system does not track guest writes\n");
            return;
        }
        open();
        _verified = oracle != nullptr && _pagemap >= 0;
        if (_verified) std::fprintf(stderr, "[memory] write watch: every written page is checked against the reports of the host (APS5_WRITE_WATCH_ORACLE=verify)\n");
    }

    void open() {
        _uffd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK));
        if (_uffd < 0 && errno == EPERM) _uffd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
        if (_uffd < 0) return unavailable("userfaultfd", errno);
        uffdio_api api{};
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_WP_ASYNC | UFFD_FEATURE_WP_UNPOPULATED;
        if (ioctl(_uffd, UFFDIO_API, &api) != 0) return unavailable("asynchronous userfaultfd write protection", errno);
        if ((api.features & UFFD_FEATURE_WP_ASYNC) == 0 || (api.features & UFFD_FEATURE_WP_UNPOPULATED) == 0) return unavailable("asynchronous userfaultfd write protection (UFFD_FEATURE_WP_ASYNC and UFFD_FEATURE_WP_UNPOPULATED)", 0);
        const int pagemap = ::open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
        if (pagemap < 0) return unavailable("/proc/self/pagemap", errno);
        if (!probe(pagemap)) {
            const int error = errno;
            close(pagemap);
            return unavailable("PAGEMAP_SCAN", error);
        }
        _pagemap = pagemap;
    }

    void unavailable(const char* what, int error) {
        if (_uffd >= 0) close(_uffd);
        _uffd = -1;
        if (error != 0) {
            std::snprintf(g_unavailableReason, sizeof(g_unavailableReason), "%s failed (%s, errno %d); the running kernel needs PAGEMAP_SCAN and asynchronous userfaultfd write protection (Linux 6.7 or newer)", what, std::strerror(error), error);
            std::fprintf(stderr, "[memory] write watch unavailable: %s failed (%s); guest memory is compared instead\n", what, std::strerror(error));
        } else {
            std::snprintf(g_unavailableReason, sizeof(g_unavailableReason), "%s is not offered by the running kernel; it needs PAGEMAP_SCAN and asynchronous userfaultfd write protection (Linux 6.7 or newer)", what);
            std::fprintf(stderr, "[memory] write watch unavailable: %s is not offered by the kernel; guest memory is compared instead\n", what);
        }
    }

    bool protect(std::uintptr_t begin, std::uintptr_t end) const {
        if (_reported) return true;
        uffdio_writeprotect protection{};
        protection.range.start = begin;
        protection.range.len = end - begin;
        protection.mode = UFFDIO_WRITEPROTECT_MODE_WP;
        return ioctl(_uffd, UFFDIO_WRITEPROTECT, &protection) == 0;
    }

    bool probe(int pagemap) {
        constexpr std::uintptr_t probeBytes = 4 * PageBytes;
        void* pages = mmap(nullptr, probeBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (pages == MAP_FAILED) return false;
        const auto address = reinterpret_cast<std::uintptr_t>(pages);
        auto* bytes = static_cast<volatile char*>(pages);
        bytes[0] = 1;
        uffdio_register registration{};
        registration.range.start = address;
        registration.range.len = probeBytes;
        registration.mode = UFFDIO_REGISTER_MODE_WP;
        bool working = ioctl(_uffd, UFFDIO_REGISTER, &registration) == 0 && protect(address, address + probeBytes);
        std::array<page_region, 4> regions{};
        const auto scan = [&]() -> long {
            pm_scan_arg arguments{};
            arguments.size = sizeof(arguments);
            arguments.flags = PM_SCAN_WP_MATCHING | PM_SCAN_CHECK_WPASYNC;
            arguments.start = address;
            arguments.end = address + probeBytes;
            arguments.vec = reinterpret_cast<std::uintptr_t>(regions.data());
            arguments.vec_len = regions.size();
            arguments.category_mask = PAGE_IS_WRITTEN;
            arguments.return_mask = PAGE_IS_WRITTEN;
            const auto count = ioctl(pagemap, PAGEMAP_SCAN, &arguments);
            if (count < 0) return -1;
            long pagesWritten = 0;
            for (long i = 0; i < count; ++i) pagesWritten += static_cast<long>((regions[i].end - regions[i].start) / PageBytes);
            return pagesWritten;
        };
        working = working && scan() == 0;
        if (working) {
            bytes[0] = 2;
            bytes[2 * PageBytes] = 1;
            working = scan() == 2 && scan() == 0;
        }
        munmap(pages, probeBytes);
        return working;
    }

    bool covers(std::uintptr_t begin, std::uintptr_t end) const {
        auto next = _ranges.upper_bound(begin);
        if (next == _ranges.begin()) return false;
        return std::prev(next)->second >= end;
    }

    static void insert(std::map<std::uintptr_t, std::uintptr_t>& ranges, std::uintptr_t begin, std::uintptr_t end) {
        auto next = ranges.upper_bound(begin);
        if (next != ranges.begin() && std::prev(next)->second == begin) {
            begin = std::prev(next)->first;
            ranges.erase(std::prev(next));
        }
        if (next != ranges.end() && next->first == end) {
            end = next->second;
            ranges.erase(next);
        }
        ranges.emplace(begin, end);
    }

    static bool remove(std::map<std::uintptr_t, std::uintptr_t>& ranges, std::uintptr_t begin, std::uintptr_t end) {
        bool removed = false;
        auto it = ranges.upper_bound(begin);
        if (it != ranges.begin()) --it;
        while (it != ranges.end() && it->first < end) {
            const auto rangeBegin = it->first;
            const auto rangeEnd = it->second;
            if (rangeEnd <= begin) {
                ++it;
                continue;
            }
            removed = true;
            it = ranges.erase(it);
            if (rangeBegin < begin) ranges.emplace(rangeBegin, begin);
            if (rangeEnd > end) ranges.emplace(end, rangeEnd);
        }
        return removed;
    }

    int _uffd = -1;
    int _pagemap = -1;
    bool _reported = false;
    bool _verified = false;
    mutable std::mutex _notedLock;
    mutable std::map<std::uintptr_t, std::uintptr_t> _noted;
    mutable std::atomic<std::uint64_t> _unreported{0};
    std::shared_mutex _lock;
    std::map<std::uintptr_t, std::uintptr_t> _ranges;
    std::map<std::uintptr_t, std::uintptr_t> _fresh;
    std::multimap<std::uintptr_t, Link> _links;
    std::uintptr_t _linkBytes = 0;
};

const bool g_opened = (Watch::Get(), true);

#if defined(APS5_LINUX_READ_TRAPS)
class TrapLock {
public:
    void lock() {
        while (_held.test_and_set(std::memory_order_acquire)) sched_yield();
    }

    void unlock() {
        _held.clear(std::memory_order_release);
    }

private:
    std::atomic_flag _held = ATOMIC_FLAG_INIT;
};

class ReadTraps {
public:
    static ReadTraps& Get() {
        static auto* traps = new ReadTraps;
        return *traps;
    }

    void Registered(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin) return;
        const std::lock_guard lock(_lock);
        if (std::erase_if(_traps, [&](const Trap& trap) { return trap.owner == 0 && trap.first < pageEnd(end) && pageStart(begin) < trap.last; }) != 0) _dropped.fetch_add(1, std::memory_order_release);
        eraseProtections(begin, end);
        cut(_unlearned, begin, end);
        _unlearned.emplace(begin, end);
    }

    void Forget(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin) return;
        const std::lock_guard lock(_lock);
        std::vector<Trap> dropped;
        for (auto it = _traps.begin(); it != _traps.end();) {
            if (it->owner == 0 && it->first < pageEnd(end) && pageStart(begin) < it->last) {
                dropped.push_back(std::move(*it));
                it = _traps.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto& trap : dropped) restore(trap);
        if (!dropped.empty()) _dropped.fetch_add(1, std::memory_order_release);
        eraseProtections(begin, end);
        cut(_unlearned, begin, end);
    }

    bool SetResolver(void (*resolver)(std::uintptr_t, std::uintptr_t)) {
        if (resolver == nullptr || !Watch::Get().Available()) return false;
        std::call_once(_started, [this] { start(); });
        if (!_running.load(std::memory_order_acquire)) return false;
        _resolver.store(resolver, std::memory_order_release);
        return true;
    }

    void SetBypass(bool (*bypass)()) {
        _bypass.store(bypass, std::memory_order_release);
    }

    void Trapped(std::uintptr_t begin, std::uintptr_t end, void (*trapped)(void*, std::uintptr_t, std::uintptr_t), void* context) {
        if (end <= begin || !_trapping.load(std::memory_order_acquire)) return;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
        {
            const std::lock_guard lock(_lock);
            for (auto page = pageStart(begin); page < pageEnd(end); page += PageBytes) {
                if (!closed(page)) continue;
                if (!runs.empty() && runs.back().second == page) runs.back().second = page + PageBytes;
                else runs.emplace_back(page, page + PageBytes);
            }
        }
        for (const auto& [runBegin, runEnd] : runs) trapped(context, runBegin, runEnd);
    }

    bool TrapReads(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin || _resolver.load(std::memory_order_acquire) == nullptr) return false;
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        const auto held = [&] { return std::any_of(_traps.begin(), _traps.end(), [&](const Trap& trap) { return trap.owner == 0 && trap.begin == begin && trap.end == end; }); };
        {
            const std::lock_guard lock(_lock);
            if (held()) return true;
        }
        if (!Watch::Get().Covers(first, last)) return false;
        learn(first, last);
        Trap trap{begin, end, first, last, 0, 0, std::vector<std::uint8_t>((last - first) / PageBytes, 0)};
        const std::lock_guard lock(_lock);
        if (held()) return true;
        for (const auto& [hostBegin, hostEnd] : _hostWrites)
            if (hostBegin < last && first < hostEnd) return false;
        if (!protectionIs(first, last, PROT_READ | PROT_WRITE)) return false;
        if (std::any_of(_traps.begin(), _traps.end(), [&](const Trap& held) { return held.owner != 0 && held.first < last && first < held.last; })) return false;
        if (::mprotect(reinterpret_cast<void*>(first), last - first, PROT_NONE) != 0) return false;
        _traps.push_back(std::move(trap));
        _trapping.store(true, std::memory_order_release);
        return true;
    }

    std::uint64_t Generation() const {
        return _dropped.load(std::memory_order_acquire);
    }

    void ReleaseReads(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin) return;
        const std::lock_guard lock(_lock);
        const auto found = std::find_if(_traps.begin(), _traps.end(), [&](const Trap& trap) { return trap.owner == 0 && trap.begin == begin && trap.end == end; });
        if (found == _traps.end()) return;
        const auto released = std::move(*found);
        _traps.erase(found);
        _dropped.fetch_add(1, std::memory_order_release);
        restore(released);
    }

    bool HandleFault(std::uintptr_t address) {
        if (!_trapping.load(std::memory_order_acquire)) return false;
        const KeepErrno keep;
        const auto page = pageStart(address);
        const auto self = CurrentThread();
        {
            const std::lock_guard lock(_lock);
            if (!closed(page)) return retryOpen(page);
            const auto bypass = _bypass.load(std::memory_order_acquire);
            if (self == _resolverThread.load(std::memory_order_acquire) || ownedBy(page, self) || (bypass != nullptr && bypass())) {
                for (auto& trap : _traps)
                    if (trap.first <= page && page < trap.last) trap.open[(page - trap.first) / PageBytes] = 1;
                ++_generation;
                return ::mprotect(reinterpret_cast<void*>(page), PageBytes, PROT_READ | PROT_WRITE) == 0;
            }
        }
        Slot* slot = nullptr;
        while (slot == nullptr) {
            for (auto& candidate : _slots) {
                unsigned expected = Free;
                if (candidate.state.compare_exchange_strong(expected, Claimed, std::memory_order_acq_rel)) {
                    slot = &candidate;
                    break;
                }
            }
            if (slot == nullptr) sched_yield();
        }
        slot->page = page;
        slot->thread.store(self, std::memory_order_release);
        slot->state.store(Queued, std::memory_order_release);
        sem_post(&_wake);
        while (sem_wait(&slot->done) != 0) {}
        slot->thread.store(0, std::memory_order_release);
        slot->state.store(Free, std::memory_order_release);
        return true;
    }

    bool Awaits(pid_t thread) const {
        if (thread == 0) return false;
        return std::any_of(_slots.begin(), _slots.end(), [&](const Slot& slot) { return slot.thread.load(std::memory_order_acquire) == thread; });
    }

    int Protect(std::uintptr_t begin, std::uintptr_t end, int protection) {
        return withoutTraps(begin, end, [&] {
            if (unlearnedOverlaps(begin, end)) ++_protects;
            if (::mprotect(reinterpret_cast<void*>(begin), end - begin, protection) != 0) return errno;
            reprotect(pageStart(begin), pageEnd(end), protection);
            return 0;
        });
    }

    void BeginHostWrite(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin) return;
        withoutTraps(begin, end, [&] {
            _hostWrites.emplace_back(pageStart(begin), pageEnd(end));
            return 0;
        });
    }

    void EndHostWrite(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin) return;
        const std::lock_guard lock(_lock);
        const auto found = std::find(_hostWrites.begin(), _hostWrites.end(), std::pair{pageStart(begin), pageEnd(end)});
        if (found != _hostWrites.end()) _hostWrites.erase(found);
    }

    bool BeginUntrapped(std::uintptr_t begin, std::uintptr_t end) {
        if (end <= begin) return true;
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        const std::lock_guard lock(_lock);
        if (std::any_of(_traps.begin(), _traps.end(), [&](const Trap& trap) { return trap.first < last && first < trap.last; })) return false;
        _hostWrites.emplace_back(first, last);
        return true;
    }

private:
    static constexpr std::size_t ResolveSlots = 32;
    static constexpr unsigned UntrappedRetries = 16;
    static constexpr unsigned PageRetries = 1024;

    enum SlotState : unsigned { Free, Claimed, Queued, Running, Done };

    struct Slot {
        std::atomic<unsigned> state{Free};
        std::atomic<pid_t> thread{0};
        std::uintptr_t page = 0;
        sem_t done{};
    };

    struct Trap {
        std::uintptr_t begin;
        std::uintptr_t end;
        std::uintptr_t first;
        std::uintptr_t last;
        pid_t owner;
        std::uint64_t batch;
        std::vector<std::uint8_t> open;
    };

    struct Protection {
        std::uintptr_t end;
        int protection;
    };

    struct Span {
        std::uintptr_t begin;
        std::uintptr_t end;
        int protection;
    };

    struct KeepErrno {
        int saved = errno;
        ~KeepErrno() { errno = saved; }
    };

    static pid_t CurrentThread() {
        return static_cast<pid_t>(::syscall(SYS_gettid));
    }

    static std::uintptr_t pageStart(std::uintptr_t address) {
        return address & ~(PageBytes - 1);
    }

    static std::uintptr_t pageEnd(std::uintptr_t address) {
        return (address + PageBytes - 1) & ~(PageBytes - 1);
    }

    static std::vector<Span> mappedProtections(std::uintptr_t begin, std::uintptr_t end) {
        std::vector<Span> spans;
        const int file = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
        if (file < 0) return spans;
        std::string text;
        std::array<char, 65536> buffer{};
        for (;;) {
            const auto count = ::read(file, buffer.data(), buffer.size());
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) break;
            text.append(buffer.data(), static_cast<std::size_t>(count));
        }
        ::close(file);
        std::size_t at = 0;
        while (at < text.size()) {
            const auto lineEnd = std::min(text.find('\n', at), text.size());
            const char* cursor = text.c_str() + at;
            char* parsed = nullptr;
            const auto first = static_cast<std::uintptr_t>(std::strtoull(cursor, &parsed, 16));
            if (parsed != nullptr && *parsed == '-') {
                const auto last = static_cast<std::uintptr_t>(std::strtoull(parsed + 1, &parsed, 16));
                if (parsed != nullptr && *parsed == ' ' && first < end && begin < last && static_cast<std::size_t>(parsed - text.c_str()) + 4 < lineEnd) {
                    const int protection = (parsed[1] == 'r' ? PROT_READ : 0) | (parsed[2] == 'w' ? PROT_WRITE : 0) | (parsed[3] == 'x' ? PROT_EXEC : 0);
                    spans.push_back({std::max(first, begin), std::min(last, end), protection});
                }
            }
            at = lineEnd + 1;
        }
        return spans;
    }

    void learn(std::uintptr_t first, std::uintptr_t last) {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> pending;
        std::uint64_t protects = 0;
        {
            const std::lock_guard lock(_lock);
            auto it = _unlearned.upper_bound(first);
            if (it != _unlearned.begin()) --it;
            for (; it != _unlearned.end() && it->first < last; ++it)
                if (it->second > first) pending.emplace_back(*it);
            protects = _protects;
        }
        for (const auto& [begin, end] : pending) {
            const auto spans = mappedProtections(begin, end);
            const std::lock_guard lock(_lock);
            const auto found = _unlearned.find(begin);
            if (found == _unlearned.end() || found->second != end || _protects != protects) continue;
            _unlearned.erase(found);
            for (const auto& span : spans) _protections.emplace(span.begin, Protection{span.end, span.protection});
        }
    }

    bool unlearnedOverlaps(std::uintptr_t begin, std::uintptr_t end) const {
        const auto next = _unlearned.upper_bound(begin);
        return (next != _unlearned.begin() && std::prev(next)->second > begin) || (next != _unlearned.end() && next->first < end);
    }

    static void cut(std::map<std::uintptr_t, std::uintptr_t>& ranges, std::uintptr_t begin, std::uintptr_t end) {
        auto it = ranges.upper_bound(begin);
        if (it != ranges.begin()) --it;
        while (it != ranges.end() && it->first < end) {
            const auto rangeBegin = it->first;
            const auto rangeEnd = it->second;
            if (rangeEnd <= begin) {
                ++it;
                continue;
            }
            it = ranges.erase(it);
            if (rangeBegin < begin) ranges.emplace(rangeBegin, begin);
            if (rangeEnd > end) it = ranges.emplace(end, rangeEnd).first;
        }
    }

    void start() {
        if (sem_init(&_wake, 0, 0) != 0) return;
        for (auto& slot : _slots)
            if (sem_init(&slot.done, 0, 0) != 0) return;
        struct sigaction action{};
        action.sa_sigaction = &ReadTraps::fault;
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&action.sa_mask);
        pthread_t thread{};
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
        const bool created = pthread_create(&thread, &attributes, [](void* traps) -> void* {
            static_cast<ReadTraps*>(traps)->resolveLoop();
            return nullptr;
        }, this) == 0;
        pthread_attr_destroy(&attributes);
        if (!created) return;
        while (_resolverThread.load(std::memory_order_acquire) == 0) sched_yield();
        if (sigaction(SIGSEGV, &action, &_previousSegv) != 0 || sigaction(SIGBUS, &action, &_previousBus) != 0) return;
        _running.store(true, std::memory_order_release);
    }

    static void fault(int signal, siginfo_t* info, void* context) {
        auto& traps = Get();
        if (info->si_code > 0 && traps.HandleFault(reinterpret_cast<std::uintptr_t>(info->si_addr))) return;
        const auto& previous = signal == SIGBUS ? traps._previousBus : traps._previousSegv;
        if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
            previous.sa_sigaction(signal, info, context);
            return;
        }
        if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN && previous.sa_handler != nullptr) {
            previous.sa_handler(signal);
            return;
        }
        ::signal(signal, SIG_DFL);
        if (info->si_code <= 0) ::raise(signal);
    }

    void resolveLoop() {
        pthread_setname_np(pthread_self(), "read traps");
        _resolverThread.store(CurrentThread(), std::memory_order_release);
        for (;;) {
            while (sem_wait(&_wake) != 0) {}
            for (auto& slot : _slots) {
                unsigned expected = Queued;
                if (!slot.state.compare_exchange_strong(expected, Running, std::memory_order_acq_rel)) continue;
                const auto page = slot.page;
                waitForResolving(page, page + PageBytes);
                resolve(page, page + PageBytes);
                slot.state.store(Done, std::memory_order_release);
                sem_post(&slot.done);
            }
        }
    }

    void resolve(std::uintptr_t begin, std::uintptr_t end) {
        const auto self = CurrentThread();
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> taken;
        std::uint64_t batch = 0;
        {
            const std::lock_guard lock(_lock);
            batch = ++_batches;
            auto first = pageStart(begin);
            auto last = pageEnd(end);
            for (bool grew = true; grew;) {
                grew = false;
                for (auto& trap : _traps) {
                    if (trap.owner != 0 || trap.last <= first || last <= trap.first) continue;
                    trap.owner = self;
                    trap.batch = batch;
                    _dropped.fetch_add(1, std::memory_order_release);
                    taken.emplace_back(trap.begin, trap.end);
                    first = std::min(first, trap.first);
                    last = std::max(last, trap.last);
                    grew = true;
                }
            }
        }
        if (taken.empty()) return;
        const auto resolver = _resolver.load(std::memory_order_acquire);
        for (const auto& [first, last] : taken) resolver(first, last);
        const std::lock_guard lock(_lock);
        std::vector<Trap> resolved;
        for (auto it = _traps.begin(); it != _traps.end();) {
            if (it->batch == batch) {
                resolved.push_back(std::move(*it));
                it = _traps.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto& trap : resolved) restore(trap);
    }

    void waitForResolving(std::uintptr_t begin, std::uintptr_t end) {
        const auto self = CurrentThread();
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        for (;;) {
            {
                const std::lock_guard lock(_lock);
                if (std::none_of(_traps.begin(), _traps.end(), [&](const Trap& trap) { return trap.owner != 0 && trap.owner != self && trap.first < last && first < trap.last; })) return;
            }
            sched_yield();
        }
    }

    template <typename TFunction>
    int withoutTraps(std::uintptr_t begin, std::uintptr_t end, TFunction function) {
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        const auto self = CurrentThread();
        for (;;) {
            if (_trapping.load(std::memory_order_acquire)) {
                waitForResolving(first, last);
                resolve(first, last);
            }
            const std::lock_guard lock(_lock);
            if (std::any_of(_traps.begin(), _traps.end(), [&](const Trap& trap) { return trap.owner != self && trap.first < last && first < trap.last; })) continue;
            return function();
        }
    }

    bool closed(std::uintptr_t page) const {
        return std::any_of(_traps.begin(), _traps.end(), [&](const Trap& trap) { return trap.first <= page && page < trap.last && trap.open[(page - trap.first) / PageBytes] == 0; });
    }

    bool ownedBy(std::uintptr_t page, pid_t thread) const {
        return std::any_of(_traps.begin(), _traps.end(), [&](const Trap& trap) { return trap.owner == thread && trap.first <= page && page < trap.last; });
    }

    bool retryOpen(std::uintptr_t page) {
        if (!protectionIs(page, page + PageBytes, PROT_READ | PROT_WRITE)) return false;
        thread_local std::uintptr_t retryPage = 0;
        thread_local std::uint64_t retryGeneration = 0;
        thread_local unsigned retries = 0;
        thread_local unsigned pageRetries = 0;
        if (retryPage != page) {
            retryPage = page;
            pageRetries = 0;
            retryGeneration = _generation;
            retries = 0;
        } else if (retryGeneration != _generation) {
            retryGeneration = _generation;
            retries = 0;
        }
        return ++pageRetries <= PageRetries && ++retries <= UntrappedRetries;
    }

    void restore(const Trap& removed) {
        std::uintptr_t runStart = 0;
        std::uintptr_t runEnd = 0;
        const auto flush = [&] {
            if (runEnd > runStart) ::mprotect(reinterpret_cast<void*>(runStart), runEnd - runStart, PROT_READ | PROT_WRITE);
            runStart = runEnd = 0;
        };
        for (auto page = removed.first; page < removed.last; page += PageBytes) {
            const bool reopen = removed.open[(page - removed.first) / PageBytes] == 0 && !closed(page);
            if (!reopen) {
                flush();
                continue;
            }
            if (runEnd != page) {
                flush();
                runStart = page;
            }
            runEnd = page + PageBytes;
        }
        flush();
        ++_generation;
    }

    bool protectionIs(std::uintptr_t begin, std::uintptr_t end, int protection) const {
        auto it = _protections.upper_bound(begin);
        if (it == _protections.begin()) return false;
        --it;
        for (auto cursor = begin; cursor < end; ++it) {
            if (it == _protections.end() || it->first > cursor || it->second.end <= cursor || it->second.protection != protection) return false;
            cursor = it->second.end;
        }
        return true;
    }

    void eraseProtections(std::uintptr_t begin, std::uintptr_t end) {
        auto it = _protections.upper_bound(begin);
        if (it != _protections.begin()) --it;
        while (it != _protections.end() && it->first < end) {
            const auto spanBegin = it->first;
            const auto span = it->second;
            if (span.end <= begin) {
                ++it;
                continue;
            }
            it = _protections.erase(it);
            if (spanBegin < begin) _protections.emplace(spanBegin, Protection{begin, span.protection});
            if (span.end > end) it = _protections.emplace(end, Protection{span.end, span.protection}).first;
        }
    }

    void reprotect(std::uintptr_t begin, std::uintptr_t end, int protection) {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> known;
        auto it = _protections.upper_bound(begin);
        if (it != _protections.begin()) --it;
        for (; it != _protections.end() && it->first < end; ++it)
            if (it->second.end > begin) known.emplace_back(std::max(it->first, begin), std::min(it->second.end, end));
        eraseProtections(begin, end);
        for (const auto& [first, last] : known) _protections.emplace(first, Protection{last, protection});
    }

    TrapLock _lock;
    std::vector<Trap> _traps;
    std::atomic<std::uint64_t> _dropped{1};
    std::map<std::uintptr_t, Protection> _protections;
    std::map<std::uintptr_t, std::uintptr_t> _unlearned;
    std::uint64_t _protects = 0;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> _hostWrites;
    std::uint64_t _generation = 0;
    std::uint64_t _batches = 0;
    std::atomic<bool> _trapping{false};
    std::atomic<void (*)(std::uintptr_t, std::uintptr_t)> _resolver{nullptr};
    std::atomic<bool (*)()> _bypass{nullptr};
    std::once_flag _started;
    std::atomic<bool> _running{false};
    std::atomic<pid_t> _resolverThread{0};
    sem_t _wake{};
    std::array<Slot, ResolveSlots> _slots;
    struct sigaction _previousBus {};
    struct sigaction _previousSegv {};
};

const bool g_trapsOpened = (ReadTraps::Get(), true);
#endif
#elif defined(__APPLE__)
class Watch {
public:
    static Watch& Get() {
        static auto* watch = new Watch;
        return *watch;
    }

    bool Available() const {
        return _available;
    }

    void Register(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available || end <= begin) return;
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin) return;
        std::vector<std::uint8_t> pages((end - begin) / _page, Written | Fresh);
        std::vector<bool> queried(pages.size(), false);
        const Lock lock(_lock);
        for (auto cursor = begin; cursor < end;) {
            mach_vm_address_t address = cursor;
            mach_vm_size_t size = 0;
            vm_region_basic_info_data_64_t info{};
            mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object = MACH_PORT_NULL;
            if (mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64, reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS || address >= end) break;
            const auto first = std::max<std::uintptr_t>(cursor, address);
            const auto last = std::min<std::uintptr_t>(end, address + size);
            const std::uint8_t protection = ((info.protection & VM_PROT_READ) ? PROT_READ : 0) | ((info.protection & VM_PROT_WRITE) ? PROT_WRITE : 0) | ((info.protection & VM_PROT_EXECUTE) ? PROT_EXEC : 0);
            for (auto page = first; page < last; page += _page) {
                pages[(page - begin) / _page] = Written | Fresh | protection;
                queried[(page - begin) / _page] = true;
            }
            cursor = last;
        }
        forEachPage(begin, end, [&](std::uintptr_t page, std::uint8_t& state) {
            auto& entry = pages[(page - begin) / _page];
            const std::uint8_t recorded = state & ProtectionMask;
            const bool armed = (state & Protected) != 0 && (entry & ProtectionMask) == (recorded & ~PROT_WRITE);
            if (armed || (state & ReadTrapped) != 0 || !queried[(page - begin) / _page]) entry = static_cast<std::uint8_t>(Written | Fresh | recorded);
            else state = static_cast<std::uint8_t>(state & ~Protected);
        });
        dropTraps(begin, end);
        erase(begin, end);
        _ranges.emplace(begin, Range{end, std::move(pages)});
    }

    bool Unregister(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available || end <= begin) return false;
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin) return false;
        const Lock lock(_lock);
        dropTraps(begin, end);
        return erase(begin, end);
    }

    bool Covers(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available) return false;
        const Lock lock(_lock);
        return covers(begin, end);
    }

    bool Overlaps(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available) return false;
        const Lock lock(_lock);
        const auto next = _ranges.upper_bound(begin);
        return (next != _ranges.begin() && std::prev(next)->second.end > begin) || (next != _ranges.end() && next->first < end);
    }

    bool HandleFault(std::uintptr_t address) {
        return _available && record(address);
    }

    enum class Take { All, Armed, Fresh };

    bool Collect(std::uintptr_t begin, std::uintptr_t end, void (*written)(void*, std::uintptr_t, std::uintptr_t), void* context, Take take = Take::All) {
        if (!_available) return false;
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin) return true;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
        {
            const Lock lock(_lock);
            if (take != Take::Fresh && !covers(begin, end)) return false;
            forEachPage(begin, end, [&](std::uintptr_t page, std::uint8_t& state) {
                if ((state & Written) == 0) return;
                const bool fresh = (state & Fresh) != 0;
                if ((take == Take::Armed && fresh) || (take == Take::Fresh && !fresh)) return;
                state &= static_cast<std::uint8_t>(~Fresh);
                if (!runs.empty() && runs.back().second == page) runs.back().second = page + _page;
                else runs.emplace_back(page, page + _page);
                for (const auto& [first, last] : _hostWrites)
                    if (page >= first && page < last) return;
                state &= static_cast<std::uint8_t>(~Written);
                if ((state & PROT_WRITE) != 0) state |= Protected;
            });
            for (const auto& [first, last] : runs) protect(first, last);
        }
        for (const auto& [first, last] : runs) written(context, first, last);
        return true;
    }

    void ProtectionChanged(std::uintptr_t begin, std::uintptr_t end, int protection) {
        if (!_available || end <= begin) return;
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin) return;
        withoutTraps(begin, end, false, [&] {
            forEachPage(begin, end, [&](std::uintptr_t, std::uint8_t& state) {
                state = static_cast<std::uint8_t>(Written | (state & Fresh) | (protection & ProtectionMask));
            });
        });
    }

    int Protect(std::uintptr_t begin, std::uintptr_t end, int protection) {
        if (!_available) return ::mprotect(reinterpret_cast<void*>(begin), end - begin, protection) == 0 ? 0 : errno;
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        return withoutTraps(first, last, true, [&] {
            if (::mprotect(reinterpret_cast<void*>(begin), end - begin, protection) != 0) return errno;
            if (first < last) {
                forEachPage(first, last, [&](std::uintptr_t, std::uint8_t& state) {
                    state = static_cast<std::uint8_t>(Written | (state & Fresh) | (protection & ProtectionMask));
                });
            }
            return 0;
        });
    }

    void BeginHostWrite(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available || end <= begin) return;
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin) return;
        withoutTraps(begin, end, true, [&] {
            _hostWrites.emplace_back(begin, end);
            forEachPage(begin, end, [&](std::uintptr_t page, std::uint8_t& state) {
                if ((state & Protected) != 0) ::mprotect(reinterpret_cast<void*>(page), _page, state & ProtectionMask);
                state = static_cast<std::uint8_t>((state & (ProtectionMask | Fresh)) | Written);
            });
        });
    }

    void EndHostWrite(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available || end <= begin) return;
        begin = pageStart(begin);
        end = pageEnd(end);
        if (end <= begin) return;
        const Lock lock(_lock);
        const auto found = std::find(_hostWrites.begin(), _hostWrites.end(), std::pair{begin, end});
        if (found != _hostWrites.end()) _hostWrites.erase(found);
    }

    int Protection(std::uintptr_t address, std::uintptr_t limit, std::uintptr_t* runEnd) {
        const auto page = pageStart(address);
        const Lock lock(_lock);
        auto it = _ranges.upper_bound(page);
        if (_available && it != _ranges.begin() && page < std::prev(it)->second.end) {
            --it;
            const auto& pages = it->second.pages;
            const std::uint8_t protection = pages[(page - it->first) / _page] & ProtectionMask;
            auto end = page + _page;
            while (end < it->second.end && end < limit && (pages[(end - it->first) / _page] & ProtectionMask) == protection) end += _page;
            *runEnd = std::min(end, limit);
            return protection;
        }
        *runEnd = it != _ranges.end() ? std::min(it->first, limit) : limit;
        return -1;
    }

    bool SetReadTrapResolver(void (*resolver)(std::uintptr_t, std::uintptr_t)) {
        if (!_available || resolver == nullptr) return false;
        std::call_once(_resolverStarted, [this] { startResolver(); });
        if (!_resolverRunning.load(std::memory_order_acquire)) return false;
        _resolver.store(resolver, std::memory_order_release);
        return true;
    }

    void SetReadTrapBypass(bool (*bypass)()) {
        _bypass.store(bypass, std::memory_order_release);
    }

    bool TrapReads(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available || end <= begin || _resolver.load(std::memory_order_acquire) == nullptr) return false;
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        const Lock lock(_lock);
        if (!covers(first, last)) return false;
        const auto [same, other] = _traps.equal_range(begin);
        if (std::any_of(same, other, [&](const auto& trap) { return trap.second == end; })) return true;
        for (const auto& [hostBegin, hostEnd] : _hostWrites)
            if (hostBegin < last && first < hostEnd) return false;
        if (::mprotect(reinterpret_cast<void*>(first), last - first, PROT_NONE) != 0) return false;
        _traps.emplace(begin, end);
        forEachPage(first, last, [](std::uintptr_t, std::uint8_t& state) { state |= ReadTrapped; });
        _trapping.store(true, std::memory_order_release);
        return true;
    }

    void ReadTraps(std::uintptr_t begin, std::uintptr_t end, void (*trapped)(void*, std::uintptr_t, std::uintptr_t), void* context) {
        if (!_available || end <= begin || !_trapping.load(std::memory_order_acquire)) return;
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
        {
            const Lock lock(_lock);
            forEachPage(pageStart(begin), pageEnd(end), [&](std::uintptr_t page, std::uint8_t& state) {
                if ((state & ReadTrapped) == 0) return;
                if (!runs.empty() && runs.back().second == page) runs.back().second = page + _page;
                else runs.emplace_back(page, page + _page);
            });
        }
        for (const auto& [runBegin, runEnd] : runs) trapped(context, runBegin, runEnd);
    }

    void ReleaseReads(std::uintptr_t begin, std::uintptr_t end) {
        if (!_available || end <= begin) return;
        const Lock lock(_lock);
        const auto [first, last] = _traps.equal_range(begin);
        const auto found = std::find_if(first, last, [&](const auto& trap) { return trap.second == end; });
        if (found == last) return;
        _traps.erase(found);
        untrap(begin, end);
    }

private:
    static constexpr std::uint8_t ProtectionMask = PROT_READ | PROT_WRITE | PROT_EXEC;
    static constexpr std::uint8_t Written = 0x10;
    static constexpr std::uint8_t Protected = 0x20;
    static constexpr std::uint8_t Fresh = 0x40;
    static constexpr std::uintptr_t FaultAheadBlock = 65536;
    static constexpr std::uint8_t ReadTrapped = 0x80;
    static constexpr std::size_t ResolveSlots = 32;
    static constexpr std::size_t ResolvingCapacity = 4096;
    static constexpr std::size_t RetrySlots = 64;
    static constexpr unsigned UntrappedRetries = 16;
    static constexpr unsigned GenerationRetries = RetrySlots * UntrappedRetries;

    enum SlotState : unsigned { Free, Claimed, Queued, Running, Done };
    enum class Owner { Any, Self, Others };

    struct Slot {
        std::atomic<unsigned> state{Free};
        std::uintptr_t page = 0;
        semaphore_t done = MACH_PORT_NULL;
    };

    struct Resolving {
        std::uintptr_t begin;
        std::uintptr_t end;
        pthread_t owner;
    };

    struct Retry {
        std::uintptr_t page = 0;
        pthread_t thread = nullptr;
        std::uint64_t generation = 0;
        std::uint64_t used = 0;
        unsigned count = 0;
    };

    struct Range {
        std::uintptr_t end;
        std::vector<std::uint8_t> pages;
    };

    struct Mutex {
        Mutex() {
            pthread_mutexattr_t attributes;
            pthread_mutexattr_init(&attributes);
            pthread_mutexattr_setpolicy_np(&attributes, PTHREAD_MUTEX_POLICY_FIRSTFIT_NP);
            pthread_mutex_init(&mutex, &attributes);
            pthread_mutexattr_destroy(&attributes);
            pthread_cond_init(&idle, nullptr);
        }
        pthread_mutex_t mutex;
        pthread_cond_t idle;
        std::atomic<unsigned> faulting{0};
    };

    struct Lock {
        explicit Lock(Mutex& mutex) : _mutex(mutex) {
            pthread_mutex_lock(&_mutex.mutex);
            while (_mutex.faulting.load(std::memory_order_acquire) != 0) pthread_cond_wait(&_mutex.idle, &_mutex.mutex);
        }
        ~Lock() {
            pthread_mutex_unlock(&_mutex.mutex);
        }
        Mutex& _mutex;
    };

    struct FaultLock {
        explicit FaultLock(Mutex& mutex) : _mutex(mutex) {
            _mutex.faulting.fetch_add(1, std::memory_order_acq_rel);
            pthread_mutex_lock(&_mutex.mutex);
        }
        ~FaultLock() {
            if (_mutex.faulting.fetch_sub(1, std::memory_order_acq_rel) == 1) pthread_cond_broadcast(&_mutex.idle);
            pthread_mutex_unlock(&_mutex.mutex);
        }
        Mutex& _mutex;
    };

    Watch() {
        if (const char* ahead = std::getenv("APS5_WRITE_WATCH_FAULT_AHEAD")) _faultAhead = static_cast<std::size_t>(std::strtoul(ahead, nullptr, 0));
        if (std::getenv("APS5_NO_WRITE_WATCH") != nullptr) return;
        struct sigaction action {};
        action.sa_sigaction = &Watch::fault;
        action.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGBUS, &action, &_previousBus) != 0 || sigaction(SIGSEGV, &action, &_previousSegv) != 0) {
            std::fprintf(stderr, "[memory] write watch: cannot install the fault handler (%s); guest buffers are fully re-read every use\n", std::strerror(errno));
            return;
        }
        _available = true;
    }

    static bool isWriteFault(const siginfo_t* info, void* context) {
        if (info->si_code <= 0 || info->si_code >= SI_USER || context == nullptr) return false;
        const auto state = static_cast<ucontext_t*>(context)->uc_mcontext;
        if (state == nullptr) return false;
#if defined(__x86_64__)
        constexpr unsigned PageFault = 14;
        constexpr unsigned WriteAccess = 1u << 1;
        return state->__es.__trapno == PageFault && (state->__es.__err & WriteAccess) != 0;
#elif defined(__aarch64__)
        constexpr unsigned DataAbort = 0x24;
        constexpr unsigned ExceptionClassShift = 26;
        constexpr unsigned WriteAccess = 1u << 6;
        return (state->__es.__esr >> ExceptionClassShift) == DataAbort && (state->__es.__esr & WriteAccess) != 0;
#else
        return false;
#endif
    }

    static void fault(int signal, siginfo_t* info, void* context) {
        auto& watch = Get();
        const bool write = isWriteFault(info, context);
        if (info->si_code > 0 && info->si_code < SI_USER && watch.resolveTrap(reinterpret_cast<std::uintptr_t>(info->si_addr), write)) return;
        if (write && watch.record(reinterpret_cast<std::uintptr_t>(info->si_addr))) return;
        const auto& previous = signal == SIGBUS ? watch._previousBus : watch._previousSegv;
        if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
            previous.sa_sigaction(signal, info, context);
            return;
        }
        if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN && previous.sa_handler != nullptr) {
            previous.sa_handler(signal);
            return;
        }
        ::signal(signal, SIG_DFL);
        if (previous.sa_handler == SIG_DFL) ::raise(signal);
    }

    bool record(std::uintptr_t address) {
        const auto page = pageStart(address);
        const FaultLock lock(_lock);
        auto it = _ranges.upper_bound(page);
        if (it == _ranges.begin()) return false;
        --it;
        if (page >= it->second.end) return false;
        auto& pages = it->second.pages;
        const auto index = static_cast<std::size_t>((page - it->first) / _page);
        auto& state = pages[index];
        if ((state & Protected) != 0) {
            const auto protection = state & ProtectionMask;
            const auto writtenOpen = [&](std::size_t at) { return (pages[at] & (Written | Protected)) == Written && (pages[at] & ProtectionMask) == protection; };
            std::size_t count = 1;
            if (_faultAhead > 1 && index >= 2 && writtenOpen(index - 1) && writtenOpen(index - 2)) {
                const auto blockPages = static_cast<std::size_t>(((page | (FaultAheadBlock - 1)) + 1 - page) / _page);
                const auto limit = std::min({_faultAhead, blockPages, pages.size() - index});
                while (count < limit && (pages[index + count] & (Protected | ReadTrapped)) == Protected && (pages[index + count] & ProtectionMask) == protection) ++count;
            }
            if (::mprotect(reinterpret_cast<void*>(page), count * _page, protection) != 0) return false;
            for (std::size_t at = index; at < index + count; ++at) pages[at] = static_cast<std::uint8_t>((pages[at] & (ProtectionMask | Fresh)) | Written);
            return true;
        }
        if ((state & PROT_WRITE) == 0) return false;
        std::uint8_t value = 0;
        mach_vm_size_t copied = 0;
        return mach_vm_read_overwrite(mach_task_self(), address, sizeof(value), reinterpret_cast<mach_vm_address_t>(&value), &copied) == KERN_SUCCESS && copied == sizeof(value);
    }

    bool resolveTrap(std::uintptr_t address, bool write) {
        if (!_trapping.load(std::memory_order_acquire)) return false;
        const auto page = pageStart(address);
        {
            const FaultLock lock(_lock);
            auto* state = pageState(page);
            if (state == nullptr) return false;
            if ((*state & ReadTrapped) == 0) return retryUntrapped(page, *state, write);
            const bool own = resolvingCovers(page, page + _page, Owner::Self) || pthread_equal(pthread_self(), _resolverThread);
            const auto bypass = _bypass.load(std::memory_order_acquire);
            if (own || (bypass != nullptr && bypass()) || (!trapCovers(page, page + _page) && !resolvingCovers(page, page + _page, Owner::Any))) {
                if (own) deferCovering(page);
                *state = static_cast<std::uint8_t>(*state & ~ReadTrapped);
                ++_untrapGeneration;
                return ::mprotect(reinterpret_cast<void*>(page), _page, untrappedProtection(*state)) == 0;
            }
        }
        Slot* slot = nullptr;
        while (slot == nullptr) {
            for (auto& candidate : _slots) {
                unsigned expected = Free;
                if (candidate.state.compare_exchange_strong(expected, Claimed, std::memory_order_acq_rel)) {
                    slot = &candidate;
                    break;
                }
            }
            if (slot == nullptr) sched_yield();
        }
        slot->page = page;
        slot->state.store(Queued, std::memory_order_release);
        semaphore_signal(_wake);
        while (semaphore_wait(slot->done) == KERN_ABORTED) {}
        slot->state.store(Free, std::memory_order_release);
        return true;
    }

    void startResolver() {
        _resolving.reserve(ResolvingCapacity);
        _deferred.reserve(ResolvingCapacity);
        if (semaphore_create(mach_task_self(), &_wake, SYNC_POLICY_FIFO, 0) != KERN_SUCCESS) return;
        for (auto& slot : _slots)
            if (semaphore_create(mach_task_self(), &slot.done, SYNC_POLICY_FIFO, 0) != KERN_SUCCESS) return;
        pthread_attr_t attributes;
        pthread_attr_init(&attributes);
        pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
        const bool created = pthread_create(&_resolverThread, &attributes, [](void* watch) -> void* {
            static_cast<Watch*>(watch)->resolveLoop();
            return nullptr;
        }, this) == 0;
        pthread_attr_destroy(&attributes);
        _resolverRunning.store(created, std::memory_order_release);
    }

    void resolveLoop() {
        pthread_setname_np("write watch read traps");
        for (;;) {
            if (semaphore_wait(_wake) != KERN_SUCCESS) continue;
            for (auto& slot : _slots) {
                unsigned expected = Queued;
                if (!slot.state.compare_exchange_strong(expected, Running, std::memory_order_acq_rel)) continue;
                const auto page = slot.page;
                waitForResolving(page, page + _page);
                resolve(page, page + _page);
                {
                    const Lock lock(_lock);
                    untrap(page, page + _page);
                }
                slot.state.store(Done, std::memory_order_release);
                semaphore_signal(slot.done);
            }
        }
    }

    void resolve(std::uintptr_t begin, std::uintptr_t end) {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> taken;
        const auto self = pthread_self();
        {
            const Lock lock(_lock);
            taken = takeTraps(begin, end);
            for (const auto& [first, last] : taken) _resolving.push_back({first, last, self});
        }
        if (taken.empty()) return;
        const auto resolver = _resolver.load(std::memory_order_acquire);
        for (const auto& [first, last] : taken) resolver(first, last);
        for (;;) {
            std::pair<std::uintptr_t, std::uintptr_t> deferred;
            {
                const Lock lock(_lock);
                const auto found = std::find_if(_deferred.begin(), _deferred.end(), [&](const Resolving& entry) { return pthread_equal(entry.owner, self); });
                if (found == _deferred.end()) break;
                deferred = {found->begin, found->end};
                _deferred.erase(found);
                auto [first, last] = _traps.equal_range(deferred.first);
                while (first != last && first->second != deferred.second) ++first;
                if (first == last) {
                    const auto entry = std::find_if(_resolving.begin(), _resolving.end(), [&](const Resolving& resolving) { return resolving.begin == deferred.first && resolving.end == deferred.second && pthread_equal(resolving.owner, self); });
                    if (entry != _resolving.end()) _resolving.erase(entry);
                    continue;
                }
                _traps.erase(first);
            }
            taken.push_back(deferred);
            resolver(deferred.first, deferred.second);
        }
        const Lock lock(_lock);
        for (const auto& [first, last] : taken) {
            const auto found = std::find_if(_resolving.begin(), _resolving.end(), [&](const Resolving& entry) { return entry.begin == first && entry.end == last && pthread_equal(entry.owner, self); });
            if (found != _resolving.end()) _resolving.erase(found);
        }
        for (const auto& [first, last] : taken) untrap(first, last);
    }

    void deferCovering(std::uintptr_t page) {
        const auto self = pthread_self();
        for (auto it = _traps.begin(); it != _traps.end() && it->first < page + _page; ++it) {
            if (it->second <= page || _deferred.size() == _deferred.capacity() || _resolving.size() == _resolving.capacity()) continue;
            if (std::any_of(_deferred.begin(), _deferred.end(), [&](const Resolving& entry) { return entry.begin == it->first && entry.end == it->second; })) continue;
            _resolving.push_back({it->first, it->second, self});
            _deferred.push_back({it->first, it->second, self});
        }
    }

    void waitForResolving(std::uintptr_t begin, std::uintptr_t end) {
        for (;;) {
            {
                const Lock lock(_lock);
                if (!resolvingCovers(begin, end, Owner::Others)) return;
            }
            sched_yield();
        }
    }

    bool resolvingCovers(std::uintptr_t begin, std::uintptr_t end, Owner owner) const {
        const auto self = pthread_self();
        return std::any_of(_resolving.begin(), _resolving.end(), [&](const Resolving& entry) {
            if (pageStart(entry.begin) >= end || begin >= pageEnd(entry.end)) return false;
            return owner == Owner::Any || (pthread_equal(entry.owner, self) != 0) == (owner == Owner::Self);
        });
    }

    bool retryUntrapped(std::uintptr_t page, std::uint8_t state, bool write) {
        if ((untrappedProtection(state) & (write ? PROT_WRITE : PROT_READ)) == 0) return false;
        if (_retryGeneration != _untrapGeneration) {
            _retryGeneration = _untrapGeneration;
            _generationRetries = 0;
        }
        if (++_generationRetries > GenerationRetries) return false;
        const auto thread = pthread_self();
        auto found = std::find_if(_retries.begin(), _retries.end(), [&](const Retry& retry) { return retry.used != 0 && retry.page == page && pthread_equal(retry.thread, thread); });
        if (found == _retries.end()) found = std::min_element(_retries.begin(), _retries.end(), [](const Retry& left, const Retry& right) { return left.used < right.used; });
        if (found->used == 0 || found->page != page || !pthread_equal(found->thread, thread) || found->generation != _untrapGeneration) *found = {page, thread, _untrapGeneration, 0, 0};
        found->used = ++_retryUses;
        return ++found->count <= UntrappedRetries;
    }

    template <typename TFunction>
    std::invoke_result_t<TFunction> withoutTraps(std::uintptr_t begin, std::uintptr_t end, bool restore, TFunction function) {
        for (;;) {
            if (_trapping.load(std::memory_order_acquire)) {
                waitForResolving(begin, end);
                resolve(begin, end);
            }
            const Lock lock(_lock);
            if (trapCovers(pageStart(begin), pageEnd(end)) || resolvingCovers(pageStart(begin), pageEnd(end), Owner::Others)) continue;
            if (restore) untrap(begin, end);
            else forEachPage(pageStart(begin), pageEnd(end), [](std::uintptr_t, std::uint8_t& state) { state = static_cast<std::uint8_t>(state & ~ReadTrapped); });
            return function();
        }
    }

    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> takeTraps(std::uintptr_t begin, std::uintptr_t end) {
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> taken;
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        for (auto it = _traps.begin(); it != _traps.end();) {
            if (it->first >= last || it->second <= first) {
                ++it;
                continue;
            }
            taken.emplace_back(*it);
            it = _traps.erase(it);
        }
        return taken;
    }

    void dropTraps(std::uintptr_t begin, std::uintptr_t end) {
        for (const auto& [first, last] : takeTraps(begin, end)) untrap(first, last);
        untrap(begin, end);
    }

    bool trapCovers(std::uintptr_t begin, std::uintptr_t end) const {
        return std::any_of(_traps.begin(), _traps.end(), [&](const auto& trap) { return trap.first < end && begin < trap.second; });
    }

    void untrap(std::uintptr_t begin, std::uintptr_t end) {
        const auto first = pageStart(begin);
        const auto last = pageEnd(end);
        std::vector<std::pair<std::uintptr_t, std::uintptr_t>> remaining;
        for (const auto& trap : _traps)
            if (trap.first < last && first < trap.second) remaining.push_back(trap);
        forEachPage(first, last, [&](std::uintptr_t page, std::uint8_t& state) {
            if ((state & ReadTrapped) == 0) return;
            if (std::any_of(remaining.begin(), remaining.end(), [&](const auto& trap) { return trap.first < page + _page && page < trap.second; })) return;
            if (resolvingCovers(page, page + _page, Owner::Others)) return;
            state = static_cast<std::uint8_t>(state & ~ReadTrapped);
            ::mprotect(reinterpret_cast<void*>(page), _page, untrappedProtection(state));
            ++_untrapGeneration;
        });
    }

    static int untrappedProtection(std::uint8_t state) {
        return (state & Protected) != 0 ? (state & ProtectionMask & ~PROT_WRITE) : (state & ProtectionMask);
    }

    std::uint8_t* pageState(std::uintptr_t page) {
        auto it = _ranges.upper_bound(page);
        if (it == _ranges.begin()) return nullptr;
        --it;
        if (page >= it->second.end) return nullptr;
        return &it->second.pages[(page - it->first) / _page];
    }

    std::uintptr_t pageStart(std::uintptr_t address) const {
        return address & ~(_page - 1);
    }

    std::uintptr_t pageEnd(std::uintptr_t address) const {
        return (address + _page - 1) & ~(_page - 1);
    }

    template <typename TFunction>
    void forEachPage(std::uintptr_t begin, std::uintptr_t end, TFunction function) {
        auto it = _ranges.upper_bound(begin);
        if (it != _ranges.begin()) --it;
        for (; it != _ranges.end() && it->first < end; ++it) {
            const auto first = std::max(begin, it->first);
            const auto last = std::min(end, it->second.end);
            for (auto page = first; page < last; page += _page) function(page, it->second.pages[(page - it->first) / _page]);
        }
    }

    void protect(std::uintptr_t begin, std::uintptr_t end) {
        std::uintptr_t runStart = 0;
        std::uintptr_t previous = begin;
        int runProtection = -1;
        const auto flush = [&](std::uintptr_t runEnd) {
            if (runProtection >= 0) ::mprotect(reinterpret_cast<void*>(runStart), runEnd - runStart, runProtection);
            runProtection = -1;
        };
        forEachPage(begin, end, [&](std::uintptr_t page, std::uint8_t& state) {
            const int wanted = (state & ReadTrapped) != 0 ? PROT_NONE : (state & Protected) != 0 ? (state & ProtectionMask & ~PROT_WRITE) : -1;
            if (wanted != runProtection || page != previous) {
                flush(previous);
                runStart = page;
                runProtection = wanted;
            }
            previous = page + _page;
        });
        flush(previous);
    }

    bool covers(std::uintptr_t begin, std::uintptr_t end) const {
        auto it = _ranges.upper_bound(begin);
        if (it == _ranges.begin()) return false;
        --it;
        auto cursor = begin;
        for (; it != _ranges.end() && it->first <= cursor; ++it) {
            if (it->second.end > cursor) cursor = it->second.end;
            if (cursor >= end) return true;
        }
        return false;
    }

    bool erase(std::uintptr_t begin, std::uintptr_t end) {
        bool removed = false;
        auto it = _ranges.upper_bound(begin);
        if (it != _ranges.begin()) --it;
        while (it != _ranges.end() && it->first < end) {
            const auto rangeBegin = it->first;
            const auto rangeEnd = it->second.end;
            if (rangeEnd <= begin) {
                ++it;
                continue;
            }
            removed = true;
            auto pages = std::move(it->second.pages);
            it = _ranges.erase(it);
            const auto cutBegin = std::max(begin, rangeBegin);
            const auto cutEnd = std::min(end, rangeEnd);
            for (auto page = cutBegin; page < cutEnd; page += _page) {
                const auto state = pages[(page - rangeBegin) / _page];
                if ((state & (Protected | ReadTrapped)) != 0) ::mprotect(reinterpret_cast<void*>(page), _page, state & ProtectionMask);
            }
            if (rangeBegin < cutBegin) _ranges.emplace(rangeBegin, Range{cutBegin, std::vector<std::uint8_t>(pages.begin(), pages.begin() + (cutBegin - rangeBegin) / _page)});
            if (cutEnd < rangeEnd) it = _ranges.emplace(cutEnd, Range{rangeEnd, std::vector<std::uint8_t>(pages.begin() + (cutEnd - rangeBegin) / _page, pages.end())}).first;
        }
        return removed;
    }

    const std::uintptr_t _page = static_cast<std::uintptr_t>(::getpagesize());
    std::size_t _faultAhead = 32;
    bool _available = false;
    Mutex _lock;
    std::map<std::uintptr_t, Range> _ranges;
    std::vector<std::pair<std::uintptr_t, std::uintptr_t>> _hostWrites;
    std::multimap<std::uintptr_t, std::uintptr_t> _traps;
    std::vector<Resolving> _resolving;
    std::vector<Resolving> _deferred;
    std::array<Retry, RetrySlots> _retries{};
    std::uint64_t _retryUses = 0;
    std::uint64_t _retryGeneration = 0;
    unsigned _generationRetries = 0;
    std::uint64_t _untrapGeneration = 0;
    std::atomic<bool> _trapping{false};
    std::atomic<void (*)(std::uintptr_t, std::uintptr_t)> _resolver{nullptr};
    std::atomic<bool (*)()> _bypass{nullptr};
    std::once_flag _resolverStarted;
    std::atomic<bool> _resolverRunning{false};
    pthread_t _resolverThread{};
    semaphore_t _wake = MACH_PORT_NULL;
    std::array<Slot, ResolveSlots> _slots;
    struct sigaction _previousBus {};
    struct sigaction _previousSegv {};
};

const bool g_opened = (Watch::Get(), true);
#endif

#if defined(__linux__) && defined(ANYPS5_KYTY)
std::atomic<GpuFaultHandler> gpuFaultHandler{nullptr};

bool ResolveGpuFault(int signal, siginfo_t* info, void* context) {
    if ((signal != SIGSEGV && signal != SIGBUS) || info == nullptr || info->si_code <= 0 || context == nullptr) return false;
    const auto handler = gpuFaultHandler.load(std::memory_order_acquire);
    if (handler == nullptr) return false;
    sigset_t asynchronous;
    sigfillset(&asynchronous);
    for (const int synchronous : {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGTRAP, SIGABRT}) sigdelset(&asynchronous, synchronous);
    sigset_t interrupted;
    pthread_sigmask(SIG_BLOCK, &asynchronous, &interrupted);
    const int savedErrno = errno;
    const bool handled = handler(signal, info, context);
    errno = savedErrno;
    if (!handled) pthread_sigmask(SIG_SETMASK, &interrupted, nullptr);
    return handled;
}
#endif

}

bool GuestWriteWatchAvailable_nid_postfix() {
#ifdef APS5_GUEST_WRITE_WATCH
    return Watch::Get().Available();
#else
    return false;
#endif
}

const char* GuestWriteWatchUnavailableReason_nid_postfix() {
#if defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC)
    return Watch::Get().Available() ? Watch::Get().RegistrationFailure() : g_unavailableReason;
#elif defined(__linux__)
    return "this build has no PAGEMAP_SCAN or UFFD_FEATURE_WP_ASYNC: its kernel headers are older than Linux 6.7";
#else
    return "";
#endif
}

void GuestWriteWatchRegister_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(__linux__) && !defined(APS5_LINUX_READ_TRAPS)
    if (!GpuOwnsFaults && bytes != 0) ReadTrapTable::Get().Register(reinterpret_cast<std::uintptr_t>(pointer), reinterpret_cast<std::uintptr_t>(pointer) + bytes);
#endif
#ifdef APS5_GUEST_WRITE_WATCH
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    Watch::Get().Register(begin, begin + bytes);
#if defined(APS5_LINUX_READ_TRAPS)
    if (Watch::Get().Covers(begin, begin + bytes)) ReadTraps::Get().Registered(begin, begin + bytes);
#endif
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
#endif
}

bool GuestWriteWatchUnregister_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(__linux__) && !defined(APS5_LINUX_READ_TRAPS)
    if (bytes != 0) ReadTrapTable::Get().Unregister(reinterpret_cast<std::uintptr_t>(pointer), reinterpret_cast<std::uintptr_t>(pointer) + bytes);
#endif
#ifdef APS5_GUEST_WRITE_WATCH
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
#if defined(APS5_LINUX_READ_TRAPS)
    ReadTraps::Get().Forget(begin, begin + bytes);
#endif
    return Watch::Get().Unregister(begin, begin + bytes);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
    return false;
#endif
}

bool GuestWriteWatchAlias_nid_postfix(const void* first, const void* second, std::size_t bytes) {
#if defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC)
    return Watch::Get().Alias(reinterpret_cast<std::uintptr_t>(first), reinterpret_cast<std::uintptr_t>(second), bytes);
#else
    static_cast<void>(first);
    static_cast<void>(second);
    static_cast<void>(bytes);
    return false;
#endif
}

#if defined(__linux__)
void GuestWriteWatchNoteWritten_nid_postfix(const void* pointer, std::size_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    Watch::Get().Note(begin, begin + bytes);
}
#endif

bool GuestWriteWatchCovers_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#ifdef APS5_GUEST_WRITE_WATCH
    return bytes != 0 && address + bytes > address && Watch::Get().Covers(address, address + bytes);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    return false;
#endif
}

bool GuestWriteWatchOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#ifdef APS5_GUEST_WRITE_WATCH
    return bytes != 0 && address + bytes > address && Watch::Get().Overlaps(address, address + bytes);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    return false;
#endif
}

bool GuestWriteWatchHandleFault_nid_postfix(std::uintptr_t address) {
#if defined(__APPLE__)
    return Watch::Get().HandleFault(address);
#elif defined(APS5_LINUX_READ_TRAPS)
    return ReadTraps::Get().HandleFault(address);
#else
    static_cast<void>(address);
    return false;
#endif
}

bool GuestWriteWatchCollect_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context) {
#ifdef APS5_GUEST_WRITE_WATCH
    if (bytes == 0 || address + bytes < address) return false;
    return Watch::Get().Collect(address, address + bytes, written, context);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    static_cast<void>(written);
    static_cast<void>(context);
    return false;
#endif
}

void GuestWriteWatchBeginHostWrite_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(__APPLE__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    Watch::Get().BeginHostWrite(begin, begin + bytes);
#elif defined(APS5_LINUX_READ_TRAPS)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    ReadTraps::Get().BeginHostWrite(begin, begin + bytes);
#elif defined(__linux__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes != 0 && begin + bytes > begin) ReadTrapTable::Get().BeginHostWrite(begin, begin + bytes);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
#endif
}

void GuestWriteWatchEndHostWrite_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(__APPLE__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    Watch::Get().EndHostWrite(begin, begin + bytes);
#elif defined(APS5_LINUX_READ_TRAPS)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    ReadTraps::Get().EndHostWrite(begin, begin + bytes);
#elif defined(__linux__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes != 0 && begin + bytes > begin) ReadTrapTable::Get().EndHostWrite(begin, begin + bytes);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
#endif
}

void GuestWriteWatchProtectionChanged_nid_postfix(const void* pointer, std::size_t bytes, int protection) {
#if defined(__APPLE__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    Watch::Get().ProtectionChanged(begin, begin + bytes, protection);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
    static_cast<void>(protection);
#endif
}

int GuestWriteWatchProtect_nid_postfix(void* pointer, std::size_t bytes, int protection) {
#if defined(__APPLE__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return Watch::Get().Protect(begin, begin + bytes, protection);
#elif defined(__linux__)
#if defined(APS5_LINUX_READ_TRAPS)
    if (Watch::Get().Available()) {
        const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
        return ReadTraps::Get().Protect(begin, begin + bytes, protection);
    }
#endif
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return ReadTrapTable::Get().Protect(begin, begin + bytes, protection);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
    static_cast<void>(protection);
    return ENOSYS;
#endif
}

int GuestWriteWatchProtection_nid_postfix(std::uintptr_t address, std::uintptr_t limit, std::uintptr_t* runEnd) {
#if defined(__APPLE__)
    return Watch::Get().Protection(address, limit, runEnd);
#elif defined(__linux__)
    return ReadTrapTable::Get().Protection(address, limit, runEnd);
#else
    static_cast<void>(address);
    *runEnd = limit;
    return -1;
#endif
}

bool GuestWriteWatchCollectFresh_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context) {
#if defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC)
    if (bytes == 0 || address + bytes < address) return false;
    return Watch::Get().CollectFresh(address, address + bytes, written, context);
#elif defined(__APPLE__)
    if (bytes == 0 || address + bytes < address) return false;
    return Watch::Get().Collect(address, address + bytes, written, context, Watch::Take::Fresh);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    static_cast<void>(written);
    static_cast<void>(context);
    return false;
#endif
}

bool GuestWriteWatchCollectArmed_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context) {
#if defined(__linux__) && defined(PAGEMAP_SCAN) && defined(UFFD_FEATURE_WP_ASYNC)
    if (bytes == 0 || address + bytes < address) return false;
    return Watch::Get().CollectArmed(address, address + bytes, written, context);
#elif defined(__APPLE__)
    if (bytes == 0 || address + bytes < address) return false;
    return Watch::Get().Collect(address, address + bytes, written, context, Watch::Take::Armed);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    static_cast<void>(written);
    static_cast<void>(context);
    return false;
#endif
}

bool GuestWriteWatchSetReadTrapResolver_nid_postfix(void (*resolve)(std::uintptr_t begin, std::uintptr_t end)) {
#if defined(__APPLE__)
    return Watch::Get().SetReadTrapResolver(resolve);
#elif defined(APS5_LINUX_READ_TRAPS)
    return ReadTraps::Get().SetResolver(resolve);
#elif defined(__linux__) && defined(__x86_64__)
    if (resolve != nullptr && (GpuOwnsFaults || !GuardedCopy::FaultsCaught())) return false;
    return ReadTrapTable::Get().SetResolver(resolve);
#else
    static_cast<void>(resolve);
    return false;
#endif
}

void GuestWriteWatchSetReadTrapBypass_nid_postfix(bool (*bypass)()) {
#if defined(__APPLE__)
    Watch::Get().SetReadTrapBypass(bypass);
#elif defined(APS5_LINUX_READ_TRAPS)
    ReadTraps::Get().SetBypass(bypass);
#elif defined(__linux__)
    ReadTrapTable::Get().SetBypass(bypass);
#else
    static_cast<void>(bypass);
#endif
}

bool GuestWriteWatchTrapReads_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#if defined(__APPLE__)
    return bytes != 0 && address + bytes > address && Watch::Get().TrapReads(address, address + bytes);
#elif defined(APS5_LINUX_READ_TRAPS)
    return bytes != 0 && address + bytes > address && ReadTraps::Get().TrapReads(address, address + bytes);
#elif defined(__linux__)
    return bytes != 0 && address + bytes > address && ReadTrapTable::Get().Trap(address, address + bytes, false);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    return false;
#endif
}

void GuestWriteWatchReadTraps_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*trapped)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context) {
#if defined(__APPLE__)
    if (bytes != 0 && address + bytes > address) Watch::Get().ReadTraps(address, address + bytes, trapped, context);
#elif defined(APS5_LINUX_READ_TRAPS)
    if (bytes != 0 && address + bytes > address) ReadTraps::Get().Trapped(address, address + bytes, trapped, context);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    static_cast<void>(trapped);
    static_cast<void>(context);
#endif
}

void GuestWriteWatchReleaseReads_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#if defined(__APPLE__)
    if (bytes != 0 && address + bytes > address) Watch::Get().ReleaseReads(address, address + bytes);
#elif defined(APS5_LINUX_READ_TRAPS)
    if (bytes != 0 && address + bytes > address) ReadTraps::Get().ReleaseReads(address, address + bytes);
#elif defined(__linux__)
    if (bytes != 0 && address + bytes > address) ReadTrapTable::Get().Release(address, address + bytes, false);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
#endif
}

bool GuestWriteWatchTrapWrites_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#if defined(__linux__)
    return bytes != 0 && address + bytes > address && ReadTrapTable::Get().Trap(address, address + bytes, true);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    return false;
#endif
}

void GuestWriteWatchReleaseWrites_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#if defined(__linux__)
    if (bytes != 0 && address + bytes > address) ReadTrapTable::Get().Release(address, address + bytes, true);
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
#endif
}

void GuestWriteWatchSweepReadTraps_nid_postfix() {
#if defined(__linux__)
    ReadTrapTable::Get().Sweep();
#endif
}

void GuestWriteWatchSettleReads_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(__linux__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes != 0 && begin + bytes > begin) ReadTrapTable::Get().Settle(begin, begin + bytes);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
#endif
}

void GuestWriteWatchOpenReads_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(__linux__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes != 0 && begin + bytes > begin) ReadTrapTable::Get().Open(begin, begin + bytes);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
#endif
}

int GuestWriteWatchOpenIdleReads_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(__linux__)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes != 0 && begin + bytes > begin ? ReadTrapTable::Get().OpenIdle(begin, begin + bytes) : 0;
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
    return 0;
#endif
}

bool GuestWriteWatchReadTrapsHeld_nid_postfix() {
#if defined(__linux__)
    return ReadTrapTable::Get().Active();
#else
    return false;
#endif
}

std::uint64_t GuestWriteWatchReadTrapEpoch_nid_postfix() {
#if defined(__linux__)
    return ReadTrapTable::Get().Epoch();
#else
    return 0;
#endif
}

void GuestWriteWatchReadTrapCounts_nid_postfix(ReadTrapCounts* counts) {
    if (counts == nullptr) return;
    *counts = {};
#if defined(__linux__)
    ReadTrapTable::Get().Counts(*counts);
#endif
}

std::size_t GuestWriteWatchReadTrapFaults_nid_postfix(ReadTrapFault* faults, std::size_t capacity) {
#if defined(__linux__)
    return faults != nullptr ? ReadTrapTable::Get().Faults(faults, capacity) : 0;
#else
    static_cast<void>(faults);
    static_cast<void>(capacity);
    return 0;
#endif
}

#if defined(__linux__)
bool GuestWriteWatchHandleReadTrap_nid_postfix(int signal, siginfo_t* info, void* context) {
#if defined(ANYPS5_KYTY)
    return ResolveGpuFault(signal, info, context);
#else
    return ReadTrapTable::Get().HandleFault(signal, info, context);
#endif
}
#endif

#if defined(__linux__) && defined(ANYPS5_KYTY)
void GuestWriteWatchSetGpuFaultHandler_nid_no_patch(GpuFaultHandler handler) {
    gpuFaultHandler.store(handler, std::memory_order_release);
}
#endif
std::uint64_t GuestWriteWatchReadTrapGeneration_nid_postfix() {
#if defined(APS5_LINUX_READ_TRAPS)
    return ReadTraps::Get().Generation();
#else
    static std::atomic<std::uint64_t> asked{1};
    return asked.fetch_add(1, std::memory_order_relaxed);
#endif
}

bool GuestWriteWatchBeginUntrappedAccess_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(APS5_LINUX_READ_TRAPS)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    return bytes == 0 || begin + bytes < begin || ReadTraps::Get().BeginUntrapped(begin, begin + bytes);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
    return true;
#endif
}

void GuestWriteWatchEndUntrappedAccess_nid_postfix(const void* pointer, std::size_t bytes) {
#if defined(APS5_LINUX_READ_TRAPS)
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes != 0 && begin + bytes > begin) ReadTraps::Get().EndHostWrite(begin, begin + bytes);
#else
    static_cast<void>(pointer);
    static_cast<void>(bytes);
#endif
}

bool GuestWriteWatchAwaitsReadTrap_nid_postfix(std::uint64_t thread) {
#if defined(APS5_LINUX_READ_TRAPS)
    return ReadTraps::Get().Awaits(static_cast<pid_t>(thread));
#else
    static_cast<void>(thread);
    return false;
#endif
}

}
