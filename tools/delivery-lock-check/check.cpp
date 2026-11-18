#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#include <unwind.h>

namespace {

enum class Kind : std::uint8_t { Mutex, Read, Write, Spin };

struct Held {
    const void* lock;
    const void* site;
    Kind kind;
};

constexpr int MaxHeld = 32;
constexpr std::size_t MaxSites = 1024;
constexpr std::uint64_t ParkedNanos = 200000;

constexpr int StackFrames = 16;
constexpr std::size_t MaxStacks = 4096;

struct ThreadLocks {
    int depth;
    int overflow;
    bool waiting;
    int pending;
    int pendingStack;
    std::array<Held, MaxHeld> held;
    std::array<Held, MaxHeld> parked;
};

struct Stack {
    std::atomic<std::uint64_t> hash;
    std::atomic<int> frames;
    std::array<std::uintptr_t, StackFrames> ips;
    std::atomic<std::uint64_t> arrivals;
    std::atomic<std::uint64_t> ran;
    std::atomic<std::uint64_t> deferred;
};

std::array<Stack, MaxStacks> g_stacks {};
std::atomic<std::uint64_t> g_guest {0};
std::atomic<std::uint64_t> g_host {0};

[[gnu::tls_model("initial-exec")]] constinit thread_local ThreadLocks t_locks {};

struct Site {
    std::atomic<const void*> site;
    std::atomic<const void*> lock;
    std::atomic<std::uint8_t> kind;
    std::atomic<std::uint64_t> ran;
    std::atomic<std::uint64_t> deferred;
};

std::array<Site, MaxSites> g_sites {};
std::atomic<std::uint64_t> g_arrivals {0};
std::atomic<std::uint64_t> g_ran {0};
std::atomic<std::uint64_t> g_deferred {0};
std::atomic<std::uint64_t> g_clean {0};
std::atomic<std::uint64_t> g_lost {0};
std::atomic<bool> g_maps_written {false};
std::atomic<void (*)(int, siginfo_t*, void*)> g_handler {nullptr};

class Line {
public:
    void Text(const char* text) {
        while (*text != '\0' && used < sizeof(buffer)) buffer[used++] = *text++;
    }
    void Hex(std::uintptr_t value) {
        Text("0x");
        char digits[16];
        int count = 0;
        do {
            digits[count++] = "0123456789abcdef"[value & 15u];
            value >>= 4u;
        } while (value != 0 && count < 16);
        while (count > 0 && used < sizeof(buffer)) buffer[used++] = digits[--count];
    }
    void Decimal(std::uint64_t value) {
        char digits[20];
        int count = 0;
        do {
            digits[count++] = static_cast<char>('0' + value % 10u);
            value /= 10u;
        } while (value != 0 && count < 20);
        while (count > 0 && used < sizeof(buffer)) buffer[used++] = digits[--count];
    }
    void Flush() {
        if (used < sizeof(buffer)) buffer[used++] = '\n';
        std::size_t done = 0;
        while (done < used) {
            const auto written = ::write(STDERR_FILENO, buffer + done, used - done);
            if (written <= 0) break;
            done += static_cast<std::size_t>(written);
        }
        used = 0;
    }

private:
    char buffer[1024];
    std::size_t used = 0;
};

const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Mutex: return "mutex";
    case Kind::Read: return "read";
    case Kind::Write: return "write";
    case Kind::Spin: return "spin";
    }
    return "?";
}

template <class Function>
Function Next(std::atomic<void*>& slot, const char* name) {
    void* function = slot.load(std::memory_order_acquire);
    if (function == nullptr) {
        function = dlsym(RTLD_NEXT, name);
        if (function == nullptr) {
            Line line;
            line.Text("[delivery-lock-check] no next definition of ");
            line.Text(name);
            line.Flush();
            std::abort();
        }
        slot.store(function, std::memory_order_release);
    }
    return reinterpret_cast<Function>(function);
}

#define NEXT(name) Next<decltype(&::name)>(next_##name, #name)
#define SLOT(name) std::atomic<void*> next_##name {nullptr};

SLOT(pthread_mutex_lock)
SLOT(pthread_mutex_trylock)
SLOT(pthread_mutex_timedlock)
SLOT(pthread_mutex_clocklock)
SLOT(pthread_mutex_unlock)
SLOT(pthread_rwlock_rdlock)
SLOT(pthread_rwlock_tryrdlock)
SLOT(pthread_rwlock_timedrdlock)
SLOT(pthread_rwlock_clockrdlock)
SLOT(pthread_rwlock_wrlock)
SLOT(pthread_rwlock_trywrlock)
SLOT(pthread_rwlock_timedwrlock)
SLOT(pthread_rwlock_clockwrlock)
SLOT(pthread_rwlock_unlock)
SLOT(pthread_spin_lock)
SLOT(pthread_spin_trylock)
SLOT(pthread_spin_unlock)
SLOT(pthread_cond_wait)
SLOT(pthread_cond_timedwait)
SLOT(pthread_cond_clockwait)
SLOT(pthread_kill)
SLOT(sigaction)

void Push(const void* lock, const void* site, Kind kind) {
    auto& locks = t_locks;
    if (locks.depth >= MaxHeld) {
        ++locks.overflow;
        g_lost.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    locks.held[static_cast<std::size_t>(locks.depth)] = {lock, site, kind};
    std::atomic_signal_fence(std::memory_order_seq_cst);
    ++locks.depth;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

bool Pop(const void* lock, Held* removed = nullptr) {
    auto& locks = t_locks;
    for (int index = locks.depth - 1; index >= 0; --index) {
        if (locks.held[static_cast<std::size_t>(index)].lock != lock) continue;
        if (removed != nullptr) *removed = locks.held[static_cast<std::size_t>(index)];
        const auto last = locks.depth - 1;
        locks.held[static_cast<std::size_t>(index)] = locks.held[static_cast<std::size_t>(last)];
        std::atomic_signal_fence(std::memory_order_seq_cst);
        locks.depth = last;
        std::atomic_signal_fence(std::memory_order_seq_cst);
        return true;
    }
    if (locks.overflow > 0) --locks.overflow;
    return false;
}

Site* SiteOf(const Held& held) {
    const auto hash = (reinterpret_cast<std::uintptr_t>(held.site) >> 2u) * 0x9e3779b97f4a7c15ull;
    for (std::size_t probe = 0; probe < MaxSites; ++probe) {
        auto& entry = g_sites[(hash + probe) % MaxSites];
        const void* current = entry.site.load(std::memory_order_acquire);
        if (current == held.site) return &entry;
        if (current == nullptr) {
            const void* expected = nullptr;
            if (entry.site.compare_exchange_strong(expected, held.site, std::memory_order_acq_rel) || expected == held.site) {
                entry.lock.store(held.lock, std::memory_order_relaxed);
                entry.kind.store(static_cast<std::uint8_t>(held.kind), std::memory_order_relaxed);
                return &entry;
            }
        }
    }
    return nullptr;
}

void WriteMaps() {
    if (g_maps_written.exchange(true)) return;
    const char* prefix = std::getenv("DELIVERY_LOCK_CHECK_OUT");
    if (prefix == nullptr) return;
    char path[512];
    const auto length = std::strlen(prefix);
    if (length + sizeof(".maps") > sizeof(path)) return;
    std::memcpy(path, prefix, length);
    std::memcpy(path + length, ".maps", sizeof(".maps"));
    const int source = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
    const int target = ::open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (source >= 0 && target >= 0) {
        char buffer[4096];
        for (;;) {
            const auto read = ::read(source, buffer, sizeof(buffer));
            if (read <= 0) break;
            if (::write(target, buffer, static_cast<std::size_t>(read)) != read) break;
        }
    }
    if (source >= 0) ::close(source);
    if (target >= 0) ::close(target);
}

void Report(const char* verdict, const std::array<Held, MaxHeld>& held, int count, std::uint64_t nanos) {
    bool fresh = false;
    for (int index = 0; index < count; ++index) {
        auto* site = SiteOf(held[static_cast<std::size_t>(index)]);
        if (site == nullptr) continue;
        auto& counter = verdict[0] == 'r' ? site->ran : site->deferred;
        if (counter.fetch_add(1, std::memory_order_relaxed) == 0) fresh = true;
    }
    (verdict[0] == 'r' ? g_ran : g_deferred).fetch_add(1, std::memory_order_relaxed);
    if (!fresh) return;
    Line line;
    line.Text("[delivery-lock-check] ");
    line.Text(verdict);
    line.Text(" tid=");
    line.Decimal(static_cast<std::uint64_t>(::syscall(SYS_gettid)));
    line.Text(" us=");
    line.Decimal(nanos / 1000u);
    for (int index = 0; index < count; ++index) {
        const auto& entry = held[static_cast<std::size_t>(index)];
        line.Text(" | ");
        line.Text(KindName(entry.kind));
        line.Text(" lock=");
        line.Hex(reinterpret_cast<std::uintptr_t>(entry.lock));
        line.Text(" site=");
        line.Hex(reinterpret_cast<std::uintptr_t>(entry.site));
    }
    line.Flush();
}

void Summary() {
    Line line;
    line.Text("[delivery-lock-check] arrivals=");
    line.Decimal(g_arrivals.load(std::memory_order_relaxed));
    line.Text(" clean=");
    line.Decimal(g_clean.load(std::memory_order_relaxed));
    line.Text(" ran-with-locks=");
    line.Decimal(g_ran.load(std::memory_order_relaxed));
    line.Text(" deferred-with-locks=");
    line.Decimal(g_deferred.load(std::memory_order_relaxed));
    line.Text(" untracked=");
    line.Decimal(g_lost.load(std::memory_order_relaxed));
    line.Text(" in-guest-code=");
    line.Decimal(g_guest.load(std::memory_order_relaxed));
    line.Text(" in-host-code=");
    line.Decimal(g_host.load(std::memory_order_relaxed));
    line.Flush();
    for (auto& entry : g_sites) {
        const void* site = entry.site.load(std::memory_order_acquire);
        if (site == nullptr) continue;
        line.Text("[delivery-lock-check] site=");
        line.Hex(reinterpret_cast<std::uintptr_t>(site));
        line.Text(" ");
        line.Text(KindName(static_cast<Kind>(entry.kind.load(std::memory_order_relaxed))));
        line.Text(" lock=");
        line.Hex(reinterpret_cast<std::uintptr_t>(entry.lock.load(std::memory_order_relaxed)));
        line.Text(" ran=");
        line.Decimal(entry.ran.load(std::memory_order_relaxed));
        line.Text(" deferred=");
        line.Decimal(entry.deferred.load(std::memory_order_relaxed));
        line.Flush();
    }
}

std::uint64_t Now() {
    timespec now {};
    ::clock_gettime(CLOCK_MONOTONIC, &now);
    return static_cast<std::uint64_t>(now.tv_sec) * 1000000000ull + static_cast<std::uint64_t>(now.tv_nsec);
}

bool GuestCode(std::uintptr_t ip) {
    dl_find_object object {};
    if (_dl_find_object(reinterpret_cast<void*>(ip), &object) != 0 || object.dlfo_link_map == nullptr) return true;
    const char* name = object.dlfo_link_map->l_name != nullptr ? object.dlfo_link_map->l_name : "";
    const auto length = std::strlen(name);
    constexpr char suffix[] = ".guest.prx";
    return length == 0 || (length >= sizeof(suffix) - 1 && std::strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0);
}

struct Walk {
    std::array<std::uintptr_t, 64> ips;
    int count;
};

_Unwind_Reason_Code Step(_Unwind_Context* context, void* argument) {
    auto& walk = *static_cast<Walk*>(argument);
    if (walk.count >= static_cast<int>(walk.ips.size())) return _URC_END_OF_STACK;
    walk.ips[static_cast<std::size_t>(walk.count++)] = _Unwind_GetIP(context);
    return _URC_NO_REASON;
}

int HostStack(std::uintptr_t ip) {
    Walk walk {};
    _Unwind_Backtrace(&Step, &walk);
    int first = 0;
    while (first < walk.count && walk.ips[static_cast<std::size_t>(first)] != ip) ++first;
    if (first == walk.count) return -1;
    const int frames = walk.count - first < StackFrames ? walk.count - first : StackFrames;
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (int index = 0; index < frames; ++index) hash = (hash ^ walk.ips[static_cast<std::size_t>(first + index)]) * 0x100000001b3ull;
    if (hash == 0) hash = 1;
    for (std::size_t probe = 0; probe < MaxStacks; ++probe) {
        const auto index = (hash + probe) % MaxStacks;
        auto& entry = g_stacks[index];
        std::uint64_t current = entry.hash.load(std::memory_order_acquire);
        if (current == 0) {
            if (!entry.hash.compare_exchange_strong(current, hash, std::memory_order_acq_rel) && current != hash) continue;
            if (current == 0) {
                for (int frame = 0; frame < frames; ++frame) entry.ips[static_cast<std::size_t>(frame)] = walk.ips[static_cast<std::size_t>(first + frame)];
                entry.frames.store(frames, std::memory_order_release);
                Line line;
                line.Text("[delivery-lock-check] stack id=");
                line.Decimal(index);
                line.Text(" frames=");
                for (int frame = 0; frame < frames; ++frame) {
                    line.Text(frame == 0 ? "" : ",");
                    line.Hex(walk.ips[static_cast<std::size_t>(first + frame)]);
                }
                line.Flush();
            }
        } else if (current != hash) {
            continue;
        }
        entry.arrivals.fetch_add(1, std::memory_order_relaxed);
        return static_cast<int>(index);
    }
    return -1;
}

void Outcome(bool ran, std::uint64_t nanos) {
    auto& locks = t_locks;
    if (locks.pending != 0) Report(ran ? "ran" : "deferred", locks.parked, locks.pending, nanos);
    if (locks.pendingStack >= 0) (ran ? g_stacks[static_cast<std::size_t>(locks.pendingStack)].ran : g_stacks[static_cast<std::size_t>(locks.pendingStack)].deferred).fetch_add(1, std::memory_order_relaxed);
    locks.pending = 0;
    locks.pendingStack = -1;
    locks.waiting = false;
}

void StackSummary() {
    Line line;
    for (std::size_t index = 0; index < MaxStacks; ++index) {
        auto& entry = g_stacks[index];
        if (entry.hash.load(std::memory_order_acquire) == 0 || entry.frames.load(std::memory_order_acquire) == 0) continue;
        line.Text("[delivery-lock-check] stack-count id=");
        line.Decimal(index);
        line.Text(" arrivals=");
        line.Decimal(entry.arrivals.load(std::memory_order_relaxed));
        line.Text(" ran=");
        line.Decimal(entry.ran.load(std::memory_order_relaxed));
        line.Text(" deferred=");
        line.Decimal(entry.deferred.load(std::memory_order_relaxed));
        line.Flush();
    }
}

void Wrapper(int signal, siginfo_t* info, void* context) {
    const auto handler = g_handler.load(std::memory_order_acquire);
    if (handler == nullptr) return;
    const int saved = errno;
    WriteMaps();
    auto& locks = t_locks;
    if (locks.waiting) Outcome(true, 0);
    const auto ip = static_cast<std::uintptr_t>(static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP]);
    int stack = -1;
    if (GuestCode(ip)) {
        g_guest.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_host.fetch_add(1, std::memory_order_relaxed);
        stack = HostStack(ip);
    }
    const int count = locks.depth < MaxHeld ? locks.depth : MaxHeld;
    std::array<Held, MaxHeld> held;
    for (int index = 0; index < count; ++index) held[static_cast<std::size_t>(index)] = locks.held[static_cast<std::size_t>(index)];
    const auto arrivals = g_arrivals.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count == 0) g_clean.fetch_add(1, std::memory_order_relaxed);
    errno = saved;
    const auto start = Now();
    handler(signal, info, context);
    const auto nanos = Now() - start;
    const int after = errno;
    locks.parked = held;
    locks.pending = count;
    locks.pendingStack = stack;
    locks.waiting = true;
    if (nanos >= ParkedNanos) Outcome(true, nanos);
    if (arrivals % 64u == 0) Summary();
    if (arrivals % 512u == 0) StackSummary();
    errno = after;
}

[[gnu::destructor]] void Final() {
    if (g_arrivals.load(std::memory_order_relaxed) == 0) return;
    Summary();
    StackSummary();
}

int Track(int result, const volatile void* lock, const void* site, Kind kind) {
    if (result == 0) Push(const_cast<const void*>(lock), site, kind);
    return result;
}

}

extern "C" {

int pthread_mutex_lock(pthread_mutex_t* mutex) {
    return Track(NEXT(pthread_mutex_lock)(mutex), mutex, __builtin_return_address(0), Kind::Mutex);
}

int pthread_mutex_trylock(pthread_mutex_t* mutex) {
    return Track(NEXT(pthread_mutex_trylock)(mutex), mutex, __builtin_return_address(0), Kind::Mutex);
}

int pthread_mutex_timedlock(pthread_mutex_t* mutex, const timespec* deadline) {
    return Track(NEXT(pthread_mutex_timedlock)(mutex, deadline), mutex, __builtin_return_address(0), Kind::Mutex);
}

int pthread_mutex_clocklock(pthread_mutex_t* mutex, clockid_t clock, const timespec* deadline) {
    return Track(NEXT(pthread_mutex_clocklock)(mutex, clock, deadline), mutex, __builtin_return_address(0), Kind::Mutex);
}

int pthread_mutex_unlock(pthread_mutex_t* mutex) {
    Pop(mutex);
    return NEXT(pthread_mutex_unlock)(mutex);
}

int pthread_rwlock_rdlock(pthread_rwlock_t* lock) {
    return Track(NEXT(pthread_rwlock_rdlock)(lock), lock, __builtin_return_address(0), Kind::Read);
}

int pthread_rwlock_tryrdlock(pthread_rwlock_t* lock) {
    return Track(NEXT(pthread_rwlock_tryrdlock)(lock), lock, __builtin_return_address(0), Kind::Read);
}

int pthread_rwlock_timedrdlock(pthread_rwlock_t* lock, const timespec* deadline) {
    return Track(NEXT(pthread_rwlock_timedrdlock)(lock, deadline), lock, __builtin_return_address(0), Kind::Read);
}

int pthread_rwlock_clockrdlock(pthread_rwlock_t* lock, clockid_t clock, const timespec* deadline) {
    return Track(NEXT(pthread_rwlock_clockrdlock)(lock, clock, deadline), lock, __builtin_return_address(0), Kind::Read);
}

int pthread_rwlock_wrlock(pthread_rwlock_t* lock) {
    return Track(NEXT(pthread_rwlock_wrlock)(lock), lock, __builtin_return_address(0), Kind::Write);
}

int pthread_rwlock_trywrlock(pthread_rwlock_t* lock) {
    return Track(NEXT(pthread_rwlock_trywrlock)(lock), lock, __builtin_return_address(0), Kind::Write);
}

int pthread_rwlock_timedwrlock(pthread_rwlock_t* lock, const timespec* deadline) {
    return Track(NEXT(pthread_rwlock_timedwrlock)(lock, deadline), lock, __builtin_return_address(0), Kind::Write);
}

int pthread_rwlock_clockwrlock(pthread_rwlock_t* lock, clockid_t clock, const timespec* deadline) {
    return Track(NEXT(pthread_rwlock_clockwrlock)(lock, clock, deadline), lock, __builtin_return_address(0), Kind::Write);
}

int pthread_rwlock_unlock(pthread_rwlock_t* lock) {
    Pop(lock);
    return NEXT(pthread_rwlock_unlock)(lock);
}

int pthread_spin_lock(pthread_spinlock_t* lock) {
    return Track(NEXT(pthread_spin_lock)(lock), lock, __builtin_return_address(0), Kind::Spin);
}

int pthread_spin_trylock(pthread_spinlock_t* lock) {
    return Track(NEXT(pthread_spin_trylock)(lock), lock, __builtin_return_address(0), Kind::Spin);
}

int pthread_spin_unlock(pthread_spinlock_t* lock) {
    Pop(const_cast<const int*>(lock));
    return NEXT(pthread_spin_unlock)(lock);
}

int pthread_cond_wait(pthread_cond_t* condition, pthread_mutex_t* mutex) {
    Held held {};
    const bool tracked = Pop(mutex, &held);
    const int result = NEXT(pthread_cond_wait)(condition, mutex);
    if (tracked) Push(held.lock, held.site, held.kind);
    return result;
}

int pthread_cond_timedwait(pthread_cond_t* condition, pthread_mutex_t* mutex, const timespec* deadline) {
    Held held {};
    const bool tracked = Pop(mutex, &held);
    const int result = NEXT(pthread_cond_timedwait)(condition, mutex, deadline);
    if (tracked) Push(held.lock, held.site, held.kind);
    return result;
}

int pthread_cond_clockwait(pthread_cond_t* condition, pthread_mutex_t* mutex, clockid_t clock, const timespec* deadline) {
    Held held {};
    const bool tracked = Pop(mutex, &held);
    const int result = NEXT(pthread_cond_clockwait)(condition, mutex, clock, deadline);
    if (tracked) Push(held.lock, held.site, held.kind);
    return result;
}

int pthread_kill(pthread_t thread, int signal) {
    if (signal == SIGUSR1 && pthread_equal(thread, pthread_self()) != 0) {
        if (t_locks.waiting) Outcome(false, 0);
    }
    return NEXT(pthread_kill)(thread, signal);
}

int sigaction(int signal, const struct sigaction* action, struct sigaction* previous) {
    if (signal == SIGUSR1 && action != nullptr) {
        if ((action->sa_flags & SA_SIGINFO) != 0 && action->sa_sigaction != nullptr && action->sa_sigaction != &Wrapper) {
            g_handler.store(action->sa_sigaction, std::memory_order_release);
            struct sigaction wrapped = *action;
            wrapped.sa_sigaction = &Wrapper;
            Line line;
            line.Text("[delivery-lock-check] SIGUSR1 handler ");
            line.Hex(reinterpret_cast<std::uintptr_t>(action->sa_sigaction));
            line.Text(" wrapped");
            line.Flush();
            return NEXT(sigaction)(signal, &wrapped, previous);
        }
        if ((action->sa_flags & SA_SIGINFO) == 0 || action->sa_sigaction == nullptr) g_handler.store(nullptr, std::memory_order_release);
    }
    return NEXT(sigaction)(signal, action, previous);
}

}
