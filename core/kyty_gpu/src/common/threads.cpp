#include "common/threads.h"

#include "common/assert.h"

#include <atomic>
#include <cerrno>
#include <chrono>             // IWYU pragma: keep
#include <condition_variable> // IWYU pragma: keep
#include <mutex>

// macOS has no clock_nanosleep.
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS && !defined(__APPLE__)
#define KYTY_POSIX_HIGH_RES_SLEEP
#include <ctime>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
// IWYU pragma: no_include <winbase.h>
constexpr DWORD KYTY_CS_SPIN_COUNT = 4000;

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

static void SleepHighResolution100ns(uint64_t units_100ns) {
	if (units_100ns == 0) {
		return;
	}

	thread_local HANDLE timer = CreateWaitableTimerExW(
	    nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
	if (timer == nullptr) {
		thread_local HANDLE fallback_timer = CreateWaitableTimerW(nullptr, TRUE, nullptr);
		timer                              = fallback_timer;
	}

	if (timer != nullptr) {
		LARGE_INTEGER due_time {};
		due_time.QuadPart = -static_cast<LONGLONG>(units_100ns);
		if (SetWaitableTimerEx(timer, &due_time, 0, nullptr, nullptr, nullptr, 0) != 0) {
			WaitForSingleObject(timer, INFINITE);
			return;
		}
	}

	std::this_thread::sleep_for(std::chrono::nanoseconds(units_100ns * 100));
}

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <synchapi.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <__mutex_base>
// IWYU pragma: no_include <__threading_support>
// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <winerror.h>

using InitializeConditionVariable_func_t = /*WINBASEAPI*/ VOID WINAPI (*)(PCONDITION_VARIABLE);
using WakeConditionVariable_func_t    = /*WINBASEAPI*/ VOID       WINAPI (*)(PCONDITION_VARIABLE);
using WakeAllConditionVariable_func_t = /*WINBASEAPI*/ VOID    WINAPI (*)(PCONDITION_VARIABLE);
using SleepConditionVariableCS_func_t = /*WINBASEAPI*/ BOOL    WINAPI (*)(PCONDITION_VARIABLE,
                                                                          PCRITICAL_SECTION, DWORD);

static InitializeConditionVariable_func_t ResolveInitializeConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<InitializeConditionVariable_func_t>(
		    GetProcAddress(h, "InitializeConditionVariable"));
	}
	return nullptr;
}
static WakeConditionVariable_func_t ResolveWakeConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<WakeConditionVariable_func_t>(
		    GetProcAddress(h, "WakeConditionVariable"));
	}
	return nullptr;
}
static WakeAllConditionVariable_func_t ResolveWakeAllConditionVariable() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<WakeAllConditionVariable_func_t>(
		    GetProcAddress(h, "WakeAllConditionVariable"));
	}
	return nullptr;
}
static SleepConditionVariableCS_func_t ResolveSleepConditionVariableCS() {
	if (HMODULE h = GetModuleHandle("KernelBase"); h != nullptr) // @suppress("Invalid arguments")
	{
		return reinterpret_cast<SleepConditionVariableCS_func_t>(
		    GetProcAddress(h, "SleepConditionVariableCS"));
	}
	return nullptr;
}

#endif

#ifdef KYTY_POSIX_HIGH_RES_SLEEP
// An absolute deadline preserves the requested sleep across signal interruptions.
static void SleepHighResolutionNanos(uint64_t nanos) {
	if (nanos == 0) {
		return;
	}

	constexpr uint64_t NANOS_PER_SEC = 1000000000;

	timespec deadline {};
	if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
		return;
	}

	auto target_nsec = static_cast<uint64_t>(deadline.tv_nsec) + nanos;
	deadline.tv_sec += static_cast<time_t>(target_nsec / NANOS_PER_SEC);
	deadline.tv_nsec = static_cast<long>(target_nsec % NANOS_PER_SEC);

	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {
	}
}
#endif

namespace Common {

using thread_id_t = std::thread::id;

struct MutexPrivate {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	MutexPrivate() { InitializeCriticalSectionAndSpinCount(&m_cs, KYTY_CS_SPIN_COUNT); }
	~MutexPrivate() { DeleteCriticalSection(&m_cs); }
	KYTY_CLASS_NO_COPY(MutexPrivate);
	CRITICAL_SECTION m_cs {};
#else
	std::recursive_mutex m_mutex;
#endif
};

struct CondVarPrivate {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	CondVarPrivate() {
		static auto func = ResolveInitializeConditionVariable();
		EXIT_NOT_IMPLEMENTED(func == nullptr);
		func(&m_cv);
	}
	~CondVarPrivate() = default;
	KYTY_CLASS_NO_COPY(CondVarPrivate);
	CONDITION_VARIABLE m_cv {};
#else
	std::condition_variable_any m_cv;
#endif
};

static wait_poll_func_t g_cond_wait_poll_callback = nullptr;

struct ThreadPrivate {
	ThreadPrivate(thread_func_t f, void* a): func(f), arg(a), m_thread(&Run, this) {}

	static void Run(ThreadPrivate* t) {
		t->unique_id = Thread::GetThreadIdUnique();
		t->started   = true;
		t->func(t->arg);
	}

	thread_func_t    func;
	void*            arg;
	std::atomic_bool finished    = false;
	std::atomic_bool auto_delete = false;
	std::atomic_bool started     = false;
	int              unique_id   = 0;
	std::thread      m_thread;
};

static thread_id_t      g_main_thread;
static int              g_main_thread_int;
static std::atomic<int> g_thread_counter = 0;

void InitializeThreads() {
	g_main_thread     = std::this_thread::get_id();
	g_main_thread_int = Thread::GetThreadIdUnique();
}

Thread::Thread(thread_func_t func, void* arg)
    : m_thread(std::make_unique<ThreadPrivate>(func, arg)) {
	while (!m_thread->started) {
		Common::Thread::SleepMicro(1000);
	}
}

Thread::~Thread() {
	EXIT_IF(!m_thread->finished && !m_thread->auto_delete);

	m_thread.reset();
}

void Thread::Join() {
	EXIT_IF(m_thread->finished || m_thread->auto_delete);

	m_thread->m_thread.join();

	m_thread->finished = true;
}

void Thread::Detach() {
	EXIT_IF(m_thread->finished || m_thread->auto_delete);

	m_thread->auto_delete = true;
	m_thread->m_thread.detach();
}

void Thread::SleepMicro(uint32_t micros) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns(static_cast<uint64_t>(micros) * 10);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(static_cast<uint64_t>(micros) * 1000);
#else
	std::this_thread::sleep_for(std::chrono::microseconds(micros));
#endif
}

void Thread::SleepNano(uint64_t nanos) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns((nanos + 99) / 100);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(nanos);
#else
	std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
#endif
}

bool Thread::IsMainThread() {
	return g_main_thread == std::this_thread::get_id();
}

std::string Thread::GetId() const {
	std::stringstream ss;
	ss << m_thread->m_thread.get_id();
	return ss.str();
}

int Thread::GetUniqueId() const {
	return m_thread->unique_id;
}

std::string Thread::GetThreadId() {
	std::stringstream ss;
	ss << std::this_thread::get_id();
	return ss.str();
}

void Thread::SetCurrentName(const char* name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (name == nullptr || name[0] == '\0') return;
	// Windows 10 1607+; looked up so older kernels and SDK targets still run.
	using SetThreadDescriptionFunc = HRESULT(WINAPI*)(HANDLE, PCWSTR);
	static const auto set_description = reinterpret_cast<SetThreadDescriptionFunc>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription")));
	if (set_description == nullptr) return;
	const std::string  narrow(name);
	const std::wstring wide(narrow.begin(), narrow.end());
	set_description(GetCurrentThread(), wide.c_str());
#else
	(void)name;
#endif
}

std::string Thread::GetCurrentName() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	using GetThreadDescriptionFunc = HRESULT(WINAPI*)(HANDLE, PWSTR*);
	static const auto get_description = reinterpret_cast<GetThreadDescriptionFunc>(
	    reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
	if (get_description == nullptr) return {};
	PWSTR wide = nullptr;
	if (FAILED(get_description(GetCurrentThread(), &wide)) || wide == nullptr) return {};
	std::string name;
	for (const wchar_t* c = wide; *c != 0; ++c) name.push_back(static_cast<char>(*c < 128 ? *c : '?'));
	LocalFree(wide);
	return name;
#else
	return {};
#endif
}

Mutex::Mutex(): m_mutex(std::make_unique<MutexPrivate>()) {}

Mutex::~Mutex() {
	m_mutex.reset();
}

void Mutex::Lock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	EnterCriticalSection(&m_mutex->m_cs);
#else
	m_mutex->m_mutex.lock();
#endif
}

void Mutex::Unlock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	LeaveCriticalSection(&m_mutex->m_cs);
#else
	m_mutex->m_mutex.unlock();
#endif
}

bool Mutex::TryLock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return (TryEnterCriticalSection(&m_mutex->m_cs) != 0);
#else
	return m_mutex->m_mutex.try_lock();
#endif
}

CondVar::CondVar(): m_cond_var(std::make_unique<CondVarPrivate>()) {}

CondVar::~CondVar() {
	m_cond_var.reset();
}

void CondVar::Wait(Mutex* mutex) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	std::unique_lock<std::recursive_mutex> cpp_lock(mutex->m_mutex->m_mutex, std::adopt_lock_t());
#endif
	auto poll_callback = [&] {
		auto* callback = g_cond_wait_poll_callback;
		if (callback == nullptr) {
			return;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		LeaveCriticalSection(&mutex->m_mutex->m_cs);
		callback();
		EnterCriticalSection(&mutex->m_mutex->m_cs);
#else
		cpp_lock.unlock();
		callback();
		cpp_lock.lock();
#endif
	};
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveSleepConditionVariableCS();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	if (g_cond_wait_poll_callback == nullptr) {
		func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, INFINITE);
	} else {
		if (func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, 10) == 0 &&
		    GetLastError() == ERROR_TIMEOUT) {
			poll_callback();
		}
	}
#else
	if (g_cond_wait_poll_callback == nullptr) {
		m_cond_var->m_cv.wait(cpp_lock);
	} else {
		if (m_cond_var->m_cv.wait_for(cpp_lock, std::chrono::microseconds(10000)) ==
		    std::cv_status::timeout) {
			poll_callback();
		}
	}
	cpp_lock.release();
#endif
}

void CondVar::SetWaitPollCallback(wait_poll_func_t callback) {
	g_cond_wait_poll_callback = callback;
}

bool CondVar::WaitFor(Mutex* mutex, uint32_t micros) {
	bool ok = false;
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	std::unique_lock<std::recursive_mutex> cpp_lock(mutex->m_mutex->m_mutex, std::adopt_lock_t());
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveSleepConditionVariableCS();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	ok = !(func(&m_cond_var->m_cv, &mutex->m_mutex->m_cs, (micros < 1000 ? 1 : micros / 1000)) ==
	           0 &&
	       GetLastError() == ERROR_TIMEOUT);
#else
	ok = (m_cond_var->m_cv.wait_for(cpp_lock, std::chrono::microseconds(micros)) ==
	      std::cv_status::no_timeout);
	cpp_lock.release();
#endif
	return ok;
}

void CondVar::Signal() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveWakeConditionVariable();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	func(&m_cond_var->m_cv);
#else
	m_cond_var->m_cv.notify_one();
#endif
}

void CondVar::SignalAll() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static auto func = ResolveWakeAllConditionVariable();
	EXIT_NOT_IMPLEMENTED(func == nullptr);
	func(&m_cond_var->m_cv);
#else
	m_cond_var->m_cv.notify_all();
#endif
}

int Thread::GetThreadIdUnique() {
	static thread_local int tid = ++g_thread_counter;
	return tid;
}

uint64_t CurrentThreadCpuNs() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	FILETIME creation {}, exit {}, kernel {}, user {};
	if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user) == 0) return 0;
	const auto ticks = [](const FILETIME& t) { return (static_cast<uint64_t>(t.dwHighDateTime) << 32u) | t.dwLowDateTime; };
	return (ticks(kernel) + ticks(user)) * 100u;
#else
	return 0;
#endif
}

void Thread::RaiseCurrentPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
}

void Thread::PinCurrent(int logical_cpu) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (logical_cpu >= 0 && logical_cpu < 64) {
		const auto mask = DWORD_PTR {1} << static_cast<uint32_t>(logical_cpu);
		::SetThreadAffinityMask(::GetCurrentThread(), mask);
		::SetThreadIdealProcessor(::GetCurrentThread(), static_cast<DWORD>(logical_cpu));
	}
#else
	(void)logical_cpu;
#endif
}

std::string Thread::ReserveCores(std::span<const int> logical_cpus, size_t min_other_cores) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	ULONG length = 0;
	(void)::GetSystemCpuSetInformation(nullptr, 0, &length, ::GetCurrentProcess(), 0);
	if (length == 0) {
		return "no CPU set information";
	}
	std::vector<uint8_t> buffer(length);
	auto* const          first = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
	if (!::GetSystemCpuSetInformation(first, length, &length, ::GetCurrentProcess(), 0)) {
		return "no CPU set information";
	}
	const auto for_each_set = [&](auto&& func) {
		for (ULONG offset = 0; offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= length;) {
			const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
			if (info->Type == CpuSetInformation) {
				func(info->CpuSet);
			}
			offset += info->Size;
		}
	};
	// The physical cores (group, core index) of the reserved logical processors.
	std::vector<std::pair<WORD, BYTE>> reserved;
	for_each_set([&](const auto& set) {
		for (const auto cpu: logical_cpus) {
			if (set.Group == 0 && cpu >= 0 && set.LogicalProcessorIndex == cpu) {
				reserved.emplace_back(set.Group, set.CoreIndex);
			}
		}
	});
	std::vector<ULONG>                 others;
	std::vector<std::pair<WORD, BYTE>> other_cores;
	for_each_set([&](const auto& set) {
		const std::pair core {set.Group, set.CoreIndex};
		if (std::ranges::find(reserved, core) == reserved.end()) {
			others.push_back(set.Id);
			if (std::ranges::find(other_cores, core) == other_cores.end()) {
				other_cores.push_back(core);
			}
		}
	});
	// Smaller processors keep every core for the guest.
	if (reserved.empty() || other_cores.size() < min_other_cores) {
		return "not reserved, other cores " + std::to_string(other_cores.size());
	}
	const BOOL ok = ::SetProcessDefaultCpuSets(::GetCurrentProcess(), others.data(), static_cast<ULONG>(others.size()));
	return (ok ? "reserved, other cores " : "reservation failed, other cores ") + std::to_string(other_cores.size());
#else
	(void)logical_cpus;
	(void)min_other_cores;
	return "unsupported";
#endif
}

std::vector<LogicalCpu> Thread::SystemLogicalCpus() {
	std::vector<LogicalCpu> cpus;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	ULONG length = 0;
	(void)::GetSystemCpuSetInformation(nullptr, 0, &length, ::GetCurrentProcess(), 0);
	std::vector<uint8_t> buffer(length);
	if (length == 0 ||
	    !::GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data()), length, &length,
	                                  ::GetCurrentProcess(), 0)) {
		return cpus;
	}
	for (ULONG offset = 0; offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= length;) {
		const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
		if (info->Type == CpuSetInformation && info->CpuSet.Group == 0 && info->CpuSet.LogicalProcessorIndex < 64) {
			cpus.push_back({info->CpuSet.LogicalProcessorIndex, info->CpuSet.CoreIndex, info->CpuSet.EfficiencyClass,
			                info->CpuSet.LastLevelCacheIndex, 0});
		}
		offset += info->Size;
	}
	// The size of the last-level cache each logical processor shares (level 3, else level 2).
	DWORD bytes = 0;
	(void)::GetLogicalProcessorInformationEx(RelationCache, nullptr, &bytes);
	std::vector<uint8_t> caches(bytes);
	if (bytes != 0 && ::GetLogicalProcessorInformationEx(
	                      RelationCache, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(caches.data()), &bytes)) {
		std::vector<std::pair<uint32_t, uint64_t>> best(cpus.size()); // (level, size)
		for (DWORD offset = 0; offset < bytes;) {
			const auto* info  = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(caches.data() + offset);
			const auto& cache = info->Cache;
			if (cache.Level >= 2 && cache.GroupMask.Group == 0) {
				for (size_t i = 0; i < cpus.size(); i++) {
					if (((cache.GroupMask.Mask >> cpus[i].logical) & 1u) != 0u) {
						best[i] = std::max(best[i], std::pair {uint32_t {cache.Level}, uint64_t {cache.CacheSize}});
					}
				}
			}
			offset += info->Size;
		}
		for (size_t i = 0; i < cpus.size(); i++) cpus[i].llc_bytes = best[i].second;
	}
#endif
	return cpus;
}

} // namespace Common
