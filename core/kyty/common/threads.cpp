#include "common/threads.h"

#include <algorithm>
#include <bit>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <ctime>
#include <memory>
#include <thread>
#include <vector>

#if defined(ANYPS5_KYTY)
#include "common/cacheTopology.h"
#endif

#if defined(ANYPS5_KYTY) && KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <pthread.h>
#include <sched.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#include <tlhelp32.h>
constexpr DWORD KYTY_CS_SPIN_COUNT = 4000;

static std::atomic<uint64_t> g_unreserved_processors {0};
static std::atomic<uint32_t> g_next_ideal_processor {0};
static thread_local bool t_reserved_core = false;

static void KeepThreadOffReservedCores(HANDLE thread) noexcept {
	const auto unreserved = g_unreserved_processors.load(std::memory_order_acquire);
	if (unreserved == 0) {
		return;
	}
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	if (!::GetProcessAffinityMask(::GetCurrentProcess(), &process_mask, &system_mask)) {
		return;
	}
	const auto previous = ::SetThreadAffinityMask(thread, static_cast<DWORD_PTR>(unreserved));
	if (previous == 0) {
		return;
	}
	if (previous != process_mask) {
		(void)::SetThreadAffinityMask(thread, previous);
		return;
	}
	auto       allowed = unreserved;
	const auto count   = static_cast<uint32_t>(std::popcount(allowed));
	for (auto skip = g_next_ideal_processor.fetch_add(1, std::memory_order_relaxed) % count;
	     skip != 0; skip--) {
		allowed &= allowed - 1;
	}
	(void)::SetThreadIdealProcessor(thread, static_cast<DWORD>(std::countr_zero(allowed)));
}

static void NTAPI KeepNewThreadOffReservedCores(PVOID , DWORD reason, PVOID ) {
	if (reason == DLL_THREAD_ATTACH) {
		KeepThreadOffReservedCores(::GetCurrentThread());
	}
}

#if defined(ANYPS5_KYTY)
extern "C" [[gnu::section(".CRT$XLK"), gnu::used]] PIMAGE_TLS_CALLBACK kyty_keep_new_thread_off_reserved_cores =
    KeepNewThreadOffReservedCores;
#else
#pragma section(".CRT$XLK", read)
extern "C" __declspec(allocate(".CRT$XLK")) const PIMAGE_TLS_CALLBACK kyty_keep_new_thread_off_reserved_cores =
    KeepNewThreadOffReservedCores;
#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:kyty_keep_new_thread_off_reserved_cores")
#endif

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

static void SleepHighResolution100ns(uint64_t units_100ns) {
	if (units_100ns == 0) {
		return;
	}
	thread_local const std::unique_ptr<void, decltype(&CloseHandle)> timer {
	    CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
		                       TIMER_ALL_ACCESS),
	    &CloseHandle};
	LARGE_INTEGER due_time {};
	due_time.QuadPart = -static_cast<LONGLONG>(units_100ns);
	if (SetWaitableTimerEx(timer.get(), &due_time, 0, nullptr, nullptr, nullptr, 0) != 0) {
		WaitForSingleObject(timer.get(), INFINITE);
	}
}

static CRITICAL_SECTION* ToCs(unsigned char* storage) {
	return reinterpret_cast<CRITICAL_SECTION*>(storage);
}

static CONDITION_VARIABLE* ToCv(void** storage) {
	return reinterpret_cast<CONDITION_VARIABLE*>(storage);
}

#endif

namespace Common {

static wait_poll_func_t g_cond_wait_poll_callback = nullptr;

static std::thread::id g_main_thread;

void InitializeThreads() {
	g_main_thread = std::this_thread::get_id();
}

void Thread::RaiseCurrentPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
}

void Thread::LowerCurrentPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

void Thread::PinCurrent(int logical_cpu) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (logical_cpu >= 0 && logical_cpu < 64) {
		const auto mask = DWORD_PTR {1} << static_cast<uint32_t>(logical_cpu);
		::SetThreadAffinityMask(::GetCurrentThread(), mask);
		::SetThreadIdealProcessor(::GetCurrentThread(), static_cast<DWORD>(logical_cpu));
		const auto unreserved = g_unreserved_processors.load(std::memory_order_acquire);
		t_reserved_core = unreserved != 0 && (unreserved & mask) == 0;
	}
#elif defined(ANYPS5_KYTY)
	if (logical_cpu >= 0 && logical_cpu < CPU_SETSIZE) {
		cpu_set_t set;
		CPU_ZERO(&set);
		CPU_SET(logical_cpu, &set);
		(void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	}
#else
	(void)logical_cpu;
#endif
}

#if defined(ANYPS5_KYTY) && KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
void Thread::KeepCurrentOnLargestCache() {
	static const std::vector<int> cache_cpus = LargestCacheCpus("/sys/devices/system/cpu");
	cpu_set_t                     allowed;
	CPU_ZERO(&allowed);
	if (cache_cpus.empty() || sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
		return;
	}
	cpu_set_t set;
	CPU_ZERO(&set);
	for (const int cpu: cache_cpus) {
		if (cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) {
			CPU_SET(cpu, &set);
		}
	}
	if (CPU_COUNT(&set) != 0 && !CPU_EQUAL(&set, &allowed)) {
		(void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	}
}
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool Thread::LogicalCpuExists(int logical_cpu) {
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	if (logical_cpu < 0 || logical_cpu >= 64 ||
	    !::GetProcessAffinityMask(::GetCurrentProcess(), &process_mask, &system_mask)) {
		return false;
	}
	return ((system_mask >> static_cast<uint32_t>(logical_cpu)) & 1u) != 0;
}
#endif

#if defined(ANYPS5_KYTY) && KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
template <typename Func>
static void ForEachProcessorInfo(LOGICAL_PROCESSOR_RELATIONSHIP relation, Func&& func) {
	DWORD length = 0;
	(void)::GetLogicalProcessorInformationEx(relation, nullptr, &length);
	if (length == 0) {
		return;
	}
	std::vector<uint8_t> buffer(length);
	if (!::GetLogicalProcessorInformationEx(
	        relation, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()), &length)) {
		return;
	}
	constexpr size_t HeaderSize = offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor);
	for (DWORD offset = 0; offset + HeaderSize <= length;) {
		const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
		if (info->Size == 0 || offset + info->Size > length) {
			break;
		}
		if (info->Relationship == relation) {
			func(*info);
		}
		offset += info->Size;
	}
}

static std::vector<int> GroupMaskCpus(const GROUP_AFFINITY* masks, WORD count) {
	std::vector<int> cpus;
	for (WORD index = 0; index < count; index++) {
		for (uint32_t bit = 0; bit < 64; bit++) {
			if (((masks[index].Mask >> bit) & 1u) != 0) {
				cpus.push_back(static_cast<int>(masks[index].Group * 64u + bit));
			}
		}
	}
	std::ranges::sort(cpus);
	return cpus;
}

static std::vector<CacheRecord> WindowsCaches() {
	constexpr size_t MasksOffset =
	    offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Cache) + offsetof(CACHE_RELATIONSHIP, GroupMasks);
	std::vector<CacheRecord> caches;
	ForEachProcessorInfo(RelationCache, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& info) {
		const auto& cache = info.Cache;
		const auto  masks = std::max<WORD>(cache.GroupCount, 1);
		if (cache.Type != CacheInstruction && MasksOffset + masks * sizeof(GROUP_AFFINITY) <= info.Size) {
			caches.push_back({cache.Level, cache.CacheSize, GroupMaskCpus(cache.GroupMasks, masks)});
		}
	});
	return caches;
}

static std::vector<CoreRecord> WindowsCores() {
	constexpr size_t MasksOffset =
	    offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor) + offsetof(PROCESSOR_RELATIONSHIP, GroupMask);
	std::vector<CoreRecord> cores;
	ForEachProcessorInfo(RelationProcessorCore, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& info) {
		const auto& core  = info.Processor;
		const auto  masks = std::max<WORD>(core.GroupCount, 1);
		if (MasksOffset + masks * sizeof(GROUP_AFFINITY) <= info.Size) {
			cores.push_back({core.EfficiencyClass, GroupMaskCpus(core.GroupMask, masks)});
		}
	});
	return cores;
}

bool Thread::PinCurrentToOwnCore(uint32_t index, uint32_t count) {
	static const std::vector<int> cpus = [count] {
		GROUP_AFFINITY current {};
		if (!::GetThreadGroupAffinity(::GetCurrentThread(), &current)) {
			return std::vector<int> {};
		}
		std::vector<int> allowed;
		for (const int cpu: LargestCacheCpus(WindowsCaches())) {
			if (cpu / 64 == current.Group && ((current.Mask >> static_cast<uint32_t>(cpu % 64)) & 1u) != 0) {
				allowed.push_back(cpu);
			}
		}
		return OneCpuPerCore(WindowsCores(), allowed, count);
	}();
	if (index >= cpus.size()) {
		return false;
	}
	PinCurrent(cpus[index] % 64);
	return true;
}

void Thread::KeepCurrentOnLargestCache() {
	static const std::vector<int> cache_cpus = LargestCacheCpus(WindowsCaches());
	GROUP_AFFINITY                current {};
	if (cache_cpus.empty() || !::GetThreadGroupAffinity(::GetCurrentThread(), &current)) {
		return;
	}
	KAFFINITY mask = 0;
	for (const int cpu: cache_cpus) {
		if (cpu / 64 == current.Group) {
			mask |= KAFFINITY {1} << static_cast<uint32_t>(cpu % 64);
		}
	}
	mask &= current.Mask;
	if (mask != 0 && mask != current.Mask) {
		GROUP_AFFINITY wanted {};
		wanted.Group = current.Group;
		wanted.Mask  = mask;
		(void)::SetThreadGroupAffinity(::GetCurrentThread(), &wanted, nullptr);
	}
}
#endif

bool Thread::HasReservedCore() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return t_reserved_core;
#else
	return false;
#endif
}

void Thread::ReserveCores(std::span<const int> logical_cpus) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	ULONG length = 0;
	(void)::GetSystemCpuSetInformation(nullptr, 0, &length, ::GetCurrentProcess(), 0);
	if (length == 0) {
		return;
	}
	std::vector<uint8_t> buffer(length);
	auto* const          first = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
	if (!::GetSystemCpuSetInformation(first, length, &length, ::GetCurrentProcess(), 0)) {
		return;
	}
	const auto for_each_set = [&](auto&& func) {
		for (ULONG offset = 0; offset + sizeof(SYSTEM_CPU_SET_INFORMATION) <= length;) {
			const auto* info =
			    reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
			if (info->Type == CpuSetInformation) {
				func(info->CpuSet);
			}
			offset += info->Size;
		}
	};
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
	constexpr size_t MinOtherCores = 6;
	if (reserved.empty() || other_cores.size() < MinOtherCores) {
		return;
	}
	(void)::SetProcessDefaultCpuSets(::GetCurrentProcess(), others.data(),
	                                 static_cast<ULONG>(others.size()));
	uint64_t unreserved = 0;
	for_each_set([&](const auto& set) {
		if (set.Group == 0 && set.LogicalProcessorIndex < 64 &&
		    std::ranges::find(reserved, std::pair {set.Group, set.CoreIndex}) == reserved.end()) {
			unreserved |= uint64_t {1} << set.LogicalProcessorIndex;
		}
	});
	g_unreserved_processors.store(unreserved, std::memory_order_release);
	if (HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	    snapshot != INVALID_HANDLE_VALUE) {
		THREADENTRY32 entry {};
		entry.dwSize = sizeof(entry);
		for (BOOL ok = ::Thread32First(snapshot, &entry); ok; ok = ::Thread32Next(snapshot, &entry)) {
			if (entry.th32OwnerProcessID != ::GetCurrentProcessId() ||
			    entry.th32ThreadID == ::GetCurrentThreadId()) {
				continue;
			}
			if (HANDLE thread = ::OpenThread(THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION,
			                                 FALSE, entry.th32ThreadID);
			    thread != nullptr) {
				KeepThreadOffReservedCores(thread);
				::CloseHandle(thread);
			}
		}
		::CloseHandle(snapshot);
	}
#else
	(void)logical_cpus;
#endif
}

void Thread::SleepMicro(uint32_t micros) {
	SleepNano(uint64_t {micros} * 1000);
}

void Thread::SleepNano(uint64_t nanos) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns((nanos + 99) / 100);
#elif defined(__APPLE__)
	std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
#else
	if (nanos == 0) {
		return;
	}
	constexpr uint64_t NANOS_PER_SEC = 1000000000;
	timespec deadline {};
	clock_gettime(CLOCK_MONOTONIC, &deadline);
	auto target_nsec = static_cast<uint64_t>(deadline.tv_nsec) + nanos;
	deadline.tv_sec += static_cast<time_t>(target_nsec / NANOS_PER_SEC);
	deadline.tv_nsec = static_cast<long>(target_nsec % NANOS_PER_SEC);
	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {
	}
#endif
}

bool Thread::IsMainThread() {
	return g_main_thread == std::this_thread::get_id();
}

int Thread::GetThreadIdUnique() {
	static std::atomic<int> counter = 0;
	static thread_local int tid     = ++counter;
	return tid;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
Mutex::Mutex() {
	static_assert(sizeof(CRITICAL_SECTION) == sizeof(m_cs) &&
	              alignof(CRITICAL_SECTION) <= alignof(Mutex));
	InitializeCriticalSectionAndSpinCount(ToCs(m_cs), KYTY_CS_SPIN_COUNT);
}

Mutex::~Mutex() {
	DeleteCriticalSection(ToCs(m_cs));
}

void Mutex::Lock() {
	EnterCriticalSection(ToCs(m_cs));
}

void Mutex::Unlock() {
	LeaveCriticalSection(ToCs(m_cs));
}

void CondVar::Signal() {
	WakeConditionVariable(ToCv(&m_cv));
}

void CondVar::SignalAll() {
	WakeAllConditionVariable(ToCv(&m_cv));
}
#endif

void CondVar::Wait(Mutex* mutex) {
	auto* callback = g_cond_wait_poll_callback;
	if (callback == nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs), INFINITE);
#else
		m_cv.wait(mutex->m_mutex);
#endif
	} else if (!WaitFor(mutex, 10000)) {
		mutex->Unlock();
		callback();
		mutex->Lock();
	}
}

void CondVar::SetWaitPollCallback(wait_poll_func_t callback) {
	g_cond_wait_poll_callback = callback;
}

bool CondVar::WaitFor(Mutex* mutex, uint32_t micros) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs),
	                                micros < 1000 ? 1 : micros / 1000) != 0 ||
	       GetLastError() != ERROR_TIMEOUT;
#else
	return m_cv.wait_for(mutex->m_mutex, std::chrono::microseconds(micros)) ==
	       std::cv_status::no_timeout;
#endif
}

}
