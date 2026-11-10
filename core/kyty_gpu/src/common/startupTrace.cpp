#include "common/startupTrace.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <thread>

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#include <psapi.h>
#endif

namespace Common::StartupTrace {

namespace {

constexpr uint32_t Slots = static_cast<uint32_t>(Counter::Count);

struct Snapshot {
	std::array<uint64_t, Slots> count {};
	std::array<uint64_t, Slots> nanos {};
	std::array<uint64_t, Slots> bytes {};
};

// The OS side: process CPU time, GPU thread CPU time, ReadFile operations/bytes, page faults, private bytes.
struct OsSample {
	uint64_t cpu_100ns     = 0;
	uint64_t gpu_cpu_100ns = 0;
	uint64_t read_ops      = 0;
	uint64_t read_bytes    = 0;
	uint64_t page_faults   = 0;
	uint64_t private_bytes = 0;
};

std::atomic<void*> g_gpu_thread {nullptr}; // Windows: a duplicated handle of the guest GPU thread

#if defined(_WIN32)
uint64_t FileTime100ns(const FILETIME& time) {
	return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
}
#endif

OsSample SampleOs() {
	OsSample sample;
#if defined(_WIN32)
	FILETIME create {}, exit {}, kernel {}, user {};
	if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user) != 0) {
		sample.cpu_100ns = FileTime100ns(kernel) + FileTime100ns(user);
	}
	if (void* thread = g_gpu_thread.load(std::memory_order_acquire); thread != nullptr) {
		if (GetThreadTimes(static_cast<HANDLE>(thread), &create, &exit, &kernel, &user) != 0) {
			sample.gpu_cpu_100ns = FileTime100ns(kernel) + FileTime100ns(user);
		}
	}
	IO_COUNTERS io {};
	if (GetProcessIoCounters(GetCurrentProcess(), &io) != 0) {
		sample.read_ops   = io.ReadOperationCount;
		sample.read_bytes = io.ReadTransferCount;
	}
	PROCESS_MEMORY_COUNTERS_EX memory {};
	if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),
	                            sizeof(memory)) != 0) {
		sample.page_faults   = memory.PageFaultCount;
		sample.private_bytes = memory.PrivateUsage;
	}
#endif
	return sample;
}

// Seconds between the process creation and now (the runner counts from the process start); 0 if unknown.
double ProcessAgeSeconds() {
#if defined(_WIN32)
	FILETIME create {}, exit {}, kernel {}, user {};
	if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user) != 0) {
		FILETIME now {};
		GetSystemTimeAsFileTime(&now);
		const auto created = FileTime100ns(create);
		const auto current = FileTime100ns(now);
		return current > created ? static_cast<double>(current - created) / 1e7 : 0.0;
	}
#endif
	return 0.0;
}

Snapshot Take() {
	Snapshot snapshot;
	for (uint32_t i = 0; i < Slots; i++) {
		snapshot.count[i] = g_slots[i].count.exchange(0, std::memory_order_relaxed);
		snapshot.nanos[i] = g_slots[i].nanos.exchange(0, std::memory_order_relaxed);
		snapshot.bytes[i] = g_slots[i].bytes.exchange(0, std::memory_order_relaxed);
	}
	return snapshot;
}

unsigned long long N(const Snapshot& s, Counter c) {
	return s.count[static_cast<uint32_t>(c)];
}
unsigned long long Ms(const Snapshot& s, Counter c) {
	return s.nanos[static_cast<uint32_t>(c)] / 1000000u;
}
unsigned long long Mib(const Snapshot& s, Counter c) {
	return s.bytes[static_cast<uint32_t>(c)] >> 20u;
}

void Print(const char* label, double t, double wall_seconds, const Snapshot& s, const OsSample& os_delta,
           const OsSample& os_now) {
	using C = Counter;
	const double wall = wall_seconds > 0.0 ? wall_seconds : 1.0;
	std::fprintf(stderr,
	             "%s t=%.0f f=%u cpu=%.1f gpu_cpu=%.0f%% busy=%llu fence=%llu rb=%llu"
	             " sh=%llu/%llu(tr%llu,cmp%llu) bgsh=%llu/%llu pwait=%llu/%llu reuse=%llu/%llu(miss%llu)"
	             " pipe=g%llu/%llu,c%llu/%llu,bg%llu/%llu,slow%llu",
	             label, t, g_frame.load(std::memory_order_relaxed),
	             static_cast<double>(os_delta.cpu_100ns) / 1e7 / wall,
	             static_cast<double>(os_delta.gpu_cpu_100ns) / 1e5 / wall, Ms(s, C::GpuBusy), Ms(s, C::GpuFence),
	             Ms(s, C::GpuReadback), N(s, C::Shader), Ms(s, C::Shader),
	             Ms(s, C::ShaderTranslate), Ms(s, C::ShaderCompile), N(s, C::ShaderBackground),
	             Ms(s, C::ShaderBackground), N(s, C::PrecompileWait), Ms(s, C::PrecompileWait), N(s, C::ShaderReuse),
	             Ms(s, C::ShaderReuse), N(s, C::ShaderReuseMiss),
	             N(s, C::PipeGraphics), Ms(s, C::PipeGraphics), N(s, C::PipeCompute), Ms(s, C::PipeCompute),
	             N(s, C::PipeBackground), Ms(s, C::PipeBackground), N(s, C::PipeSlow));
	std::fprintf(stderr,
	             " img=%llu/%llu/%llu up=%llu/%llu/%llu buf=%llu/%llu/%llu bup=%llu/%llu/%llu(tmp%llu)"
	             " kopen=%llu/%llu kread=%llu/%llu/%llu(inv%llu) apr=%llu/%llu/%llu(open%llu,sub%llu/%llu,lat%llu/%llu)"
	             " amm=%llu/%llu",
	             N(s, C::ImageCreate), Mib(s, C::ImageCreate), Ms(s, C::ImageCreate), N(s, C::ImageUpload),
	             Mib(s, C::ImageUpload), Ms(s, C::ImageUpload), N(s, C::BufferCreate), Mib(s, C::BufferCreate),
	             Ms(s, C::BufferCreate), N(s, C::BufferUpload), Mib(s, C::BufferUpload), Ms(s, C::BufferUpload),
	             N(s, C::BufferUploadTemp), N(s, C::KernelOpen), Ms(s, C::KernelOpen), N(s, C::KernelRead),
	             Mib(s, C::KernelRead), Ms(s, C::KernelRead), Ms(s, C::KernelInvalidate), N(s, C::AprRead),
	             Mib(s, C::AprRead), Ms(s, C::AprRead), Ms(s, C::AprOpen), N(s, C::AprSubmit),
	             static_cast<unsigned long long>(s.bytes[static_cast<uint32_t>(C::AprSubmit)]), N(s, C::AprLatency),
	             Ms(s, C::AprLatency), N(s, C::AmmCommand), Ms(s, C::AmmCommand));
	std::fprintf(stderr,
	             " mem=map%llu/%llu,unmap%llu/%llu,pc%llu/%llu,pd%llu/%llu,prot%llu/%llu,hc%llu/%llu,hr%llu/%llu,"
	             "view%llu/%llu,unview%llu/%llu,zero%llu/%llu io=%llu/%llu pf=%llu priv=%llu\n",
	             N(s, C::MemMap), Mib(s, C::MemMap), N(s, C::MemUnmap), Mib(s, C::MemUnmap), N(s, C::MemPoolCommit),
	             Mib(s, C::MemPoolCommit), N(s, C::MemPoolDecommit), Mib(s, C::MemPoolDecommit), N(s, C::MemProtect),
	             Mib(s, C::MemProtect), N(s, C::MemHostCommit), Mib(s, C::MemHostCommit), N(s, C::MemHostRelease),
	             Mib(s, C::MemHostRelease), N(s, C::MemView), Mib(s, C::MemView), N(s, C::MemUnview),
	             Mib(s, C::MemUnview), N(s, C::MemZero), Mib(s, C::MemZero),
	             static_cast<unsigned long long>(os_delta.read_ops),
	             static_cast<unsigned long long>(os_delta.read_bytes >> 20u),
	             static_cast<unsigned long long>(os_delta.page_faults),
	             static_cast<unsigned long long>(os_now.private_bytes >> 20u));
	std::fflush(stderr);
}

OsSample Delta(const OsSample& now, const OsSample& before) {
	const auto d = [](uint64_t a, uint64_t b) { return a > b ? a - b : 0u; };
	return {.cpu_100ns     = d(now.cpu_100ns, before.cpu_100ns),
	        .gpu_cpu_100ns = d(now.gpu_cpu_100ns, before.gpu_cpu_100ns),
	        .read_ops      = d(now.read_ops, before.read_ops),
	        .read_bytes    = d(now.read_bytes, before.read_bytes),
	        .page_faults   = d(now.page_faults, before.page_faults),
	        .private_bytes = now.private_bytes};
}

void Run(uint32_t seconds) {
#if defined(_WIN32)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
	const double age   = ProcessAgeSeconds();
	const auto   start = std::chrono::steady_clock::now();
	Snapshot     total {};
	OsSample     os_start {}; // the first line's OS deltas count from the process start
	OsSample     os_before = os_start;
	auto         previous  = start;
	for (uint32_t second = 1; second <= seconds; second++) {
		std::this_thread::sleep_until(start + std::chrono::seconds(second));
		// The first line also holds what was counted before the reporter started.
		const auto snapshot = Take();
		for (uint32_t i = 0; i < Slots; i++) {
			total.count[i] += snapshot.count[i];
			total.nanos[i] += snapshot.nanos[i];
			total.bytes[i] += snapshot.bytes[i];
		}
		const auto   now     = std::chrono::steady_clock::now();
		const OsSample os_now = SampleOs();
		const double wall =
		    second == 1 ? age + std::chrono::duration<double>(now - start).count()
		                : std::chrono::duration<double>(now - previous).count();
		Print("STARTUP", age + std::chrono::duration<double>(now - start).count(), wall,
		      snapshot, Delta(os_now, os_before), os_now);
		os_before = os_now;
		previous  = now;
	}
	const auto     now    = std::chrono::steady_clock::now();
	const OsSample os_now = SampleOs();
	const double   t      = age + std::chrono::duration<double>(now - start).count();
	Print("STARTUP_TOTAL", t, t, total, Delta(os_now, os_start), os_now);
}

} // namespace

double ProcessAge() {
	return ProcessAgeSeconds();
}

VmSummary ProcessVmSummary() {
	VmSummary summary;
#if defined(_WIN32)
	// Committed bytes per allocation (a reservation's committed regions together), sorted into the kinds above.
	MEMORY_BASIC_INFORMATION info {};
	const char*              address    = nullptr;
	const void*              base       = nullptr;
	uint64_t                 committed  = 0;
	DWORD                    type       = 0;
	const auto               close_base = [&]() {
        if (base == nullptr || committed == 0) return;
        if (type == MEM_PRIVATE) {
            if (committed >= (1ull << 30u)) summary.private_huge += committed;
            else if (committed >= (64ull << 20u)) summary.private_large += committed;
            else if (committed >= (1ull << 20u)) summary.private_medium += committed;
            else summary.private_small += committed;
        } else if (type == MEM_MAPPED) {
            summary.mapped += committed;
        } else if (type == MEM_IMAGE) {
            summary.image += committed;
        }
	};
	while (VirtualQuery(address, &info, sizeof(info)) == sizeof(info)) {
		if (info.AllocationBase != base) {
			close_base();
			base      = info.AllocationBase;
			committed = 0;
			type      = info.Type;
		}
		if (info.State == MEM_COMMIT) committed += info.RegionSize;
		const char* next = static_cast<const char*>(info.BaseAddress) + info.RegionSize;
		if (next <= address) break;
		address = next;
	}
	close_base();
#endif
	return summary;
}

void StartReporter(uint32_t seconds) {
	std::fprintf(stderr,
	             "TRACE_STARTUP enabled (%u s of per-second STARTUP lines: GPU thread, shaders, pipelines, uploads, "
	             "file reads)\n",
	             seconds);
	std::fflush(stderr);
	std::thread(Run, seconds).detach();
}

void NoteGpuThread() {
	if (!Enabled()) return;
#if defined(_WIN32)
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                    THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0) != 0) {
		void* expected = nullptr;
		if (!g_gpu_thread.compare_exchange_strong(expected, handle)) {
			CloseHandle(handle);
		}
	}
#endif
}

} // namespace Common::StartupTrace
