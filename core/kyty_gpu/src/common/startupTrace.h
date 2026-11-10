#ifndef KYTY_COMMON_STARTUPTRACE_H_
#define KYTY_COMMON_STARTUPTRACE_H_

#include "common/onceValue.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

// KYTY_TRACE_STARTUP=1 (diagnostic, default off): where the wall time of a start goes. From the first call of
// Enabled() a low-priority reporter thread prints one "STARTUP t=..." line per second of wall time for the first
// 180 s (KYTY_TRACE_STARTUP=<n> with n > 1: for n seconds), then one STARTUP_TOTAL line. Each line holds what the
// counters below collected in that second (count/ms, count/MiB/ms) plus process-wide numbers read from the OS
// (CPU cores busy, GPU-thread CPU, file reads of the whole process, page faults, private bytes). Nothing else changes:
// when the switch is off every hook is one load of a constant-initialized flag and a not-taken branch.
//
// Fields (times in ms within that second):
//   t        seconds since the process was created (as the runner's elapsed_seconds)
//   f        frames presented so far
//   cpu      process CPU time / wall time (cores busy, all threads)        gpu_cpu  GPU thread CPU %
//   busy     GPU thread processing guest submissions (PERF "process")    fence    fence waits (mostly GPU thread)
//   rb / dl  synchronous readbacks of GPU-written memory, any thread (PERF read_memory) / image downloads
//   sh       shader permutations the GPU thread translated and compiled: n/ms (tr = translate, cmp = materialize,
//            SPIR-V compile and module)
//   bgsh     background (KYTY_SHADER_PRECOMPILE) shader compiles n/ms    pwait  GPU thread waits for them n/ms
//   reuse    the shader reuse: permutations the GPU thread took from the background compiles n/ms (ms = what the
//            workers spent on what it skipped, the estimate of the time saved); miss = permutations it compiled
//            itself after a lookup
//   pipe     vkCreate*Pipelines: g = graphics and c = compute on the GPU thread, bg = background workers,
//            slow = creates of 50 ms or more (driver-cache misses)
//   img      images created n/MiB/ms     up   image uploads (guest -> GPU, detile) n/MiB/ms
//   buf      buffers created n/MiB/ms    bup  buffer uploads through staging n/MiB/ms (tmp = staging was full)
//   kopen    sceKernelOpen n/ms          kread  sceKernelRead/Pread/Preadv of files n/MiB/ms (inv = the
//            guest-memory invalidation inside them)
//   apr      APR (libAmpr) file reads n/MiB/ms (open = opening the host file); sub = submissions/commands queued;
//            lat = submissions completed / their submit-to-complete ms (queueing, reads and guest waits)
//   amm      AMM map/unmap commands executed n/ms
//   mem      guest memory calls n/MiB: map = sceKernelMap(Named)DirectMemory/MapFlexibleMemory, unmap = sceKernelMunmap,
//            pc/pd = sceKernelMemoryPoolCommit/Decommit, prot = sceKernelMprotect/Mtypeprotect; and what the address
//            space did for them n/MiB: hc/hr = private host pages committed/released (fresh pages: every first touch
//            after hc is a demand-zero page fault), view/unview = views of the direct-memory backing mapped/unmapped,
//            zero = backing ranges cleared
//   io       ReadFile operations of the whole process n/MiB (OS counters: includes the pipeline cache, driver, logs)
//   pf       page faults (all kinds: commit, guard, tracker write-protect)   priv  private bytes MiB
namespace Common::StartupTrace {

enum class Counter : uint32_t {
	GpuBusy,
	GpuFence,
	GpuReadback,
	Shader,
	ShaderTranslate,
	ShaderCompile,
	ShaderBackground,
	PrecompileWait,
	ShaderReuse,
	ShaderReuseMiss,
	PipeGraphics,
	PipeCompute,
	PipeBackground,
	PipeSlow,
	ImageCreate,
	ImageUpload,
	BufferCreate,
	BufferUpload,
	BufferUploadTemp,
	KernelOpen,
	KernelRead,
	KernelInvalidate,
	AprRead,
	AprOpen,
	AprSubmit,
	AprLatency,
	AmmCommand,
	MemMap,
	MemUnmap,
	MemPoolCommit,
	MemPoolDecommit,
	MemProtect,
	MemHostCommit,
	MemHostRelease,
	MemView,
	MemUnview,
	MemZero,
	Count
};

struct Slot {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> nanos {0};
	std::atomic<uint64_t> bytes {0};
};

// Constant-initialized: no guard, no registration for destruction.
inline constinit Slot g_slots[static_cast<uint32_t>(Counter::Count)] {};

// Starts the reporter thread (once). Called by Enabled() when the switch is on.
void StartReporter(uint32_t seconds);

// The counters are collected for KYTY_TRACE_STARTUP (with the reporter).
inline bool Enabled() {
	static constinit Common::OnceValue<bool> enabled_once;
	return enabled_once.Get([] {
		const char* value   = std::getenv("KYTY_TRACE_STARTUP");
		const long  seconds = value != nullptr ? std::strtol(value, nullptr, 10) : 0;
		if (seconds <= 0) return false;
		StartReporter(seconds == 1 ? 180u : static_cast<uint32_t>(seconds));
		return true;
	});
}

inline void Add(Counter counter, uint64_t nanos, uint64_t bytes = 0) noexcept {
	auto& slot = g_slots[static_cast<uint32_t>(counter)];
	slot.count.fetch_add(1, std::memory_order_relaxed);
	if (nanos != 0) {
		slot.nanos.fetch_add(nanos, std::memory_order_relaxed);
	}
	if (bytes != 0) slot.bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void AddMs(Counter counter, double ms, uint64_t bytes = 0) noexcept {
	Add(counter, ms > 0.0 ? static_cast<uint64_t>(ms * 1e6) : 0u, bytes);
}

inline uint64_t NanosSince(std::chrono::steady_clock::time_point begin) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
}

// Times a scope into one counter when the switch is on (`bytes` may be set before the scope ends).
class Scope {
public:
	explicit Scope(Counter counter, uint64_t bytes = 0): m_counter(counter), m_on(Enabled()), m_bytes(bytes) {
		if (m_on) m_start = std::chrono::steady_clock::now();
	}
	~Scope() {
		if (m_on) Add(m_counter, NanosSince(m_start), m_bytes);
	}
	void SetBytes(uint64_t bytes) { m_bytes = bytes; }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Counter                               m_counter;
	bool                                  m_on;
	uint64_t                              m_bytes;
	std::chrono::steady_clock::time_point m_start {};
};

inline constinit std::atomic<uint32_t> g_frame {0};

// The frame just presented (swapchain).
inline void NoteFrame(uint32_t frame) noexcept {
	if (Enabled()) g_frame.store(frame, std::memory_order_relaxed);
}

// Called once on the guest GPU thread: the reporter reads its CPU time.
void NoteGpuThread();

// Seconds since the process was created (the test runner's clock; 0 if unknown). Works with the switch off.
double ProcessAge();

// Committed bytes of the address space by kind (KYTY_TRACE_MEM_CENSUS): private regions of >= 1 GiB, 64 MiB - 1 GiB,
// 1 - 64 MiB and < 1 MiB, then mapped (sections) and image (modules). All zero where unknown.
struct VmSummary {
	uint64_t private_huge = 0, private_large = 0, private_medium = 0, private_small = 0, mapped = 0, image = 0;
};
VmSummary ProcessVmSummary();

} // namespace Common::StartupTrace

#endif // KYTY_COMMON_STARTUPTRACE_H_
