#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

// The calling thread resolves a host access violation in the exception handler (KytyExceptionHandler): the
// faulting instruction (guest code, or HLE code touching guest memory) runs again when the handler returns.
// BufferCache::ReadMemory lets such a thread wait for the GPU itself. Direct HandleFault calls and emulator
// writes reported through InvalidateMemory (kernel / HLE code that may hold its own locks) are not inside it.
inline bool& InRetriedFault() noexcept {
	static thread_local bool inside = false;
	return inside;
}

class RetriedFaultScope {
public:
	RetriedFaultScope() noexcept: m_previous(InRetriedFault()) { InRetriedFault() = true; }
	~RetriedFaultScope() { InRetriedFault() = m_previous; }
	RetriedFaultScope(const RetriedFaultScope&)            = delete;
	RetriedFaultScope& operator=(const RetriedFaultScope&) = delete;

private:
	bool m_previous;
};

// The access violation this thread is resolving (set by KytyExceptionHandler): the faulting instruction. Zero outside such a
// fault.
struct FaultSite {
	uint64_t rip = 0;
};

inline FaultSite& CurrentFaultSite() noexcept {
	static thread_local FaultSite site;
	return site;
}

class FaultSiteScope {
public:
	explicit FaultSiteScope(uint64_t rip) noexcept: m_previous(CurrentFaultSite()) { CurrentFaultSite() = FaultSite {rip}; }
	~FaultSiteScope() { CurrentFaultSite() = m_previous; }
	FaultSiteScope(const FaultSiteScope&)            = delete;
	FaultSiteScope& operator=(const FaultSiteScope&) = delete;

private:
	FaultSite m_previous;
};

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// RegionManager::ReleaseGpuToCpu: removes the access watcher of every page of read_open and the write watcher of
	// every page of write_open of the region at base_addr (the opening halves of UpdatePageWatchersForRegion<false, true>
	// and <false, false>) in one pass under the region's lock, with the same run rule: one protection call per run of
	// pages whose mode changed, at the run's final mode (a page in both masks goes from no access to read-write in one
	// call instead of two). A page under the image read watch stays no access.
	void OpenPageWatchersForRegion(uint64_t base_addr, const RegionBits& read_open, const RegionBits& write_open);

	// The image read watch: one bit per page, kept apart from the watcher counts (their access side is a single bit and
	// has its own owner), makes the page no-access so that the next CPU access faults while the bytes of a GPU-written
	// image are not in guest memory. Arm sets it over the range, Disarm clears it (both reprotect the pages whose
	// protection changes); a page without the bit goes back to what its watchers need.
	void ArmImageRead(uint64_t vaddr, uint64_t size);
	void DisarmImageRead(uint64_t vaddr, uint64_t size);
	// The page's bit, without the region's lock: a stale answer is possible and the callers allow for it.
	[[nodiscard]] bool ImageReadArmed(uint64_t vaddr) const noexcept;
	[[nodiscard]] bool ImageReadArmedInRange(uint64_t vaddr, uint64_t size) const noexcept;
	// True when a plain read of the page would fault only for the image read watch: the emulator's own reads then take the
	// page from the direct-memory backing. False for a page the buffer cache also watches (its fault must run).
	[[nodiscard]] bool ImageReadBypass(uint64_t vaddr) const noexcept;
	// Whether the page's watchers alone let the access through (false without a tracker region).
	[[nodiscard]] bool WatchersAllow(uint64_t vaddr, bool is_write) const;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
