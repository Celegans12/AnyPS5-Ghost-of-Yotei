#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <memory>
#include <vector>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute };

class PageManager final {
public:
	PageManager();
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] bool HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept;
	static constexpr uint64_t                   READ_GRANULE_BITS = 16;
	[[nodiscard]] const std::atomic<uint64_t>* ReadGranules() const noexcept;
	[[nodiscard]] bool IsWatched(uint64_t vaddr) const;
	[[nodiscard]] bool WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept;
	[[nodiscard]] bool AllowExecute(uint64_t vaddr);

	struct DeferredRange {
		uint64_t address = 0, size = 0;
	};
	void ReapplyProtection(uint64_t vaddr, uint64_t size);

	template <bool track, bool is_read = false>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask, std::vector<DeferredRange>* deferred = nullptr);

	using HotPageWritten = void (*)(void* context, uint64_t begin, uint64_t end);
	[[nodiscard]] static bool HotPagesEnabled() noexcept;
	[[nodiscard]] bool        HasHotPages() const noexcept;
	void NoteWriteFault(uint64_t vaddr);
	void CollectHotPageWrites(HotPageWritten written, void* context, uint64_t vaddr = 0, uint64_t size = 0);
	void ForgetHotPages(uint64_t vaddr, uint64_t size);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

}

#endif
