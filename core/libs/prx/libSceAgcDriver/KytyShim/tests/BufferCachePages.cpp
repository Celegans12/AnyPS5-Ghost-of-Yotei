#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/indirectWriteTables.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

struct BufferCacheGcTestAccess {
	static void ForceCriticalPressure(BufferCache& cache) {
		cache.m_trigger_gc_memory = 0;
		cache.m_critical_gc_memory = 0;
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t Page = TRACKER_PAGE_SIZE;
constexpr uint64_t Block = 0x10000;
constexpr uint64_t GuestBytes = 3 * Block;
constexpr int CpuReadWrite = 0x03;
constexpr uint32_t GameValue = 0x600dcafe;
constexpr uint64_t GcSlot = 2 * BufferCache::CACHING_PAGESIZE;
constexpr uint32_t GcPinned = 64;
constexpr uint64_t GcBytes = (GcPinned + 1) * GcSlot;
constexpr uint64_t UseScanCalls = 257;

uint64_t BlockAt(uint64_t address, uint32_t index) {
	return (address + Block - 1) / Block * Block + uint64_t {index} * Block;
}

const char* CheckGpuPageLeftByWriteAround(BufferCache& buffers, uint64_t block) {
	const auto page = block + 3 * Page;
	(void)buffers.ObtainBuffer(page, sizeof(GameValue), true);
	if (!buffers.IsRegionGpuModified(page, Page)) {
		return "a written buffer range did not make its page GPU-dirty";
	}
	if (!buffers.WriteAroundGpuWrites(page, &GameValue, sizeof(GameValue))) {
		return "the game write was not served around the GPU write";
	}
	uint32_t memory = 0;
	if (!Libs::LibKernel::Memory::TryReadBacking(page, &memory, sizeof(memory)) || memory != GameValue) {
		return "the game write did not reach the guest memory";
	}
	if (buffers.IsRegionGpuModified(page, Page)) {
		return "a page without GPU bytes stayed GPU-dirty after WriteAroundGpuWrites";
	}
	return nullptr;
}

const char* CheckMarkAheadWaitsForIndirectWrites(RenderContext& renderer, BufferCache& buffers, uint64_t block) {
	(void)buffers.ObtainBuffer(block, Block, false, false);
	buffers.InvalidateFaultedPage(block + Page);
	auto& tables = renderer.GetIndirectWriteTables();
	tables.NotePendingWrites();
	buffers.InvalidateFaultedPage(block + 2 * Page);
	const bool marked_ahead = buffers.IsRegionCpuDirty(block + 3 * Page, Page);
	buffers.NoteGpuWrites(block + 3 * Page, Page);
	const bool noted = buffers.IsRegionGpuModified(block + 3 * Page, Page);
	if (tables.TakePendingWrites()) {
		buffers.EndIndirectWrites();
	}
	if (marked_ahead) {
		return "a write fault marked pages ahead while indirect GPU writes were pending";
	}
	if (!noted) {
		return "an indirect GPU write noted after a write fault was ignored";
	}
	const std::array<uint8_t, Page> zero {};
	if (!buffers.WriteAroundGpuWrites(block + 3 * Page, zero.data(), zero.size()) ||
	    buffers.IsRegionGpuModified(block + 3 * Page, Page)) {
		return "the indirect GPU write could not be cleared again";
	}
	return nullptr;
}

const char* CheckMarkAheadStillMarks(BufferCache& buffers, uint64_t block) {
	buffers.InvalidateFaultedPage(block + Page);
	buffers.InvalidateFaultedPage(block + 2 * Page);
	if (!buffers.IsRegionCpuDirty(block + 3 * Page, Page) || !buffers.IsRegionCpuDirty(block + Block - Page, Page)) {
		return "a write fault after a CPU-dirty page no longer marked the rest of its block";
	}
	return nullptr;
}

const char* CheckMemoryGcKeepsPinnedBuffers(BufferCache& buffers, uint64_t address) {
	const auto slot = [address](uint32_t index) { return address + uint64_t {index} * GcSlot; };
	const auto live = [&](uint32_t first, uint32_t last) {
		uint32_t count = 0;
		for (auto index = first; index < last; index++) {
			count += buffers.RegisteredBuffer(slot(index), Page) ? 1 : 0;
		}
		return count;
	};
	uint64_t calls = 0;
	const auto collect = [&](uint64_t until) {
		for (; calls < until; calls++) {
			const auto read = buffers.RegisteredBuffer(slot(0), Page);
			if (calls % UseScanCalls == 0 && read) {
				(void)buffers.TouchLiveBuffer(read, buffers.GetBuffer(read).Handle(), slot(0), Page);
			}
			buffers.RunGarbageCollector();
		}
	};
	for (uint32_t index = 0; index < GcPinned; index++) {
		buffers.CacheFaultedMemory(slot(index), Page);
	}
	(void)buffers.FindBuffer(slot(GcPinned), Page);
	if (live(0, GcPinned + 1) != GcPinned + 1) {
		return "the buffers of the memory GC check were not registered";
	}
	BufferCacheGcTestAccess::ForceCriticalPressure(buffers);
	collect(100);
	if (live(0, GcPinned) != GcPinned) {
		return "the memory GC retired a device-address pinned buffer before a use scan could touch it";
	}
	if (live(GcPinned, GcPinned + 1) != 0) {
		return "pinned buffers kept by the memory GC stopped it from retiring the unpinned buffer behind them";
	}
	collect(400);
	if (live(1, GcPinned) != GcPinned - 1) {
		return "the memory GC retired an idle pinned buffer within one and a half use-scan intervals";
	}
	collect(1300);
	if (live(1, GcPinned) != 0) {
		return "the memory GC never retired an idle pinned buffer under memory pressure";
	}
	if (live(0, 1) != 1) {
		return "the memory GC retired a pinned buffer that every use scan touched";
	}
	return nullptr;
}

int Report(const char* failure) {
	if (failure == nullptr) {
		return 0;
	}
	std::fprintf(stderr, "kyty_shim_buffer_cache_pages: FAILED: %s\n", failure);
	return 1;
}

int CheckBufferCachePages(Libs::Graphics::GraphicContext& graphics, uint64_t address) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const auto first = BlockAt(address, 0);
	(void)buffers.ObtainBuffer(first, Block, false, false);
	int failures = Report(buffers.IsRegionCpuDirty(first, Block) ? "a buffer read did not leave its pages clean" : nullptr);
	if (failures == 0) {
		failures += Report(CheckGpuPageLeftByWriteAround(buffers, first));
	}
	if (failures == 0) {
		failures += Report(CheckMarkAheadWaitsForIndirectWrites(renderer, buffers, BlockAt(address, 1)));
		failures += Report(CheckMarkAheadStillMarks(buffers, first));
	}
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	(void)buffers.DropFreedRange(address, GuestBytes, nullptr, nullptr);
	return failures;
}

int CheckMemoryGc(Libs::Graphics::GraphicContext& graphics, uint64_t address) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const int failures = Report(CheckMemoryGcKeepsPinnedBuffers(buffers, address));
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	(void)buffers.DropFreedRange(address, GcBytes, nullptr, nullptr);
	return failures;
}

}

extern "C" int RunKytyBufferCachePagesTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize()) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_buffer_cache_pages: %s, no usable Vulkan device (%s)\n",
		             required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fputs("kyty_shim_buffer_cache_pages: FAILED: could not create the allocator\n", stderr);
		return 1;
	}
	void* guest = nullptr;
	constexpr uint64_t Mapped = GuestBytes + GcBytes + Block;
	if (sceKernelMapFlexibleMemory(&guest, Mapped, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_buffer_cache_pages: FAILED: could not map guest backing\n", stderr);
		return 1;
	}
	const auto address = reinterpret_cast<uint64_t>(guest);
	auto failures = CheckBufferCachePages(gpu.graphics, BlockAt(address, 0));
	failures += CheckMemoryGc(gpu.graphics, BlockAt(address, 0) + GuestBytes);
	if (sceKernelMunmap(address, Mapped) != 0) {
		std::fputs("kyty_shim_buffer_cache_pages: FAILED: could not unmap guest backing\n", stderr);
		return 1;
	}
	if (failures != 0) {
		return 1;
	}
	std::puts("kyty_shim_buffer_cache_pages: ok");
	return 0;
}
