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

namespace {

using namespace Libs::Graphics;

constexpr uint64_t Page = TRACKER_PAGE_SIZE;
constexpr uint64_t Block = 0x10000;
constexpr uint64_t GuestBytes = 3 * Block;
constexpr int CpuReadWrite = 0x03;
constexpr uint32_t GameValue = 0x600dcafe;

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
	constexpr uint64_t Mapped = GuestBytes + Block;
	if (sceKernelMapFlexibleMemory(&guest, Mapped, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_buffer_cache_pages: FAILED: could not map guest backing\n", stderr);
		return 1;
	}
	const auto address = reinterpret_cast<uint64_t>(guest);
	const auto failures = CheckBufferCachePages(gpu.graphics, BlockAt(address, 0));
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
