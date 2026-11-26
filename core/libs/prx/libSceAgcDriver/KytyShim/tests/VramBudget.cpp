#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

struct VramBudgetTestAccess {
	static void UpdateTextureThresholds(TextureCache& cache, uint64_t budget) {
		Common::ScopedLock lock {cache.m_lock};
		cache.UpdateMemoryThresholds(budget);
	}

	static void UpdateBufferThresholds(BufferCache& cache, uint64_t budget) {
		cache.UpdateMemoryThresholds(budget);
	}

	static uint64_t TextureTrigger(const TextureCache& cache) { return cache.m_trigger_gc_memory; }
	static uint64_t TexturePressure(const TextureCache& cache) { return cache.m_pressure_gc_memory; }
	static uint64_t TextureCritical(const TextureCache& cache) { return cache.m_critical_gc_memory; }
	static uint64_t BufferTrigger(const BufferCache& cache) { return cache.m_trigger_gc_memory; }
	static uint64_t BufferCritical(const BufferCache& cache) { return cache.m_critical_gc_memory; }

	static void ForceTextureCriticalPressure(TextureCache& cache) {
		Common::ScopedLock lock {cache.m_lock};
		cache.m_trigger_gc_memory  = 0;
		cache.m_pressure_gc_memory = 0;
		cache.m_critical_gc_memory = 0;
	}

	static ImageId CreateImage(TextureCache& cache, const ImageInfo& info) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.InsertImage(info);
	}

	static bool IsImageRegistered(TextureCache& cache, ImageId id) {
		Common::ScopedLock lock {cache.m_lock};
		const auto*        image = cache.m_slot_images.try_get(id);
		return image != nullptr && image->registered;
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t GiB          = 1024ull * 1024 * 1024;
constexpr uint64_t GuestBytes   = 0x20000;
constexpr int      CpuReadWrite = 0x03;

const char* CheckThresholdScaling(TextureCache& textures, BufferCache& buffers) {
	VramBudgetTestAccess::UpdateTextureThresholds(textures, 16 * GiB);
	const auto tex_trigger_large  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_large = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_large = VramBudgetTestAccess::TextureCritical(textures);

	VramBudgetTestAccess::UpdateTextureThresholds(textures, 6 * GiB);
	const auto tex_trigger_small  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_small = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_small = VramBudgetTestAccess::TextureCritical(textures);

	if (tex_trigger_small >= tex_trigger_large || tex_pressure_small >= tex_pressure_large ||
	    tex_critical_small >= tex_critical_large) {
		return "texture cache thresholds did not shrink when reported budget decreased";
	}
	if (tex_trigger_small != 0) {
		return "texture cache trigger threshold was non-zero at <= 8 GiB budget";
	}

	VramBudgetTestAccess::UpdateBufferThresholds(buffers, 16 * GiB);
	const auto buf_trigger_large  = VramBudgetTestAccess::BufferTrigger(buffers);
	const auto buf_critical_large = VramBudgetTestAccess::BufferCritical(buffers);

	VramBudgetTestAccess::UpdateBufferThresholds(buffers, 6 * GiB);
	const auto buf_trigger_small  = VramBudgetTestAccess::BufferTrigger(buffers);
	const auto buf_critical_small = VramBudgetTestAccess::BufferCritical(buffers);

	if (buf_trigger_small >= buf_trigger_large || buf_critical_small >= buf_critical_large) {
		return "buffer cache thresholds did not shrink when reported budget decreased";
	}
	return nullptr;
}

const char* CheckCriticalPressureEviction(GraphicContext& graphics, uint64_t address) {
	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	auto& buffers   = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	if (const auto* failure = CheckThresholdScaling(textures, buffers)) {
		return failure;
	}

	ImageInfo info;
	info.data.address = address;
	info.pixel_format = vk::Format::eR8G8B8A8Unorm;
	info.guest_format = Prospero::BufferFormat::k8_8_8_8UNorm;
	info.extent       = vk::Extent3D {64, 64, 1};
	info.pitch        = 64;
	info.UpdateSize();

	const auto id = VramBudgetTestAccess::CreateImage(textures, info);
	if (!VramBudgetTestAccess::IsImageRegistered(textures, id)) {
		return "test image was not registered";
	}

	VramBudgetTestAccess::ForceTextureCriticalPressure(textures);
	for (uint32_t i = 0; i < 4; i++) {
		textures.RunGarbageCollector();
	}
	if (VramBudgetTestAccess::IsImageRegistered(textures, id)) {
		return "critical memory pressure did not evict a recent unused texture";
	}

	graphics.TrimRecycledImages();
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	return nullptr;
}

}

extern "C" int RunKytyVramBudgetTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize()) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_vram_budget: %s, no usable Vulkan device (%s)\n",
		             required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fputs("kyty_shim_vram_budget: FAILED: could not create the allocator\n", stderr);
		return 1;
	}
	void* guest = nullptr;
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_vram_budget: FAILED: could not map guest backing\n", stderr);
		return 1;
	}
	const auto  address = reinterpret_cast<uint64_t>(guest);
	const char* failure = CheckCriticalPressureEviction(gpu.graphics, address);
	if (sceKernelMunmap(address, GuestBytes) != 0) {
		std::fputs("kyty_shim_vram_budget: FAILED: could not unmap guest backing\n", stderr);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "kyty_shim_vram_budget: FAILED: %s\n", failure);
		return 1;
	}
	std::puts("kyty_shim_vram_budget: ok");
	return 0;
}
