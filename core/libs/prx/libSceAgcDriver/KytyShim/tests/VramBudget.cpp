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
	static void SetReportedMemory(GraphicContext& graphics, uint64_t budget, uint64_t usage) {
		graphics.m_memory_override_enabled = true;
		graphics.m_memory_budget_override  = budget;
		graphics.m_memory_usage_override   = usage;
	}

	static uint64_t TextureTrigger(const TextureCache& cache) { return cache.m_trigger_gc_memory; }
	static uint64_t TexturePressure(const TextureCache& cache) { return cache.m_pressure_gc_memory; }
	static uint64_t TextureCritical(const TextureCache& cache) { return cache.m_critical_gc_memory; }
	static uint64_t BufferTrigger(const BufferCache& cache) { return cache.m_trigger_gc_memory; }
	static uint64_t BufferCritical(const BufferCache& cache) { return cache.m_critical_gc_memory; }

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

const char* CheckVramBudget(GraphicContext& graphics, TextureCache& textures, BufferCache& buffers) {
	const auto tex_trigger_large  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_large = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_large = VramBudgetTestAccess::TextureCritical(textures);
	const auto buf_trigger_large  = VramBudgetTestAccess::BufferTrigger(buffers);
	const auto buf_critical_large = VramBudgetTestAccess::BufferCritical(buffers);

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 0);
	textures.RunGarbageCollector();
	buffers.RunGarbageCollector();

	const auto tex_trigger_small  = VramBudgetTestAccess::TextureTrigger(textures);
	const auto tex_pressure_small = VramBudgetTestAccess::TexturePressure(textures);
	const auto tex_critical_small = VramBudgetTestAccess::TextureCritical(textures);
	const auto buf_trigger_small  = VramBudgetTestAccess::BufferTrigger(buffers);
	const auto buf_critical_small = VramBudgetTestAccess::BufferCritical(buffers);

	if (tex_trigger_small >= tex_trigger_large || tex_pressure_small >= tex_pressure_large ||
	    tex_critical_small >= tex_critical_large) {
		return "texture cache GC did not refresh thresholds when reported budget decreased";
	}
	if (tex_trigger_small != 0) {
		return "texture cache trigger threshold was non-zero at <= 8 GiB budget";
	}
	if (buf_trigger_small >= buf_trigger_large || buf_critical_small >= buf_critical_large) {
		return "buffer cache GC did not refresh thresholds when reported budget decreased";
	}

	vk::ImageCreateInfo recycle_info {};
	recycle_info.imageType     = vk::ImageType::e2D;
	recycle_info.format        = vk::Format::eR8G8B8A8Unorm;
	recycle_info.extent        = vk::Extent3D {64, 64, 1};
	recycle_info.mipLevels     = 1;
	recycle_info.arrayLayers   = 1;
	recycle_info.samples       = vk::SampleCountFlagBits::e1;
	recycle_info.tiling        = vk::ImageTiling::eOptimal;
	recycle_info.usage         = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	recycle_info.sharingMode   = vk::SharingMode::eExclusive;
	recycle_info.initialLayout = vk::ImageLayout::eUndefined;

	VulkanImage recycled_image {};
	if (!graphics.CreateImage(recycle_info, recycled_image)) {
		return "could not create recyclable image for recycle pool check";
	}
	graphics.DeleteImage(recycled_image);
	if (graphics.RecycledImageCount() == 0) {
		return "deleted recyclable image did not enter the recycle pool";
	}

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 6 * GiB);
	textures.RunGarbageCollector();
	if (graphics.RecycledImageCount() != 0) {
		return "pressure GC pass did not drain the recycled image pool";
	}
	return nullptr;
}

const char* CheckGuestEviction(GraphicContext& graphics, TextureCache& textures, uint64_t address) {
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

	VramBudgetTestAccess::SetReportedMemory(graphics, 6 * GiB, 6 * GiB);
	for (uint32_t i = 0; i < 4; i++) {
		textures.RunGarbageCollector();
	}
	if (VramBudgetTestAccess::IsImageRegistered(textures, id)) {
		return "critical memory pressure did not evict a recent unused texture";
	}
	return nullptr;
}

const char* CheckVramCase(GraphicContext& graphics, uint64_t address) {
	VramBudgetTestAccess::SetReportedMemory(graphics, (address == 0 ? 16 : 6) * GiB, 0);

	RenderContext    renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures  = renderer.GetTextureCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	const auto* failure = address == 0 ? CheckVramBudget(graphics, textures, renderer.GetBufferCache())
	                                  : CheckGuestEviction(graphics, textures, address);
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	return failure;
}

}

extern "C" int RunKytyVramBudgetTest(bool guest_eviction) {
	const char* name = guest_eviction ? "kyty_shim_vram_budget_guest_eviction" : "kyty_shim_vram_budget";
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	if (const auto* unavailable = gpu.Initialize()) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "%s: %s, no usable Vulkan device (%s)\n",
		             name, required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
	if (!gpu.graphics.CreateAllocator()) {
		std::fprintf(stderr, "%s: FAILED: could not create the allocator\n", name);
		return 1;
	}

	uint64_t address = 0;
	if (guest_eviction) {
		void* guest = nullptr;
		if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
			std::fprintf(stderr, "%s: FAILED: could not map guest backing\n", name);
			return 1;
		}
		address = reinterpret_cast<uint64_t>(guest);
	}
	const bool invalid_range = guest_eviction && !GuestRange {address, GuestBytes}.Valid();
	const char* failure = invalid_range ? nullptr : CheckVramCase(gpu.graphics, address);
	if (guest_eviction && sceKernelMunmap(address, GuestBytes) != 0) {
		std::fprintf(stderr, "%s: FAILED: could not unmap guest backing\n", name);
		return 1;
	}
	if (invalid_range) {
		std::fprintf(stderr, "%s: skipped, guest range 0x%llx is outside Kyty bounds\n",
		             name, static_cast<unsigned long long>(address));
		return 77;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "%s: FAILED: %s\n", name, failure);
		return 1;
	}
	std::printf("%s: ok\n", name);
	return 0;
}
