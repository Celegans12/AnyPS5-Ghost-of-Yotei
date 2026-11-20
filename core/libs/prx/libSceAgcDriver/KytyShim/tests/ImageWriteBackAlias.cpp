#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/window.h"
#include "kernel/memory.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics {

struct TextureCacheTestAccess {
	static ImageId Create(TextureCache& cache, const ImageInfo& info) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.InsertImage(info);
	}

	static bool QueueDownload(TextureCache& cache, ImageId image) {
		Common::ScopedLock lock {cache.m_lock};
		return cache.DownloadImageMemory(image);
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t GuestBytes = 0x20000;
constexpr int CpuReadWrite = 0x03;
constexpr uint32_t ImageWords = 64;
constexpr uint32_t CopyWord = 9;
constexpr uint32_t FillWord = 21;
constexpr uint32_t WriteWords = 4;
constexpr uint32_t ImageValue = 0x19283746;
constexpr uint32_t CopyValue = 0xdecafbad;
constexpr uint32_t FillValue = 0x1234fedc;

struct HeadlessDevice {
	GraphicContext graphics;

	~HeadlessDevice() {
		if (graphics.allocator) graphics.DestroyAllocator();
		if (graphics.device) graphics.device.destroy();
		if (graphics.instance) graphics.instance.destroy();
	}
};

const char* CheckWriteBackOrder(GraphicContext& graphics, uint64_t address) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint64_t Bytes = WriteWords * sizeof(uint32_t);
	const auto source = address + GuestBytes / 2;
	const auto destination = address + CopyWord * sizeof(uint32_t);
	const auto fill_address = address + FillWord * sizeof(uint32_t);
	auto [source_buffer, source_offset] = buffers.ObtainBuffer(source, Bytes, true, false);
	source_buffer->Fill(source_offset, Bytes, CopyValue);
	auto [fill_buffer, fill_offset] = buffers.ObtainBuffer(fill_address, Bytes, true, false);
	fill_buffer->Fill(fill_offset, Bytes, 0);

	ImageInfo info;
	info.data.address = address;
	info.pixel_format = vk::Format::eR32Uint;
	info.guest_format = Prospero::BufferFormat::k32UInt;
	info.extent = vk::Extent3D {ImageWords, 1, 1};
	info.pitch = ImageWords;
	info.UpdateSize();
	const auto id = TextureCacheTestAccess::Create(textures, info);
	auto& image = textures.GetImage(id);
	const auto native = scheduler.Current().Handle();
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, native);
	const vk::ClearColorValue clear {.uint32 = {std::array<uint32_t, 4> {ImageValue, 0, 0, 0}}};
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	native.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &clear, 1, &range);
	textures.MarkGpuWritten(id);
	const auto tick = scheduler.CurrentTick();
	const bool queued = TextureCacheTestAccess::QueueDownload(textures, id) &&
	                    textures.WriteBackTick(destination, Bytes) == tick;
	buffers.CopyBuffer(destination, source, Bytes, false, false);
	buffers.FillBuffer(fill_address, Bytes, FillValue, false);
	const bool same_tick = scheduler.CurrentTick() == tick;

	const auto [target_buffer, target_offset] = buffers.ObtainBuffer(destination, Bytes, false, false);
	Buffer copied(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	              2 * Bytes);
	copied.CopyFrom(scheduler.Current(), *target_buffer, target_offset, 0, Bytes);
	copied.CopyFrom(scheduler.Current(), *fill_buffer, fill_offset, Bytes, Bytes);
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	(void)buffers.DropFreedRange(address, GuestBytes, false, nullptr, nullptr);
	if (!queued) return "the image write-back was not pending before the buffer writes";
	if (!same_tick) return "the image download and buffer transfers did not share a tick";
	copied.Invalidate(0, 2 * Bytes);
	std::array<uint32_t, 2 * WriteWords> buffer_values {};
	std::memcpy(buffer_values.data(), copied.Mapped().data(), sizeof(buffer_values));
	for (uint32_t i = 0; i < WriteWords; ++i) {
		if (buffer_values[i] != CopyValue || buffer_values[WriteWords + i] != FillValue) {
			return "the buffer transfers did not write the expected GPU data";
		}
	}
	std::array<uint32_t, ImageWords> backing {};
	if (!Libs::LibKernel::Memory::TryReadBacking(address, backing.data(), sizeof(backing))) {
		return "could not read the guest backing";
	}
	for (uint32_t i = 0; i < ImageWords; ++i) {
		const bool overlap = (i >= CopyWord && i < CopyWord + WriteWords) ||
		                     (i >= FillWord && i < FillWord + WriteWords);
		if (backing[i] != (overlap ? 0 : ImageValue)) {
			return overlap ? "old image data was published over the newer GPU buffer range"
			               : "the image write-back lost nonoverlapping bytes";
		}
	}
	if (textures.WriteBackTick(destination, Bytes) != 0) return "the write-back remained pending";
	return nullptr;
}

}

extern "C" int RunKytyImageWriteBackAliasTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	HeadlessDevice gpu;
	auto& graphics = gpu.graphics;
	std::string error;
	if (!CreateHeadlessVulkan(graphics, nullptr, error)) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_image_write_back_alias: %s, no usable Vulkan device (%s)\n",
		             required ? "FAILED" : "skipped", error.c_str());
		return required ? 1 : 77;
	}
	graphics.device.getQueue(graphics.queue_family, 0, &graphics.queue);
	if (!graphics.CreateAllocator()) {
		std::fputs("kyty_shim_image_write_back_alias: FAILED: could not create the allocator\n", stderr);
		return 1;
	}
	void* guest = nullptr;
	if (sceKernelMapFlexibleMemory(&guest, GuestBytes, CpuReadWrite, 0) != 0 || guest == nullptr) {
		std::fputs("kyty_shim_image_write_back_alias: FAILED: could not map guest backing\n", stderr);
		return 1;
	}
	const auto address = reinterpret_cast<uint64_t>(guest);
	const auto* failure = CheckWriteBackOrder(graphics, address);
	if (sceKernelMunmap(address, GuestBytes) != 0) {
		std::fputs("kyty_shim_image_write_back_alias: FAILED: could not unmap guest backing\n", stderr);
		return 1;
	}
	if (failure != nullptr) {
		std::fprintf(stderr, "kyty_shim_image_write_back_alias: FAILED: %s\n", failure);
		return 1;
	}
	std::puts("kyty_shim_image_write_back_alias: ok");
	return 0;
}
