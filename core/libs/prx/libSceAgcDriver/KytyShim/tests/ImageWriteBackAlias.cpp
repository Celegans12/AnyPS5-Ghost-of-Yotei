#include "TestVulkanDevice.h"
#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

struct BufferCacheTestAccess {
	static void DiscardGpuWrites(BufferCache& cache, uint64_t address, uint64_t size) {
		cache.DiscardGpuWrites(address, size);
	}
};

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
	BufferCacheTestAccess::DiscardGpuWrites(buffers, address, GuestBytes);
	(void)buffers.DropFreedRange(address, GuestBytes, nullptr, nullptr);
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

const char* CheckWriteBackRefreshesBufferCopy(GraphicContext& graphics, uint64_t address) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint64_t Bytes = ImageWords * sizeof(uint32_t);
	constexpr uint64_t CoverBytes = GuestBytes / 4;
	(void)buffers.ObtainBuffer(address, CoverBytes, false);
	const bool clean = !buffers.IsRegionCpuDirty(address, CoverBytes);

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
	const bool queued = TextureCacheTestAccess::QueueDownload(textures, id);
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	const bool invalidated = buffers.IsRegionCpuDirty(address, Bytes);

	const auto [buffer, offset] = buffers.ObtainBuffer(address, CoverBytes, false);
	Buffer copied(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst, Bytes);
	copied.CopyFrom(scheduler.Current(), *buffer, offset, 0, Bytes);
	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	(void)buffers.DropFreedRange(address, CoverBytes, nullptr, nullptr);
	if (!clean) return "the buffer read did not leave the image pages clean";
	if (!queued) return "the second image write-back was not queued";
	if (!invalidated) return "the image write-back left the pages of an older buffer copy clean";
	copied.Invalidate(0, Bytes);
	std::array<uint32_t, ImageWords> values {};
	std::memcpy(values.data(), copied.Mapped().data(), sizeof(values));
	for (const auto value: values) {
		if (value != ImageValue) return "a buffer read after the image write-back served the copy from before it";
	}
	return nullptr;
}

const char* CheckPartialBufferCopy(GraphicContext& graphics, uint64_t address,
                                   Prospero::TileMode mode) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint32_t Width = 64;
	constexpr uint32_t Height = 32;
	constexpr uint64_t TexelBytes = sizeof(uint32_t);
	constexpr uint64_t CopyBytes = WriteWords * TexelBytes;
	ImageInfo info;
	info.data.address = address;
	info.pixel_format = vk::Format::eR32Uint;
	info.guest_format = Prospero::BufferFormat::k32UInt;
	info.extent = vk::Extent3D {Width, Height, 1};
	info.tile_mode = mode;
	info.UpdateSize();
	const auto id = TextureCacheTestAccess::Create(textures, info);
	auto& image = textures.GetImage(id);

	std::array<uint32_t, Width * Height> pixels {};
	for (uint32_t i = 0; i < pixels.size(); ++i) {
		pixels[i] = ImageValue + i;
	}
	Buffer initial(graphics, scheduler, MemoryUsage::Upload, 0,
	               vk::BufferUsageFlagBits::eTransferSrc, sizeof(pixels));
	std::memcpy(initial.Mapped().data(), pixels.data(), sizeof(pixels));
	initial.Flush(0, sizeof(pixels));
	const std::array copies {vk::BufferImageCopy {
	    0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {}, info.extent}};
	image.Upload(copies, initial.Handle(), 0, sizeof(pixels));
	textures.MarkGpuWritten(id);

	const auto& block = info.tiling.block;
	const uint64_t offset = info.IsTiled() ? block.block_size - 2 * TexelBytes
	                                      : (Width + 5) * TexelBytes;
	const auto source = address + GuestBytes / 4;
	const auto [source_buffer, source_offset] = buffers.ObtainBuffer(source, CopyBytes, true, false);
	source_buffer->Fill(source_offset, CopyBytes, CopyValue);
	buffers.CopyBuffer(address + offset, source, CopyBytes, false, false);

	TextureCache::ImageDesc desc {.info = info};
	desc.view_info.format = info.pixel_format;
	const auto view = textures.FindTexture(id, desc);
	Buffer readback(graphics, scheduler, MemoryUsage::Download, 0,
	                vk::BufferUsageFlagBits::eTransferDst, sizeof(pixels));
	image.Download(copies, readback.Handle(), 0, sizeof(pixels));
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	readback.Invalidate(0, sizeof(pixels));
	std::array<uint32_t, Width * Height> actual {};
	std::memcpy(actual.data(), readback.Mapped().data(), sizeof(actual));
	BufferCacheTestAccess::DiscardGpuWrites(buffers, address, GuestBytes / 2);
	(void)buffers.DropFreedRange(address, GuestBytes / 2, nullptr, nullptr);
	if (!view) return "could not rebind the resident image after the partial copy";

	uint32_t covered = 0;
	bool stale = false;
	const auto& mip = info.mip_layout[0];
	for (uint32_t y = 0; y < Height; ++y) {
		for (uint32_t x = 0; x < Width; ++x) {
			uint64_t guest_offset = (uint64_t {y} * mip.linear_pitch + x) * TexelBytes;
			if (info.IsTiled()) {
				uint32_t in_block = 0;
				if (!TileGetBlockOffset(block, x % block.block_width, y % block.block_height, 0,
				                        in_block)) {
					return "could not map a tiled texel in the reference";
				}
				const auto block_index = y / block.block_height * (mip.pitch / block.block_width) +
				                         x / block.block_width;
				guest_offset = mip.offset + uint64_t {block_index} * block.block_size + in_block;
			}
			const auto index = y * Width + x;
			if (guest_offset >= offset && guest_offset < offset + CopyBytes) {
				covered++;
				stale |= actual[index] != CopyValue;
			} else if (actual[index] != pixels[index]) {
				return "the partial copy changed texels outside its guest range";
			}
		}
	}
	if (covered != WriteWords) return "the reference did not cover the copied words";
	return stale ? "the resident image still held stale texels after the partial copy" : nullptr;
}

const char* CheckRangePicksLatestWrite(GraphicContext& graphics, uint64_t address) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint64_t PairStride = 0x1000;
	const auto create = [&](uint64_t at, bool packed) {
		ImageInfo info;
		info.data.address = at;
		info.pixel_format = packed ? vk::Format::eR8G8B8A8Unorm : vk::Format::eR32Uint;
		info.guest_format = packed ? Prospero::BufferFormat::k8_8_8_8UNorm : Prospero::BufferFormat::k32UInt;
		info.extent = vk::Extent3D {ImageWords, 1, 1};
		info.pitch = ImageWords;
		info.UpdateSize();
		return TextureCacheTestAccess::Create(textures, info);
	};
	const auto write = [&](ImageId id) {
		auto& image = textures.GetImage(id);
		const auto native = scheduler.Current().Handle();
		image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, native);
		const bool packed = image.info.pixel_format == vk::Format::eR8G8B8A8Unorm;
		const vk::ClearColorValue clear =
		    packed ? vk::ClearColorValue {.float32 = {std::array<float, 4> {1.0f, 0.0f, 0.0f, 1.0f}}}
		           : vk::ClearColorValue {.uint32 = {std::array<uint32_t, 4> {ImageValue, 0, 0, 0}}};
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		native.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &clear, 1, &range);
		textures.MarkGpuWritten(id);
	};

	const auto first = create(address, false);
	const auto second = create(address, true);
	const auto size = textures.GetImage(first).info.data.size;
	const bool equal = textures.GetImage(second).info.data.size == size;
	write(first);
	write(second);
	const auto later_registered = textures.FindImageFromRange(address, size);

	const auto reversed_first = create(address + PairStride, false);
	const auto reversed_second = create(address + PairStride, true);
	write(reversed_second);
	write(reversed_first);
	const auto earlier_registered = textures.FindImageFromRange(address + PairStride, size);

	const auto unwritten_first = create(address + 2 * PairStride, false);
	(void)create(address + 2 * PairStride, true);
	const auto unwritten = textures.FindImageFromRange(address + 2 * PairStride, size, false);

	const auto last = scheduler.CurrentTick();
	scheduler.Wait(last);
	scheduler.WaitPriorityOperations(last);
	if (!equal) return "the two images at one address differ in data size";
	if (!later_registered || !earlier_registered) return "the range read-back found no GPU-written image";
	if (later_registered != second || earlier_registered != reversed_first) {
		return "the range read-back did not pick the image the GPU wrote last";
	}
	if (unwritten != unwritten_first) return "the unwritten pair did not keep the first match";
	return nullptr;
}

}

extern "C" int RunKytyImageWriteBackAliasTest() {
	Common::InitializeThreads();
	Config::Initialize();
	Config::Load(Config::ConfigOptions {});
	KytyShimTests::TestVulkanDevice gpu;
	auto& graphics = gpu.graphics;
	if (const auto* unavailable = gpu.Initialize(true)) {
		const bool required = std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr;
		std::fprintf(stderr, "kyty_shim_image_write_back_alias: %s, no usable Vulkan device (%s)\n",
		             required ? "FAILED" : "skipped", unavailable);
		return required ? 1 : 77;
	}
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
	auto failure = CheckWriteBackOrder(graphics, address);
	if (failure == nullptr) {
		failure = CheckWriteBackRefreshesBufferCopy(graphics, address + GuestBytes / 4);
	}
	if (failure == nullptr) {
		const auto* linear = CheckPartialBufferCopy(graphics, address + GuestBytes / 2,
		                                           Prospero::TileMode::kLinear);
		const auto* tiled = CheckPartialBufferCopy(graphics, address + GuestBytes / 2,
		                                          Prospero::TileMode::kStandard4KB);
		failure = linear != nullptr ? linear : tiled;
	}
	if (failure == nullptr) {
		failure = CheckRangePicksLatestWrite(graphics, address + GuestBytes / 8 * 7);
	}
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
