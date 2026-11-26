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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

	static void CommitGpuWrite(TextureCache& cache, ImageId image) {
		Common::ScopedLock lock {cache.m_lock};
		cache.CommitGpuWrite(cache.m_slot_images[image]);
	}

	static bool Watched(TextureCache& cache, uint64_t address) {
		return cache.m_page_manager.IsWatched(address);
	}

	static std::array<bool, 2> ApplyPendingClears(TextureCache& cache, const ImageInfo& info) {
		Common::ScopedLock         lock {cache.m_lock};
		TextureCache::MetaDataInfo meta {};
		cache.ApplyPendingMetaClears(info, meta);
		return {meta.clear_mask.test(0), meta.clear_mask.test(1)};
	}
};

}

namespace {

using namespace Libs::Graphics;

constexpr uint64_t GuestBytes = 0x20000;
constexpr int CpuReadWrite = 0x03;
constexpr int GpuReadWrite = 0x30;
constexpr int MapFixed = 0x10;
constexpr uint32_t ImageWords = 64;
constexpr uint32_t CopyWord = 9;
constexpr uint32_t FillWord = 21;
constexpr uint32_t WriteWords = 4;
constexpr uint32_t ImageValue = 0x19283746;
constexpr uint32_t CopyValue = 0xdecafbad;
constexpr uint32_t FillValue = 0x1234fedc;
constexpr uint32_t CpuValue = 0x5aa5c33c;
constexpr uint64_t MultisampleGuestBytes = 0x100000;
constexpr uint64_t MultisampleAlignment = 0x10000;

constexpr uint64_t DepthGuestBytes = 0x100000;

const char* CheckFilledMetaClearReachesBind(GraphicContext& graphics, uint64_t base) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint32_t Width = 64;
	constexpr uint32_t Height = 64;
	constexpr uint32_t Layers = 2;
	constexpr uint64_t SliceBytes = 0x400;
	constexpr uint64_t MetaBytes = SliceBytes * Layers;
	constexpr uint64_t Stride = 0x8000;
	constexpr uint64_t MetaOffset = 0x6000;
	constexpr uint64_t SourceOffset = 0x7000;
	constexpr uint64_t WriteBytes = 16;
	constexpr uint32_t Cases = 28;
	static_assert(Stride * Cases <= DepthGuestBytes);
	constexpr uint32_t Expanded = 0xffffffffu;
	constexpr uint32_t StencilCleared = 0xf0u;
	constexpr uint32_t StencilKept = 0x100u;
	const std::vector<uint32_t> backing(DepthGuestBytes / sizeof(uint32_t), Expanded);
	if (!Libs::LibKernel::Memory::TryWriteBacking(base, backing.data(), DepthGuestBytes)) {
		return "could not write the depth backing";
	}
	uint64_t next = base;
	const auto bind = [&](const auto& between) {
		const auto depth_address = next;
		const auto meta_address = next + MetaOffset;
		next += Stride;
		between(meta_address, depth_address + SourceOffset);
		ImageInfo info;
		info.data.address = depth_address;
		info.pixel_format = vk::Format::eD16Unorm;
		info.guest_format = Prospero::BufferFormat::k16UNorm;
		info.extent = vk::Extent3D {Width, Height, 1};
		info.pitch = Width;
		info.bytes_per_block = sizeof(uint16_t);
		info.resources = {1, Layers};
		info.UpdateSize();
		info.metadata.kind = ImageMetadataKind::Htile;
		info.metadata.range = GuestRange {meta_address, MetaBytes};
		info.htile_initial_clear = false;
		const auto id = TextureCacheTestAccess::Create(textures, info);
		TextureCache::ImageDesc desc {.info = info};
		desc.type = TextureCache::BindingType::DepthTarget;
		desc.view_info.format = info.pixel_format;
		desc.view_info.type = vk::ImageViewType::e2DArray;
		desc.view_info.aspect = vk::ImageAspectFlagBits::eDepth;
		desc.view_info.base_level = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer = 0;
		desc.view_info.layer_count = Layers;
		desc.view_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
		(void)textures.FindDepthTarget(id, desc, nullptr);
		return std::array<bool, 2> {textures.IsMetaCleared(meta_address, 0), textures.IsMetaCleared(meta_address, 1)};
	};
	const auto fill = [&](uint64_t address, uint64_t size, uint32_t value) {
		buffers.FillBuffer(address, size, value, false);
	};
	const auto gpu_write = [&](uint64_t address, uint64_t size) {
		(void)buffers.ObtainBuffer(address, size, true, false);
	};
	const std::array<uint32_t, 4> overwrite {1, 2, 3, 4};
	const auto none = std::array<bool, 2> {false, false};
	const auto both = std::array<bool, 2> {true, true};
	const auto first = std::array<bool, 2> {true, false};
	const auto second = std::array<bool, 2> {false, true};
	const char* failure = nullptr;
	const auto expect = [&failure](std::array<bool, 2> actual, std::array<bool, 2> wanted, const char* message) {
		if (failure == nullptr && actual != wanted) failure = message;
	};
	expect(bind([&](uint64_t meta, uint64_t) { fill(meta, MetaBytes, 0); }), both,
	       "a fill that clears the HTile of both layers before the bind did not clear them");
	expect(bind([&](uint64_t meta, uint64_t) { fill(meta, SliceBytes, 0); }), first,
	       "a fill of the first layer's HTile did not clear exactly that layer");
	expect(bind([&](uint64_t meta, uint64_t) { fill(meta + SliceBytes, SliceBytes, 0); }), second,
	       "a fill of the second layer's HTile did not clear exactly that layer");
	expect(bind([&](uint64_t meta, uint64_t) { fill(meta, MetaBytes, Expanded); }), none,
	       "a fill that does not clear the HTile cleared it");
	expect(bind([&](uint64_t meta, uint64_t) {
		       fill(meta, MetaBytes, 0);
		       fill(meta + SliceBytes, WriteBytes, Expanded);
	       }),
	       none, "a clear overwritten by a DMA fill before the bind was kept");
	expect(bind([&](uint64_t meta, uint64_t) {
		       fill(meta, MetaBytes, 0);
		       std::memcpy(reinterpret_cast<void*>(meta), overwrite.data(), sizeof(overwrite));
	       }),
	       second, "a clear overwritten by a host store before the bind was kept for the overwritten layer");
	expect(bind([&](uint64_t meta, uint64_t) {
		       gpu_write(meta, MetaBytes);
		       fill(meta, MetaBytes, 0);
	       }),
	       both, "a GPU-side fill that clears the HTile before the bind did not clear it");
	expect(bind([&](uint64_t meta, uint64_t) {
		       gpu_write(meta, MetaBytes);
		       fill(meta, MetaBytes, 0);
		       gpu_write(meta + SliceBytes, WriteBytes);
	       }),
	       none, "a clear overwritten by a written buffer binding before the bind was kept");
	expect(bind([&](uint64_t meta, uint64_t source) {
		       gpu_write(meta, MetaBytes);
		       fill(meta, MetaBytes, 0);
		       buffers.CopyBuffer(meta + SliceBytes, source, WriteBytes, false, false);
	       }),
	       none, "a clear overwritten by a DMA copy before the bind was kept");
	expect(bind([&](uint64_t meta, uint64_t) {
		       fill(meta, MetaBytes, 0);
		       textures.RunGarbageCollector();
	       }),
	       both, "a clear from the previous GC tick was not kept");
	expect(bind([&](uint64_t meta, uint64_t) {
		       fill(meta, MetaBytes, 0);
		       textures.RunGarbageCollector();
		       textures.RunGarbageCollector();
	       }),
	       none, "a clear older than one GC tick was kept");
	expect(bind([&](uint64_t meta, uint64_t) {
		       fill(meta, MetaBytes, 0);
		       textures.UnmapMemory(meta, MetaBytes);
	       }),
	       none, "a clear in memory unmapped before the bind was kept");
	const auto stencil = [&](uint32_t value) {
		const auto meta_address = next + MetaOffset;
		next += Stride;
		fill(meta_address, MetaBytes, value);
		ImageInfo info;
		info.resources = {1, Layers};
		info.metadata.kind = ImageMetadataKind::Htile;
		info.metadata.range = GuestRange {meta_address, MetaBytes};
		info.metadata.stencil_compressed = true;
		return TextureCacheTestAccess::ApplyPendingClears(textures, info);
	};
	expect(stencil(StencilCleared), both,
	       "a fill that clears depth and stencil HTile did not clear stencil-compressed metadata");
	expect(stencil(StencilKept), none, "a fill that keeps stencil memory cleared stencil-compressed metadata");
	const auto pending = [&](uint64_t meta) {
		ImageInfo info;
		info.resources = {1, Layers};
		info.metadata.kind = ImageMetadataKind::Htile;
		info.metadata.range = GuestRange {meta, MetaBytes};
		return TextureCacheTestAccess::ApplyPendingClears(textures, info);
	};
	const auto slot = [&]() {
		const auto meta = next + MetaOffset;
		next += Stride;
		return meta;
	};
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		expect(pending(meta), both, "a record noted without a fill in memory that clears the HTile did not clear it");
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		ImageInfo info;
		info.data.address = meta;
		info.pixel_format = vk::Format::eR32Uint;
		info.guest_format = Prospero::BufferFormat::k32UInt;
		info.extent = vk::Extent3D {static_cast<uint32_t>(MetaBytes / sizeof(uint32_t)), 1, 1};
		info.pitch = static_cast<uint32_t>(MetaBytes / sizeof(uint32_t));
		info.UpdateSize();
		TextureCacheTestAccess::CommitGpuWrite(textures, TextureCacheTestAccess::Create(textures, info));
		expect(pending(meta), none, "a clear overwritten by an image write before the bind was kept");
	}
	{
		const auto meta = next + MetaOffset;
		(void)bind([](uint64_t, uint64_t) {});
		textures.NoteMetaFill(meta - SliceBytes, MetaBytes + SliceBytes, 0, false);
		expect(pending(meta), none, "a fill that overlaps a live metadata entry without starting at it was recorded");
		textures.NoteMetaFill(meta + SliceBytes, SliceBytes, 0, false);
		expect(pending(meta), none, "a fill inside a live metadata entry was recorded");
	}
	const auto watched = [&](uint64_t meta) { return TextureCacheTestAccess::Watched(textures, meta); };
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		const bool held = watched(meta);
		expect(pending(meta), both, "a replaced record of a fill on the GPU did not clear the HTile");
		expect({held, watched(meta)}, first,
		       "a record of a fill on the GPU did not write-protect its range exactly until it was applied");
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, true);
		expect({watched(meta), false}, none, "a record of a fill done in memory write-protected its range");
		(void)pending(meta);
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		(void)textures.InvalidateMemory(meta + MetaBytes, sizeof(uint32_t));
		expect({watched(meta), false}, none, "a write elsewhere on its page did not release a record's protection");
		expect(pending(meta), none, "a record of a fill on the GPU was kept after a write elsewhere on its page");
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		textures.RunGarbageCollector();
		textures.RunGarbageCollector();
		expect({watched(meta), false}, none, "a record aged out by the GC kept its protection");
		expect(pending(meta), none, "a record of a fill on the GPU older than one GC tick was kept");
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		textures.UnmapMemory(meta, MetaBytes);
		expect({watched(meta), false}, none, "a record dropped by an unmap kept its protection");
		expect(pending(meta), none, "a record of a fill on the GPU was kept after an unmap");
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		textures.NoteMetadataWrite(meta + SliceBytes, WriteBytes);
		expect({watched(meta), false}, none, "a record dropped by a written binding kept its protection");
		expect(pending(meta), none, "a record of a fill on the GPU was kept after a written binding over it");
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		(void)renderer.HandleFault(PageFaultAccess::Write, meta + MetaBytes);
		expect({watched(meta), false}, none, "a write fault elsewhere on its page did not release a record's protection");
		expect(pending(meta), none, "a record of a fill on the GPU was kept after a write fault on its page");
	}
	{
		const auto meta = slot();
		textures.NoteMetaFill(meta, SliceBytes, 0, false);
		textures.NoteMetaFill(meta + SliceBytes, SliceBytes, 0, false);
		ImageInfo info;
		info.resources = {1, 1};
		info.metadata.kind = ImageMetadataKind::Htile;
		info.metadata.range = GuestRange {meta, SliceBytes};
		const bool applied = TextureCacheTestAccess::ApplyPendingClears(textures, info)[0];
		const bool held = watched(meta);
		textures.NoteMetadataWrite(meta + SliceBytes, WriteBytes);
		expect({applied, held}, both, "two records of fills on the GPU on one page did not hold it until both were gone");
		expect({watched(meta), false}, none, "two records of fills on the GPU on one page left it protected");
	}
	{
		constexpr uint32_t Records = 300;
		const auto meta = slot();
		for (uint32_t record = 0; record < Records; ++record) {
			textures.NoteMetaFill(meta + record * sizeof(uint32_t), sizeof(uint32_t), 0, false);
		}
		const bool held = watched(meta);
		textures.NoteMetadataWrite(meta, Records * sizeof(uint32_t));
		expect({held, watched(meta)}, first, "300 records of fills on the GPU on one page did not hold it exactly until dropped");
	}
	{
		constexpr uint64_t SpanBytes = 0x2000;
		const auto middle = slot();
		const auto start = middle - 2 * SliceBytes;
		const auto tail = start + SpanBytes;
		textures.NoteMetaFill(start, SpanBytes, 0, false);
		textures.NoteMetaFill(tail, SliceBytes, 0, false);
		expect({watched(start), watched(middle)}, both,
		       "a record of a fill on the GPU over three pages did not protect its first and middle pages");
		expect({watched(tail), false}, first, "a record of a fill on the GPU over three pages did not protect its last page");
		textures.NoteMetadataWrite(start, WriteBytes);
		expect({watched(start), watched(middle)}, none,
		       "a dropped record over three pages left its first or middle page protected");
		expect({watched(tail), false}, first,
		       "a dropped record over three pages released its last page while another record held it");
		textures.NoteMetadataWrite(tail, WriteBytes);
		expect({watched(tail), false}, none, "the last record on a shared page left it protected after it was dropped");
	}
	expect(bind([&](uint64_t meta, uint64_t) {
		       fill(meta, SliceBytes, 0);
		       fill(meta + SliceBytes, SliceBytes, 0);
	       }),
	       both, "fills in memory of both layers' HTile on one page did not clear both layers");
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	BufferCacheTestAccess::DiscardGpuWrites(buffers, base, DepthGuestBytes);
	(void)buffers.DropFreedRange(base, DepthGuestBytes, nullptr, nullptr);
	return failure;
}

const char* CheckSharedBackingFill(GraphicContext& graphics) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint64_t Bytes = 0x10000;
	constexpr uint64_t SliceBytes = 0x400;
	constexpr uint64_t MetaBytes = SliceBytes * 2;
	const auto direct_size = static_cast<int64_t>(sceKernelGetDirectMemorySize());
	int64_t phys = -1;
	int64_t other = -1;
	if (sceKernelAllocateDirectMemory(0, direct_size, Bytes, 0, 0, &phys) != 0 || phys < 0 ||
	    sceKernelAllocateDirectMemory(0, direct_size, Bytes, 0, 0, &other) != 0 || other < 0) {
		return "could not allocate direct memory for the shared HTile";
	}
	void* mapped = nullptr;
	if (sceKernelMapDirectMemory(&mapped, Bytes, CpuReadWrite | GpuReadWrite, 0, phys, 0) != 0 || mapped == nullptr) {
		return "could not map the shared HTile memory";
	}
	const auto meta = reinterpret_cast<uint64_t>(mapped);
	const auto none = std::array<bool, 2> {false, false};
	const auto both = std::array<bool, 2> {true, true};
	const char* failure = nullptr;
	const auto expect = [&failure](std::array<bool, 2> actual, std::array<bool, 2> wanted, const char* message) {
		if (failure == nullptr && actual != wanted) failure = message;
	};
	const auto pending = [&]() {
		ImageInfo info;
		info.resources = {1, 2};
		info.metadata.kind = ImageMetadataKind::Htile;
		info.metadata.range = GuestRange {meta, MetaBytes};
		return TextureCacheTestAccess::ApplyPendingClears(textures, info);
	};
	const auto watched = [&]() { return TextureCacheTestAccess::Watched(textures, meta); };
	uint64_t alias = 0;
	const auto map_alias = [&]() {
		void* second = nullptr;
		if (sceKernelMapDirectMemory(&second, Bytes, CpuReadWrite | GpuReadWrite, 0, phys, 0) != 0 || second == nullptr) {
			failure = failure != nullptr ? failure : "could not map the HTile memory a second time";
			return false;
		}
		alias = reinterpret_cast<uint64_t>(second);
		return true;
	};
	const auto unmap_alias = [&]() {
		if (sceKernelMunmap(alias, Bytes) != 0 && failure == nullptr) failure = "could not unmap the second mapping";
	};

	textures.NoteMetaFill(meta, MetaBytes, 0, false);
	expect(pending(), both, "a record of a fill on the GPU over memory mapped once did not clear the HTile");
	if (map_alias()) {
		textures.NoteMetaFill(meta, MetaBytes, 0, false);
		expect({watched(), false}, none, "a declined record of a fill on the GPU over memory mapped twice write-protected it");
		expect(pending(), none, "a record of a fill on the GPU over memory mapped twice was kept");
		unmap_alias();
	}
	textures.NoteMetaFill(meta, MetaBytes, 0, false);
	if (map_alias()) {
		expect(pending(), none, "a record of a fill on the GPU was applied after a second mapping of its memory appeared");
		unmap_alias();
	}
	textures.NoteMetaFill(meta, MetaBytes, 0, false);
	if (map_alias()) {
		unmap_alias();
		expect(pending(), none, "a record of a fill on the GPU was applied after a second mapping of its memory came and went");
	}
	std::memset(mapped, 0, MetaBytes);
	textures.NoteMetaFill(meta, MetaBytes, 0, true);
	expect(pending(), both, "a record of a fill done in memory over memory mapped once did not clear the HTile");
	if (map_alias()) {
		textures.NoteMetaFill(meta, MetaBytes, 0, true);
		expect(pending(), none, "a record of a fill done in memory over memory mapped twice was kept");
		unmap_alias();
	}
	textures.NoteMetaFill(meta, MetaBytes, 0, false);
	void* remapped = mapped;
	if (sceKernelMapDirectMemory(&remapped, Bytes, CpuReadWrite | GpuReadWrite, MapFixed, other, 0) != 0 || remapped != mapped) {
		failure = failure != nullptr ? failure : "could not map other memory at the HTile address";
	} else {
		expect(pending(), none, "a record of a fill on the GPU was applied after its address was mapped to other memory");
	}
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	if (sceKernelMunmap(meta, Bytes) != 0 && failure == nullptr) failure = "could not unmap the HTile memory";
	if ((sceKernelReleaseDirectMemory(phys, Bytes) != 0 || sceKernelReleaseDirectMemory(other, Bytes) != 0) && failure == nullptr) {
		failure = "could not release the HTile memory";
	}
	return failure;
}

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

struct LayerFormat {
	vk::Format             host;
	Prospero::BufferFormat guest;
};

const char* CheckLargerRequestOverGpuWrites(GraphicContext& graphics, uint64_t address,
                                            LayerFormat cached_format, LayerFormat requested_format,
                                            uint32_t requested_levels) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint32_t Width = 64;
	constexpr uint32_t Height = 4;
	constexpr uint32_t CachedLayers = 4;
	constexpr uint32_t RequestedLayers = 6;
	constexpr uint32_t CachedWords = CachedLayers * Width * Height;
	constexpr uint32_t RequestedWords = RequestedLayers * Width * Height;
	const auto layered = [address](LayerFormat format, uint32_t levels, uint32_t layers) {
		ImageInfo info;
		info.data.address = address;
		info.pixel_format = format.host;
		info.guest_format = format.guest;
		info.extent = vk::Extent3D {Width, Height, 1};
		info.pitch = Width;
		info.resources = {levels, layers};
		info.UpdateSize();
		return info;
	};
	const auto cached_info = layered(cached_format, 1, CachedLayers);
	const auto requested_info = layered(requested_format, requested_levels, RequestedLayers);
	const bool keeps = requested_levels == 1;
	if (requested_info.data.size > GuestBytes / 8) return "the layered image does not fit its guest range";
	const std::vector<uint32_t> guest(requested_info.data.size / sizeof(uint32_t), FillValue);
	if (!Libs::LibKernel::Memory::TryWriteBacking(address, guest.data(), requested_info.data.size)) {
		return "could not write the guest backing";
	}

	std::vector<uint32_t> written(CachedWords);
	for (uint32_t i = 0; i < CachedWords; ++i) {
		written[i] = ImageValue + i;
	}
	const std::array cached_copies {vk::BufferImageCopy {
	    0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, CachedLayers}, {}, cached_info.extent}};
	const std::array requested_copies {vk::BufferImageCopy {
	    0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, RequestedLayers}, {}, requested_info.extent}};
	Buffer initial(graphics, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
	               CachedWords * sizeof(uint32_t));
	std::memcpy(initial.Mapped().data(), written.data(), CachedWords * sizeof(uint32_t));
	initial.Flush(0, CachedWords * sizeof(uint32_t));
	const auto cached = TextureCacheTestAccess::Create(textures, cached_info);
	textures.GetImage(cached).Upload(cached_copies, initial.Handle(), 0, CachedWords * sizeof(uint32_t));
	textures.MarkGpuWritten(cached);

	TextureCache::ImageDesc desc {.info = requested_info};
	desc.view_info.format = requested_info.pixel_format;
	const auto found = textures.FindImage(desc);
	Buffer readback(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	                RequestedWords * sizeof(uint32_t));
	bool kept_gpu_write = false;
	uint32_t found_layers = 0;
	if (found) {
		textures.UpdateImage(found);
		auto& image = textures.GetImage(found);
		kept_gpu_write = image.IsGpuModified();
		found_layers = image.info.resources.layers;
		image.Download(requested_copies, readback.Handle(), 0, RequestedWords * sizeof(uint32_t));
	}
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	(void)buffers.DropFreedRange(address, requested_info.data.size, nullptr, nullptr);
	if (found_layers != RequestedLayers) return "the larger request found no image with its layers";
	readback.Invalidate(0, RequestedWords * sizeof(uint32_t));
	std::vector<uint32_t> actual(RequestedWords);
	std::memcpy(actual.data(), readback.Mapped().data(), RequestedWords * sizeof(uint32_t));
	for (uint32_t i = 0; i < CachedWords; ++i) {
		if (actual[i] != (keeps ? written[i] : FillValue)) {
			return keeps ? "a request for more layers dropped the GPU-written layers"
			             : "an image with fewer levels was copied into the request's layers";
		}
	}
	for (uint32_t i = CachedWords; i < RequestedWords; ++i) {
		if (actual[i] != FillValue) return "the added layers were not loaded from guest memory";
	}
	if (kept_gpu_write != keeps) {
		return keeps ? "the image that kept the GPU-written layers is not marked GPU-written"
		             : "the image loaded from guest memory is marked GPU-written";
	}
	return nullptr;
}

const char* CheckLargerRequestOverInvalidatedWrites(GraphicContext& graphics, uint64_t address, uint64_t limit,
                                                    LayerFormat cached_format, LayerFormat requested_format,
                                                    uint32_t samples) {
	RenderContext renderer(graphics);
	CommandProcessor producer(renderer, 0);
	producer.Reset();
	auto& scheduler = renderer.GetCommandScheduler();
	auto& textures = renderer.GetTextureCache();
	auto& buffers = renderer.GetBufferCache();
	scheduler.Begin(producer.GetCtx(), producer.GetUcfg(), producer.GetShCtx());

	constexpr uint32_t Width = 32;
	constexpr uint32_t Height = 4;
	constexpr uint32_t CachedLayers = 4;
	constexpr uint32_t RequestedLayers = 6;
	constexpr uint32_t CachedWords = CachedLayers * Width * Height;
	constexpr uint32_t RequestedWords = RequestedLayers * Width * Height;
	const auto layered = [address, samples](LayerFormat format, uint32_t layers) {
		ImageInfo info;
		info.data.address = address;
		info.pixel_format = format.host;
		info.guest_format = format.guest;
		info.extent = vk::Extent3D {Width, Height, 1};
		info.tile_mode = samples == 1 ? Prospero::TileMode::kLinear : Prospero::TileMode::kRenderTarget;
		info.pitch = samples == 1 ? Width : 0;
		info.samples = samples;
		info.resources = {1, layers};
		info.UpdateSize();
		return info;
	};
	const auto cached_info = layered(cached_format, CachedLayers);
	const auto requested_info = layered(requested_format, RequestedLayers);
	if (requested_info.data.size > limit) return "the layered image does not fit its guest range";
	std::vector<uint32_t> guest(requested_info.data.size / sizeof(uint32_t), FillValue);
	if (!Libs::LibKernel::Memory::TryWriteBacking(address, guest.data(), requested_info.data.size)) {
		return "could not write the guest backing";
	}

	const auto cached = TextureCacheTestAccess::Create(textures, cached_info);
	{
		auto& image = textures.GetImage(cached);
		const auto native = scheduler.Current().Handle();
		image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, native);
		const vk::ClearColorValue clear {.float32 = {std::array<float, 4> {1.0f, 0.5f, 0.25f, 1.0f}}};
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, CachedLayers};
		native.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &clear, 1, &range);
	}
	textures.MarkGpuWritten(cached);
	std::fill(guest.begin(), guest.begin() + cached_info.data.size / sizeof(uint32_t), CpuValue);
	if (!Libs::LibKernel::Memory::TryWriteBacking(address, guest.data(), cached_info.data.size)) {
		return "could not write the guest backing";
	}
	(void)textures.InvalidateMemory(address, cached_info.data.size);
	const auto& invalidated = textures.GetImage(cached);
	const bool stale = invalidated.IsGpuModified() && invalidated.IsDefinitelyCpuDirty();

	TextureCache::ImageDesc desc {.info = requested_info};
	desc.view_info.format = requested_info.pixel_format;
	const auto found = textures.FindImage(desc);
	const std::array requested_copies {vk::BufferImageCopy {
	    0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, RequestedLayers}, {}, requested_info.extent}};
	Buffer readback(graphics, scheduler, MemoryUsage::Download, 0, vk::BufferUsageFlagBits::eTransferDst,
	                RequestedWords * sizeof(uint32_t));
	uint32_t found_layers = 0;
	if (found) {
		auto& image = textures.GetImage(found);
		found_layers = image.info.resources.layers;
		if (samples == 1) {
			textures.UpdateImage(found);
			image.Download(requested_copies, readback.Handle(), 0, RequestedWords * sizeof(uint32_t));
		}
	}
	const auto tick = scheduler.CurrentTick();
	scheduler.Wait(tick);
	scheduler.WaitPriorityOperations(tick);
	(void)buffers.DropFreedRange(address, requested_info.data.size, nullptr, nullptr);
	if (!stale) return "the guest write did not invalidate the GPU-written image";
	if (found_layers != RequestedLayers) return "the larger request found no image with its layers";
	if (samples != 1) return nullptr;
	readback.Invalidate(0, RequestedWords * sizeof(uint32_t));
	std::vector<uint32_t> actual(RequestedWords);
	std::memcpy(actual.data(), readback.Mapped().data(), RequestedWords * sizeof(uint32_t));
	for (uint32_t i = 0; i < CachedWords; ++i) {
		if (actual[i] != CpuValue) return "a request for more layers kept GPU writes over a newer guest write";
	}
	for (uint32_t i = CachedWords; i < RequestedWords; ++i) {
		if (actual[i] != FillValue) return "the added layers were not loaded from guest memory";
	}
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
	constexpr LayerFormat Words {vk::Format::eR32Uint, Prospero::BufferFormat::k32UInt};
	constexpr LayerFormat Bytes {vk::Format::eR8G8B8A8Unorm, Prospero::BufferFormat::k8_8_8_8UNorm};
	constexpr LayerFormat SwappedBytes {vk::Format::eB8G8R8A8Unorm, Prospero::BufferFormat::k8_8_8_8UNorm};
	if (failure == nullptr) {
		failure = CheckLargerRequestOverGpuWrites(graphics, address + GuestBytes / 8 * 5, Words, Bytes, 1);
	}
	if (failure == nullptr) {
		failure = CheckLargerRequestOverGpuWrites(graphics, address + GuestBytes / 8 * 6, SwappedBytes, Bytes, 1);
	}
	if (failure == nullptr) {
		failure = CheckLargerRequestOverGpuWrites(graphics, address + GuestBytes / 8 * 5, Words, Bytes, 2);
	}
	if (failure == nullptr) {
		failure = CheckLargerRequestOverInvalidatedWrites(graphics, address + GuestBytes / 8 * 5, GuestBytes / 8,
		                                                  SwappedBytes, Bytes, 1);
	}
	if (failure == nullptr) {
		constexpr uint64_t Mapped = MultisampleGuestBytes + MultisampleAlignment;
		void* multisample_guest = nullptr;
		if (sceKernelMapFlexibleMemory(&multisample_guest, Mapped, CpuReadWrite, 0) != 0 ||
		    multisample_guest == nullptr) {
			failure = "could not map the multisample backing";
		} else {
			const auto mapped = reinterpret_cast<uint64_t>(multisample_guest);
			const auto aligned = (mapped + MultisampleAlignment - 1) & ~(MultisampleAlignment - 1);
			failure = CheckLargerRequestOverInvalidatedWrites(graphics, aligned, MultisampleGuestBytes,
			                                                  SwappedBytes, Bytes, 4);
			if (sceKernelMunmap(mapped, Mapped) != 0 && failure == nullptr) {
				failure = "could not unmap the multisample backing";
			}
		}
	}
	if (failure == nullptr) {
		failure = CheckRangePicksLatestWrite(graphics, address + GuestBytes / 8 * 7);
	}
	if (failure == nullptr) {
		void* depth_guest = nullptr;
		if (sceKernelMapFlexibleMemory(&depth_guest, DepthGuestBytes, CpuReadWrite, 0) != 0 || depth_guest == nullptr) {
			failure = "could not map the depth backing";
		} else {
			const auto depth_address = reinterpret_cast<uint64_t>(depth_guest);
			failure = CheckFilledMetaClearReachesBind(graphics, depth_address);
			if (sceKernelMunmap(depth_address, DepthGuestBytes) != 0 && failure == nullptr) {
				failure = "could not unmap the depth backing";
			}
		}
	}
	if (failure == nullptr) {
		failure = CheckSharedBackingFill(graphics);
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
