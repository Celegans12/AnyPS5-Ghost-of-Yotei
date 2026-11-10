#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/liveAb.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/syncTimeline.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

namespace {

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

[[nodiscard]] VmaAllocationCreateFlags AllocationFlags(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::Upload:
		case MemoryUsage::Stream:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT |
			       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		case MemoryUsage::Download:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		case MemoryUsage::DeviceLocal: return {};
	}
	return {};
}

[[nodiscard]] VmaMemoryUsage AllocationUsage(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal:
		case MemoryUsage::Stream: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		case MemoryUsage::Upload:
		case MemoryUsage::Download: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

[[nodiscard]] bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
	if (alignment == 0) {
		result = value;
		return true;
	}
	const auto aligned = Common::AlignUp(value, alignment);
	if (aligned < value) {
		return false;
	}
	result = aligned;
	return true;
}

} // namespace

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size, bool prefer_host)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(graphics.allocator == nullptr || size == 0);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;

	const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	// Buffers with a device address keep dedicated allocations. Upstream lets VMA suballocate them (d58aefde4, with
	// 4644c5292 zeroing the BDA tables that suballocated memory no longer guarantees); not taken without a measurement
	// (allocation count, BAR placement and the VRAM budget logic all see the change).
	const VmaAllocationCreateFlags bda_flag =
	    with_bda ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT : 0;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | AllocationFlags(usage);
	// A stream buffer asked for host memory is allocated like an upload buffer (CPU writes only).
	const bool host_stream = prefer_host && usage == MemoryUsage::Stream;
	allocation_info.usage  = host_stream ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	vk::Result        result        = vk::Result::eErrorOutOfDeviceMemory;
	// Guest buffers of 256 MiB and more (the buffer cache's joined spans) take plain device-local memory, not resizable-BAR
	// memory; their CPU reads download through a copy. Ghost of Yotei 2.002 clears 11 MB of particle data every frame in
	// a 523 MiB joined span: while that span sat in BAR memory, the GPU's writes there intermittently ran at bus speed
	// (~95 ms per clear instead of ~0.03 ms; 5-7 fps in the Snake fight after its cutscene, map §3w, 4.11.2026). Measured:
	// that fight 6.6 -> 13.2 fps, the 2.002 gate 14.3 -> 15.0 fps, same image.
	constexpr uint64_t BigBufferBytes = 256ull << 20u;
	if (usage == MemoryUsage::DeviceLocal && cpu_address != 0 && size < BigBufferBytes) {
		// Device-local and host-visible (resizable BAR), persistently mapped; falls back to plain
		// device-local memory when none is left.
		auto bar           = allocation_info;
		bar.flags         |= VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		bar.requiredFlags  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
		result             = static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator,
		                                                             static_cast<const VkBufferCreateInfo*>(buffer_info), &bar,
		                                                             &native_buffer, &m_allocation, &allocation_result));
		static std::atomic_uint fallbacks {0};
		if (result != vk::Result::eSuccess && fallbacks.fetch_add(1) < 8u) {
			std::fprintf(stderr, "DIAG_BAR_READS allocation of 0x%llx bytes fell back to device-local (%d)\n",
			             static_cast<unsigned long long>(size), static_cast<int>(result));
		}
	}
	if (result != vk::Result::eSuccess) {
		result = static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info),
		                                                 &allocation_info, &native_buffer, &m_allocation, &allocation_result));
	}
	if (result != vk::Result::eSuccess && host_stream) {
		// No host memory left within the budget: the old placement of the stream buffers.
		std::fprintf(stderr, "DIAG_STREAM_HOST allocation of 0x%llx bytes fell back to device memory (%d)\n",
		             static_cast<unsigned long long>(size), static_cast<int>(result));
		allocation_info.usage = AllocationUsage(usage);
		result = static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info),
		                                                 &allocation_info, &native_buffer, &m_allocation, &allocation_result));
	}
	// KYTY_VRAM_BUDGET / KYTY_VRAM_GC_FIX: past the driver's budget a buffer the caches need (or the temporary buffer
	// an eviction downloads into) is allocated anyway (the driver pages it) instead of ending the session; the
	// collectors bring the usage back down.
	if (result != vk::Result::eSuccess && (VramBudgetEnabled() || VramGcFixEnabled())) {
		auto over = allocation_info;
		over.flags &= ~static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
		result = static_cast<vk::Result>(vmaCreateBuffer(graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info),
		                                                 &over, &native_buffer, &m_allocation, &allocation_result));
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 8u) {
			std::fprintf(stderr, "VRAM_BUDGET allocation of 0x%llx bytes made past the driver budget (%d)\n",
			             static_cast<unsigned long long>(size), static_cast<int>(result));
		}
	}
	if (result != vk::Result::eSuccess) {
		graphics.LogMemoryBudget();
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	m_buffer = native_buffer;
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
	}

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(graphics.allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
	}
}

// ---- KYTY_PERF_DIRECT_RANGES (A1 stage 2) ---------------------------------------------
namespace {

uint32_t HostPointerTypeBits(GraphicContext& graphics, uint64_t cpu_address) {
	vk::MemoryHostPointerPropertiesEXT properties {};
	const auto result = graphics.device.getMemoryHostPointerPropertiesEXT(
	    vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, reinterpret_cast<const void*>(cpu_address), &properties);
	return result == vk::Result::eSuccess ? properties.memoryTypeBits : 0u;
}
} // namespace

bool Buffer::CanImportGuestMemory(GraphicContext& graphics, uint64_t cpu_address, uint64_t size) {
	// The host pointer and the size must be multiples of minImportedHostPointerAlignment (4 KiB on the test GPU).
	if (size == 0 || (cpu_address & 0xfffu) != 0 || (size & 0xfffu) != 0) return false;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetMemoryHostPointerPropertiesEXT == nullptr) return false;
	return HostPointerTypeBits(graphics, cpu_address) != 0;
}

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, GuestDirectTag tag, uint64_t cpu_address,
               uint64_t size)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(MemoryUsage::DeviceLocal), m_cpu_address(cpu_address),
      m_size(size) {
	// D6 step K4 (GuestDirectTag::failed): a refused import leaves the buffer empty (the destructor then does nothing).
	const auto refuse = [&](bool failed) {
		if (!failed || tag.failed == nullptr) return false;
		if (m_buffer != nullptr) graphics.device.destroyBuffer(m_buffer, nullptr);
		m_buffer      = nullptr;
		*tag.failed   = true;
		return true;
	};
	vk::ExternalMemoryBufferCreateInfo external {};
	external.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
	vk::BufferCreateInfo buffer_info {};
	buffer_info.pNext = &external;
	buffer_info.size  = size;
	buffer_info.usage = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	const bool created = graphics.device.createBuffer(&buffer_info, nullptr, &m_buffer) == vk::Result::eSuccess;
	if (!created) m_buffer = nullptr;
	if (refuse(!created)) return;
	EXIT_NOT_IMPLEMENTED(!created);
	vk::MemoryRequirements requirements {};
	graphics.device.getBufferMemoryRequirements(m_buffer, &requirements);
	const uint32_t types = HostPointerTypeBits(graphics, cpu_address) & requirements.memoryTypeBits;
	if (refuse(types == 0 || requirements.size > size)) return;
	EXIT_NOT_IMPLEMENTED(types == 0 || requirements.size > size);
	vk::ImportMemoryHostPointerInfoEXT import {};
	import.handleType   = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
	import.pHostPointer = reinterpret_cast<void*>(cpu_address);
	vk::MemoryAllocateFlagsInfo flags {};
	flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	flags.pNext = &import;
	vk::MemoryAllocateInfo allocate {};
	allocate.pNext           = &flags;
	allocate.allocationSize  = size;
	allocate.memoryTypeIndex = static_cast<uint32_t>(std::countr_zero(types));
	const auto result        = graphics.device.allocateMemory(&allocate, nullptr, &m_direct_memory);
	if (result != vk::Result::eSuccess) m_direct_memory = nullptr;
	if (refuse(result != vk::Result::eSuccess)) return;
	if (result != vk::Result::eSuccess) {
		std::fprintf(stderr, "PERF_DIRECT_RANGES import of 0x%llx bytes at 0x%llx failed: %d\n",
		             static_cast<unsigned long long>(size), static_cast<unsigned long long>(cpu_address),
		             static_cast<int>(result));
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	EXIT_NOT_IMPLEMENTED(graphics.device.bindBufferMemory(m_buffer, m_direct_memory, 0) != vk::Result::eSuccess);
	vk::BufferDeviceAddressInfo address_info {};
	address_info.buffer = m_buffer;
	m_device_address    = graphics.device.getBufferAddress(address_info);
	EXIT_IF(m_device_address == 0);
	m_coherent = true;
}

Buffer::~Buffer() {
	if (m_direct_memory != nullptr) {
		CommandRecorder::DrainBeforeDestroy(); // (as below)
		m_graphics->device.destroyBuffer(m_buffer, nullptr);
		m_graphics->device.freeMemory(m_direct_memory, nullptr);
		return;
	}
	if (m_buffer != nullptr) {
		// A command in the ring may still name this buffer (VMA does not go through the
		// dispatcher's destroy functions).
		CommandRecorder::DrainBeforeDestroy();
		vmaDestroyBuffer(m_graphics->allocator, m_buffer, m_allocation);
	}
}

vk::DeviceAddress Buffer::BufferDeviceAddress() const noexcept {
	EXIT_IF(m_device_address == 0);
	return m_device_address;
}

bool Buffer::IsInBounds(uint64_t address, uint64_t size) const noexcept {
	return address >= m_cpu_address && size <= Size() && address - m_cpu_address <= Size() - size;
}

void Buffer::Flush(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaFlushAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

void Buffer::Invalidate(uint64_t offset, uint64_t size) {
	EXIT_IF(m_usage != MemoryUsage::Download || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

vk::BufferMemoryBarrier Buffer::Barrier(uint64_t offset, uint64_t size, vk::AccessFlags source,
                                        vk::AccessFlags destination) const {
	if (Handle() == nullptr || size == 0 || offset > Size() || size > Size() - offset) {
		EXIT("Buffer: invalid DMA barrier, handle=%p offset=0x%016" PRIx64 " size=0x%016" PRIx64
		     " capacity=0x%016" PRIx64 "\n",
		     static_cast<const void*>(Handle()), offset, size, Size());
	}
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = source;
	barrier.dstAccessMask       = destination;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = Handle();
	barrier.offset              = offset;
	barrier.size                = size;
	return barrier;
}

void Buffer::CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                      uint64_t destination_offset, uint64_t size, vk::AccessFlags source_before,
                      vk::AccessFlags destination_before, vk::AccessFlags source_after,
                      vk::AccessFlags destination_after) {
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, source_before, vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, destination_before, vk::AccessFlagBits::eTransferWrite),
	};
	const auto host_access  = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite;
	auto       before_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_before | destination_before) & host_access)) {
		before_stage |= vk::PipelineStageFlagBits::eHost;
	}
	const auto native = command.Handle();
	native.pipelineBarrier(before_stage, vk::PipelineStageFlagBits::eTransfer,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	GpuCheckpoints::Mark(native, 0xFEED000000000000ull | (3ull << 16u) | 184u);
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, source_after),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, destination_after),
	};
	auto after_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_after | destination_after) & host_access)) {
		after_stage |= vk::PipelineStageFlagBits::eHost;
	}
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, after_stage,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	command.EndRendering();
	const auto before =
	    Barrier(offset, size, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	            vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Handle();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	GpuCheckpoints::Mark(native, 0xFEED000000000000ull | (3ull << 16u) | 210u);
	native.fillBuffer(Handle(), offset, size, value);
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite,
	                           vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size, bool prefer_host)
    : Buffer(graphics, scheduler, usage, 0, AllFlags, size, prefer_host),
      m_current_watches(WATCHES_INITIAL_RESERVE), m_previous_watches(WATCHES_INITIAL_RESERVE) {}

bool StreamBuffer::NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
                                        uint64_t& alignment) {
	if (coherent) {
		return true;
	}
	if (!AlignUp(size, atom, size)) {
		return false;
	}
	const auto divisor = std::gcd(alignment, atom);
	if (alignment != 0 && alignment / divisor > UINT64_MAX / atom) {
		return false;
	}
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	return true;
}

std::pair<uint8_t*, uint64_t> StreamBuffer::Map(uint64_t size, uint64_t alignment,
                                                bool allow_wait) {
	if (Mapped().empty()) {
		return {nullptr, 0};
	}
	uint64_t   mapped_size = size;
	const auto atom        = Graphics().physical_device_properties.limits.nonCoherentAtomSize;
	if (!NormalizeReservation(IsCoherent(), atom, mapped_size, alignment)) {
		return {nullptr, 0};
	}
	if (mapped_size > Size()) {
		return {nullptr, 0};
	}

	uint64_t aligned_offset = 0;
	if (!AlignUp(m_offset, alignment, aligned_offset)) {
		return {nullptr, 0};
	}

	const bool wrap = aligned_offset > Size() - mapped_size;
	if (wrap) {
		aligned_offset = 0;
	}

	auto wait_cursor = wrap ? size_t {0} : m_wait_cursor;
	auto wait_bound  = wrap ? uint64_t {0} : m_wait_bound;
	auto invalidation_mark =
	    wrap ? std::optional<size_t> {m_current_watch_cursor} : m_invalidation_mark;
	auto& pending_watches = wrap ? m_current_watches : m_previous_watches;
	if (!WaitPendingOperations(pending_watches, invalidation_mark, aligned_offset + mapped_size,
	                           allow_wait, wait_cursor, wait_bound)) {
		return {nullptr, 0};
	}

	if (wrap) {
		m_invalidation_mark    = invalidation_mark;
		m_current_watch_cursor = 0;
		std::swap(m_previous_watches, m_current_watches);
	}
	m_wait_cursor = wait_cursor;
	m_wait_bound  = wait_bound;
	m_offset      = aligned_offset;
	m_mapped_size = mapped_size;
	return {Mapped().data() + m_offset, m_offset};
}

void StreamBuffer::Commit() {
	if (Usage() != MemoryUsage::Download && m_mapped_size != 0) {
		Flush(m_offset, m_mapped_size);
	}

	m_offset += m_mapped_size;
	const auto tick = Scheduler().CurrentTick();
	if (m_current_watch_cursor != 0 && m_current_watches[m_current_watch_cursor - 1].tick == tick) {
		m_current_watches[m_current_watch_cursor - 1].upper_bound = m_offset;
		return;
	}
	if (m_current_watch_cursor + 1 >= m_current_watches.size()) {
		m_current_watches.resize(m_current_watches.size() + WATCHES_RESERVE_CHUNK);
	}
	auto& watch       = m_current_watches[m_current_watch_cursor++];
	watch.upper_bound = m_offset;
	watch.tick        = tick;
}

uint64_t StreamBuffer::Copy(const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	const auto [data, offset] = Map(size, alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, source, static_cast<size_t>(size));
	Commit();
	return offset;
}

bool StreamBuffer::WaitPendingOperations(const std::vector<Watch>& watches,
                                         std::optional<size_t>     invalidation_mark,
                                         uint64_t requested_upper_bound, bool allow_wait,
                                         size_t& wait_cursor, uint64_t& wait_bound) {
	if (!invalidation_mark.has_value()) {
		return true;
	}
	while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
		const auto& watch = watches[wait_cursor];
		const bool busy = !Scheduler().IsFree(watch.tick);
		if (busy && !allow_wait) {
			return false;
		}
		SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::StreamWrap); // KYTY_TRACE_SYNC_TIMELINE
		Scheduler().Wait(watch.tick);
		if (Usage() == MemoryUsage::Download) {
			Scheduler().WaitPriorityOperations(watch.tick);
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

} // namespace Libs::Graphics
