#include "graphics/host_gpu/renderer/cache/inPlaceMemory.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <fmt/format.h>

namespace Libs::Graphics {
namespace {

constexpr uint64_t MiB        = 1024 * 1024;
constexpr uint64_t ChunkBytes = 256 * MiB;

double ElapsedMs(std::chrono::steady_clock::time_point started) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}

}

bool InPlaceMemory::Requested() noexcept {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_COMMAND_MEMORY_IN_PLACE");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

bool InPlaceMemory::Initialize(GraphicContext& graphics) {
	if (!graphics.host_import_enabled) {
		::Log::WriteToConsoleAndLog("[kyty-command-memory] off: the device has no VK_EXT_external_memory_host\n");
		return false;
	}
	if (!LibKernel::Memory::GuestViewRange(&m_view, &m_guest_bytes, &m_view_bytes) || m_view == nullptr) {
		::Log::WriteToConsoleAndLog("[kyty-command-memory] off: guest memory has no second view\n");
		return false;
	}
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkGetMemoryHostPointerPropertiesEXT == nullptr) {
		::Log::WriteToConsoleAndLog("[kyty-command-memory] off: vkGetMemoryHostPointerPropertiesEXT is not loaded\n");
		return false;
	}
	const auto alignment = std::max<uint64_t>(graphics.host_import_alignment, 64 * 1024);
	m_chunk_bytes        = std::max<uint64_t>(ChunkBytes / alignment * alignment, alignment);
	const auto device    = static_cast<VkDevice>(graphics.device);
	VkExternalMemoryBufferCreateInfo external {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	info.pNext    = &external;
	info.size     = 64 * 1024;
	info.usage    = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBuffer probe = VK_NULL_HANDLE;
	if (d.vkCreateBuffer(device, &info, nullptr, &probe) != VK_SUCCESS) {
		::Log::WriteToConsoleAndLog("[kyty-command-memory] off: no buffer can take an external host allocation\n");
		return false;
	}
	VkMemoryRequirements requirements {};
	d.vkGetBufferMemoryRequirements(device, probe, &requirements);
	d.vkDestroyBuffer(device, probe, nullptr);
	m_type_bits = requirements.memoryTypeBits;
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-command-memory] on: direct memory of the second view at {} (0x{:x} bytes after a guest file of 0x{:x}), "
	    "imported in chunks of {} MiB, buffer memory types 0x{:x}\n",
	    static_cast<const void*>(m_view), m_view_bytes - m_guest_bytes, m_guest_bytes, m_chunk_bytes / MiB, m_type_bits));
	return true;
}

void InPlaceMemory::Count(Refusal refusal) noexcept {
	switch (refusal) {
		case Refusal::None: break;
		case Refusal::Unbacked: s_refused_unbacked.fetch_add(1, std::memory_order_relaxed); break;
		case Refusal::GuestFile: s_refused_guest.fetch_add(1, std::memory_order_relaxed); break;
		case Refusal::Straddle: s_refused_straddle.fetch_add(1, std::memory_order_relaxed); break;
		case Refusal::Chunk: s_refused_chunk.fetch_add(1, std::memory_order_relaxed); break;
	}
}

InPlaceMemory::Refusal InPlaceMemory::Locate(GraphicContext& graphics, uint64_t vaddr, uint64_t size, uint64_t* offset,
                                             Chunk** chunk, uint64_t* chunk_begin, uint64_t* chunk_end) {
	if (!m_initialized) {
		m_initialized = true;
		m_ready       = Initialize(graphics);
	}
	if (!m_ready) {
		return Refusal::Chunk;
	}
	const auto* backing = static_cast<const uint8_t*>(LibKernel::Memory::FindBackingPointer(vaddr, size));
	if (backing == nullptr || backing < m_view || static_cast<uint64_t>(backing - m_view) > m_view_bytes - size) {
		return Refusal::Unbacked;
	}
	*offset = static_cast<uint64_t>(backing - m_view);
	if (*offset < m_guest_bytes) {
		return Refusal::GuestFile;
	}
	const auto index = *offset / m_chunk_bytes;
	*chunk_begin     = index * m_chunk_bytes;
	*chunk_end       = std::min(*chunk_begin + m_chunk_bytes, m_view_bytes);
	if (*chunk_begin < m_guest_bytes || *offset + size > *chunk_end) {
		return Refusal::Straddle;
	}
	auto& found = m_chunks[index];
	if (found.memory == VK_NULL_HANDLE && !found.failed) {
		MakeChunk(graphics, found, *chunk_begin, *chunk_end - *chunk_begin);
	}
	if (found.memory == VK_NULL_HANDLE) {
		return Refusal::Chunk;
	}
	*chunk = &found;
	return Refusal::None;
}

bool InPlaceMemory::CanImport(GraphicContext& graphics, uint64_t vaddr, uint64_t size) {
	uint64_t   offset = 0, begin = 0, end = 0;
	Chunk*     chunk  = nullptr;
	const auto result = Locate(graphics, vaddr, size, &offset, &chunk, &begin, &end);
	Count(result);
	return result == Refusal::None;
}

std::optional<InPlaceMemory::Imported> InPlaceMemory::Import(GraphicContext& graphics, uint64_t vaddr, uint64_t size,
                                                            vk::BufferUsageFlags usage) {
	uint64_t offset = 0, chunk_begin = 0, chunk_end = 0;
	Chunk*   chunk  = nullptr;
	if (const auto result = Locate(graphics, vaddr, size, &offset, &chunk, &chunk_begin, &chunk_end);
	    result != Refusal::None) {
		Count(result);
		return std::nullopt;
	}
	const auto& d      = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto  device = static_cast<VkDevice>(graphics.device);
	VkExternalMemoryBufferCreateInfo external {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
	external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	info.pNext = &external;
	info.size  = size;
	info.usage = static_cast<VkBufferUsageFlags>(usage);
	const std::array<uint32_t, 2> families {graphics.queue_family, graphics.readback_family};
	if (graphics.readback_family != static_cast<uint32_t>(-1)) {
		info.sharingMode           = VK_SHARING_MODE_CONCURRENT;
		info.queueFamilyIndexCount = 2;
		info.pQueueFamilyIndices   = families.data();
	}
	VkBuffer buffer = VK_NULL_HANDLE;
	if (d.vkCreateBuffer(device, &info, nullptr, &buffer) != VK_SUCCESS) {
		s_refused_bind.fetch_add(1, std::memory_order_relaxed);
		return std::nullopt;
	}
	VkMemoryRequirements requirements {};
	d.vkGetBufferMemoryRequirements(device, buffer, &requirements);
	const auto at = offset - chunk_begin;
	if ((requirements.memoryTypeBits & (1u << chunk->type)) == 0 ||
	    (requirements.alignment != 0 && at % requirements.alignment != 0) ||
	    at + requirements.size > chunk_end - chunk_begin ||
	    d.vkBindBufferMemory(device, buffer, chunk->memory, at) != VK_SUCCESS) {
		d.vkDestroyBuffer(device, buffer, nullptr);
		s_refused_bind.fetch_add(1, std::memory_order_relaxed);
		return std::nullopt;
	}
	VkBufferDeviceAddressInfo address_info {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
	address_info.buffer = buffer;
	const auto address  = d.vkGetBufferDeviceAddress(device, &address_info);
	if (address == 0) {
		d.vkDestroyBuffer(device, buffer, nullptr);
		s_refused_bind.fetch_add(1, std::memory_order_relaxed);
		return std::nullopt;
	}
	s_imported.fetch_add(1, std::memory_order_relaxed);
	s_imported_bytes.fetch_add(size, std::memory_order_relaxed);
	return Imported {vk::Buffer(buffer), address};
}

void InPlaceMemory::MakeChunk(GraphicContext& graphics, Chunk& chunk, uint64_t begin, uint64_t bytes) {
	const auto& d       = VULKAN_HPP_DEFAULT_DISPATCHER;
	const auto  device  = static_cast<VkDevice>(graphics.device);
	void*       host    = const_cast<uint8_t*>(m_view + begin);
	const auto  started = std::chrono::steady_clock::now();
	VkMemoryHostPointerPropertiesEXT properties {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
	auto result = d.vkGetMemoryHostPointerPropertiesEXT(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	                                                    host, &properties);
	const auto types = properties.memoryTypeBits & m_type_bits;
	if (result != VK_SUCCESS || types == 0) {
		chunk.failed = true;
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-command-memory] chunk 0x{:x}+0x{:x}: no memory type (result {}, host types 0x{:x}, buffer types 0x{:x})\n",
		    begin, bytes, static_cast<int>(result), properties.memoryTypeBits, m_type_bits));
		return;
	}
	auto        type              = static_cast<uint32_t>(std::countr_zero(types));
	const auto& memory_properties = graphics.physical_device_memory_properties;
	for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
		if ((types & (1u << i)) == 0) {
			continue;
		}
		const auto flags = static_cast<VkMemoryPropertyFlags>(memory_properties.memoryTypes[i].propertyFlags);
		if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
			type = i;
			if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) {
				break;
			}
		}
	}
	VkImportMemoryHostPointerInfoEXT import {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
	import.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	import.pHostPointer = host;
	VkMemoryAllocateFlagsInfo flags {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
	flags.pNext = &import;
	flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	VkMemoryAllocateInfo allocation {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	allocation.pNext           = &flags;
	allocation.allocationSize  = bytes;
	allocation.memoryTypeIndex = type;
	result                     = d.vkAllocateMemory(device, &allocation, nullptr, &chunk.memory);
	if (result != VK_SUCCESS) {
		chunk.memory = VK_NULL_HANDLE;
		chunk.failed = true;
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-command-memory] chunk 0x{:x}+0x{:x} refused by vkAllocateMemory: {} after {:.1f} ms\n", begin, bytes,
		    static_cast<int>(result), ElapsedMs(started)));
		return;
	}
	chunk.type = type;
	s_chunks.fetch_add(1, std::memory_order_relaxed);
	const auto pinned = s_pinned_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-command-memory] chunk 0x{:x}+0x{:x} of direct memory imported as memory type {} in {:.1f} ms; {} MiB pinned "
	    "in all\n",
	    begin, bytes, type, ElapsedMs(started), pinned / MiB));
}

void InPlaceMemory::Register(uint64_t bytes, uint64_t pages) noexcept {
	s_live_buffers.fetch_add(1, std::memory_order_acq_rel);
	s_live_bytes.fetch_add(bytes, std::memory_order_relaxed);
	s_pages_in.fetch_add(pages, std::memory_order_relaxed);
}

void InPlaceMemory::Unregister(uint64_t bytes, uint64_t pages) noexcept {
	s_live_buffers.fetch_sub(1, std::memory_order_acq_rel);
	s_live_bytes.fetch_sub(bytes, std::memory_order_relaxed);
	s_pages_out.fetch_add(pages, std::memory_order_relaxed);
}

void InPlaceMemory::NoteGpuWrite(uint64_t vaddr, uint64_t size, uint64_t tick, uint64_t known) {
	if (size == 0) {
		return;
	}
	std::lock_guard lock(m_writes_mutex);
	const auto end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	if (m_written_ranges.size() >= 256) {
		std::erase_if(m_written_ranges, [known](const WrittenRange& range) { return range.tick <= known; });
	}
	if (!m_written_ranges.empty() && m_written_ranges.back().begin == vaddr && m_written_ranges.back().end == end) {
		m_written_ranges.back().tick = std::max(m_written_ranges.back().tick, tick);
	} else {
		m_written_ranges.push_back({vaddr, end, tick});
	}
	for (auto block = vaddr >> BlockBits; block <= (vaddr + size - 1) >> BlockBits; ++block) {
		auto& latest = m_writes[block];
		latest       = std::max(latest, tick);
	}
	if (tick > m_writes_max.load(std::memory_order_relaxed)) {
		m_writes_max.store(tick, std::memory_order_release);
	}
	s_gpu_writes.fetch_add(1, std::memory_order_relaxed);
}

uint64_t InPlaceMemory::PendingGpuWrite(uint64_t vaddr, uint64_t size, uint64_t known) const {
	if (size == 0 || m_writes_max.load(std::memory_order_acquire) <= known) {
		return 0;
	}
	std::lock_guard lock(m_writes_mutex);
	if (m_writes.size() > 16384) {
		std::erase_if(m_writes, [known](const auto& item) { return item.second <= known; });
	}
	uint64_t latest = 0;
	for (auto block = vaddr >> BlockBits; block <= (vaddr + size - 1) >> BlockBits; ++block) {
		if (const auto it = m_writes.find(block); it != m_writes.end() && it->second > known) {
			latest = std::max(latest, it->second);
		}
	}
	return latest;
}

uint64_t InPlaceMemory::PendingGpuWriteBytes(uint64_t vaddr, uint64_t size, uint64_t known) const {
	if (size == 0 || m_writes_max.load(std::memory_order_acquire) <= known) {
		return 0;
	}
	const auto      end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	std::lock_guard lock(m_writes_mutex);
	std::erase_if(m_written_ranges, [known](const WrittenRange& range) { return range.tick <= known; });
	uint64_t latest = 0;
	for (const auto& range: m_written_ranges) {
		if (range.begin < end && vaddr < range.end) {
			latest = std::max(latest, range.tick);
		}
	}
	return latest;
}

uint64_t InPlaceMemory::NoteDeferredLabel(uint64_t vaddr, uint64_t size, uint64_t expires) {
	std::lock_guard lock(s_deferred_labels_mutex);
	const auto      id = ++s_next_deferred_label;
	s_deferred_labels.push_back({id, vaddr, size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size, expires});
	s_deferred_label_count.store(s_deferred_labels.size(), std::memory_order_release);
	return id;
}

void InPlaceMemory::ForgetDeferredLabel(uint64_t id) {
	std::lock_guard lock(s_deferred_labels_mutex);
	std::erase_if(s_deferred_labels, [id](const DeferredLabel& label) { return label.id == id; });
	s_deferred_label_count.store(s_deferred_labels.size(), std::memory_order_release);
}

InPlaceMemory::LabelMeet InPlaceMemory::MeetsDeferredLabel(uint64_t vaddr, uint64_t size, uint64_t natural_tick,
                                                           bool settle) {
	if (size == 0 || s_deferred_label_count.load(std::memory_order_acquire) == 0) {
		return LabelMeet::None;
	}
	const auto      end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
	std::lock_guard lock(s_deferred_labels_mutex);
	std::erase_if(s_deferred_labels,
	              [natural_tick](const DeferredLabel& label) { return label.expires != 0 && label.expires < natural_tick; });
	auto met = LabelMeet::None;
	for (const auto& label: s_deferred_labels) {
		if (label.begin < end && vaddr < label.end) {
			if (label.expires == 0) {
				met = LabelMeet::Pending;
				break;
			}
			met = LabelMeet::WouldBePending;
		}
	}
	if (settle && met == LabelMeet::WouldBePending) {
		std::erase_if(s_deferred_labels,
		              [vaddr, end](const DeferredLabel& label) { return label.begin < end && vaddr < label.end; });
	}
	s_deferred_label_count.store(s_deferred_labels.size(), std::memory_order_release);
	return met;
}

void InPlaceMemory::NoteGpuWriteOverLabel(uint64_t vaddr, uint64_t size) {
	if (s_writes_over_labels.fetch_add(1, std::memory_order_relaxed) < 16) {
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-label-drains] verify: the GPU writes 0x{:x}+0x{:x} in place over a label that "
		    "KYTY_IN_PLACE_LABEL_DEFER=1 would still hold back: with the switch on the emulator stops here\n",
		    vaddr, size));
	}
}

void InPlaceMemory::RequestConversion(uint64_t page) noexcept {
	const auto slot = m_request_head.fetch_add(1, std::memory_order_relaxed) % Requests;
	m_requests[slot].store(page, std::memory_order_relaxed);
	m_requested.store(true, std::memory_order_release);
}

void InPlaceMemory::LogSummary(double seconds) {
	if (!Requested()) {
		return;
	}
	const auto take = [](std::atomic<uint64_t>& value) { return value.exchange(0, std::memory_order_relaxed); };
	const auto bounded = take(s_bounded_flips);
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-command-memory] summary of the last {:.0f} s: {} buffers served in place ({:.1f} MiB), {} converted after a "
	    "fault, refused: {} with a page the GPU wrote, {} not one run of the second view, {} in the guest file, {} across "
	    "chunks, {} chunk refused, {} at bind; {} live buffers in place ({:.1f} MiB), {} chunks ({} MiB pinned); {} pages served in place, {} handed back "
	    "to tracking; {} GPU writes noted, {} reads that waited for one; {} labels written at completion; {} flips that "
	    "waited for the previous frame ({:.1f} ms)\n",
	    seconds, take(s_imported), static_cast<double>(take(s_imported_bytes)) / static_cast<double>(MiB),
	    take(s_conversions), take(s_refused_gpu), take(s_refused_unbacked), take(s_refused_guest), take(s_refused_straddle),
	    take(s_refused_chunk), take(s_refused_bind), s_live_buffers.load(std::memory_order_relaxed),
	    static_cast<double>(s_live_bytes.load(std::memory_order_relaxed)) / static_cast<double>(MiB),
	    s_chunks.load(std::memory_order_relaxed), s_pinned_bytes.load(std::memory_order_relaxed) / MiB, take(s_pages_in),
	    take(s_pages_out), take(s_gpu_writes), take(s_read_waits), take(s_labels), bounded,
	    static_cast<double>(take(s_bounded_ns)) / 1e6));
}

}
