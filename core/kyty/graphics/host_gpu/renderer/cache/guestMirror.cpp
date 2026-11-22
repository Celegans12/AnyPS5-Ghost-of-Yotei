#include "graphics/host_gpu/renderer/cache/guestMirror.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <fmt/format.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <memory>

namespace Libs::Graphics {

namespace {

constexpr auto     BufferUsage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                             vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress;
constexpr auto     BufferFlags = vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;
constexpr uint64_t ProbeWaitNs = 5'000'000'000;

uint64_t Now() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::string Gib(uint64_t bytes) {
	return bytes % GuestMirror::GiB == 0 ? fmt::format("{} GiB", bytes / GuestMirror::GiB)
	                                     : fmt::format("{:.2f} GiB", static_cast<double>(bytes) / GuestMirror::GiB);
}

uint32_t PickMemoryType(const vk::PhysicalDeviceMemoryProperties& memory, uint32_t bits) {
	for (uint32_t pass = 0; pass < 2; pass++) {
		for (uint32_t type = 0; type < memory.memoryTypeCount; type++) {
			if ((bits & (1u << type)) == 0) continue;
			if (pass == 1 || (memory.memoryTypes[type].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
				return type;
			}
		}
	}
	return UINT32_MAX;
}

enum class Outcome { Bound, Lost, Failed, Stuck };

Outcome ProbeOnce(const GraphicContext& graphics, uint64_t size, std::string& detail) {
	const float                     priority = 1.0f;
	const vk::DeviceQueueCreateInfo queue_info {
	    .queueFamilyIndex = graphics.sparse_family, .queueCount = 1, .pQueuePriorities = &priority};
	vk::PhysicalDeviceVulkan12Features features12 {.bufferDeviceAddress = VK_TRUE};
	vk::PhysicalDeviceFeatures         features {};
	features.sparseBinding         = VK_TRUE;
	features.sparseResidencyBuffer = VK_TRUE;
	const vk::DeviceCreateInfo device_info {.pNext                = &features12,
	                                        .queueCreateInfoCount = 1,
	                                        .pQueueCreateInfos    = &queue_info,
	                                        .pEnabledFeatures     = &features};
	vk::Device device;
	if (const auto made = graphics.physical_device.createDevice(&device_info, nullptr, &device);
	    made != vk::Result::eSuccess) {
		detail = fmt::format("probe device refused ({})", vk::to_string(made));
		return Outcome::Failed;
	}
	auto dispatch = std::make_unique<vk::detail::DispatchLoaderDynamic>(
	    static_cast<VkInstance>(graphics.instance), VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
	    static_cast<VkDevice>(device));
	const auto&  d       = *dispatch;
	vk::Queue    queue   = device.getQueue(graphics.sparse_family, 0, d);
	vk::Buffer   buffer  = nullptr;
	vk::DeviceMemory memory = nullptr;
	vk::Fence    fence   = nullptr;
	Outcome      outcome = Outcome::Failed;
	const auto   cleanup = [&](bool device_usable) {
		if (!device_usable) return;
		if (fence) device.destroyFence(fence, nullptr, d);
		if (buffer) device.destroyBuffer(buffer, nullptr, d);
		if (memory) device.freeMemory(memory, nullptr, d);
		device.destroy(nullptr, d);
	};
	const vk::BufferCreateInfo buffer_info {.flags = BufferFlags, .size = size, .usage = BufferUsage};
	if (const auto made = device.createBuffer(&buffer_info, nullptr, &buffer, d); made != vk::Result::eSuccess) {
		detail = fmt::format("buffer refused ({})", vk::to_string(made));
		buffer = nullptr;
		cleanup(true);
		return Outcome::Failed;
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(buffer, &requirements, d);
	const vk::BufferDeviceAddressInfo address_info {.buffer = buffer};
	const uint64_t                    address = device.getBufferAddress(&address_info, d);
	const auto                        where   = fmt::format("at 0x{:x}, end 0x{:x}", address, address + size);
	const auto type = PickMemoryType(graphics.physical_device_memory_properties, requirements.memoryTypeBits);
	const vk::MemoryAllocateFlagsInfo allocate_flags {.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
	const vk::MemoryAllocateInfo      allocate {
	         .pNext = &allocate_flags, .allocationSize = requirements.alignment, .memoryTypeIndex = type};
	const vk::FenceCreateInfo    fence_info {};
	if (type == UINT32_MAX || device.allocateMemory(&allocate, nullptr, &memory, d) != vk::Result::eSuccess ||
	    device.createFence(&fence_info, nullptr, &fence, d) != vk::Result::eSuccess) {
		detail = "no memory for one page";
		cleanup(true);
		return Outcome::Failed;
	}
	vk::SparseMemoryBind bind {.resourceOffset = size - requirements.alignment,
	                           .size           = requirements.alignment,
	                           .memory         = memory,
	                           .memoryOffset   = 0};
	const vk::SparseBufferMemoryBindInfo buffer_bind {.buffer = buffer, .bindCount = 1, .pBinds = &bind};
	const vk::BindSparseInfo             bind_info {.bufferBindCount = 1, .pBufferBinds = &buffer_bind};
	auto result = queue.bindSparse(1, &bind_info, fence, d);
	if (result == vk::Result::eSuccess) {
		result = device.waitForFences(1, &fence, VK_TRUE, ProbeWaitNs, d);
	}
	if (result == vk::Result::eSuccess) {
		(void)device.resetFences(1, &fence, d);
		bind.memory = nullptr;
		result      = queue.bindSparse(1, &bind_info, fence, d);
		if (result == vk::Result::eSuccess) {
			result = device.waitForFences(1, &fence, VK_TRUE, ProbeWaitNs, d);
		}
	}
	const uint64_t space = graphics.physical_device_properties.limits.sparseAddressSpaceSize;
	const uint64_t room_size =
	    std::min(GuestMirror::Headroom, space > size ? space - size : 0) & ~(requirements.alignment - 1);
	vk::Buffer room = nullptr;
	if (result == vk::Result::eSuccess && room_size >= requirements.alignment) {
		(void)device.resetFences(1, &fence, d);
		const vk::BufferCreateInfo room_info {.flags = BufferFlags, .size = room_size, .usage = BufferUsage};
		if (device.createBuffer(&room_info, nullptr, &room, d) != vk::Result::eSuccess) {
			room = nullptr;
			cleanup(true);
			detail = fmt::format("{}: bound, but no room for {} of other memory beside it", where, Gib(room_size));
			return Outcome::Failed;
		}
		const vk::SparseMemoryBind room_bind {.resourceOffset = room_size - requirements.alignment,
		                                      .size           = requirements.alignment,
		                                      .memory         = memory,
		                                      .memoryOffset   = 0};
		const vk::SparseBufferMemoryBindInfo room_buffer {.buffer = room, .bindCount = 1, .pBinds = &room_bind};
		const vk::BindSparseInfo             room_bind_info {.bufferBindCount = 1, .pBufferBinds = &room_buffer};
		result = queue.bindSparse(1, &room_bind_info, fence, d);
		if (result == vk::Result::eSuccess) {
			result = device.waitForFences(1, &fence, VK_TRUE, ProbeWaitNs, d);
		}
		if (result != vk::Result::eTimeout) device.destroyBuffer(room, nullptr, d);
	}
	switch (result) {
		case vk::Result::eSuccess:
			outcome = Outcome::Bound;
			detail  = fmt::format("{}: bound, {} of room beside it", where, Gib(room_size));
			break;
		case vk::Result::eErrorDeviceLost:
			outcome = Outcome::Lost;
			detail  = where + ": device lost";
			break;
		case vk::Result::eTimeout:
			outcome = Outcome::Stuck;
			detail  = fmt::format("no answer after {} s", ProbeWaitNs / 1'000'000'000);
			break;
		default:
			outcome = Outcome::Failed;
			detail  = vk::to_string(result);
			break;
	}
	cleanup(outcome != Outcome::Stuck);
	return outcome;
}

}

std::string GuestMirror::Unsupported(const GraphicContext& graphics) {
	if (!graphics.sparse_buffers_enabled) {
		return "the device has no sparseBinding or no sparseResidencyBuffer";
	}
	if (graphics.sparse_family == static_cast<uint32_t>(-1) || !graphics.sparse_queue) {
		return "no queue family can bind sparse memory";
	}
	return {};
}

GuestMirror::ProbeResult GuestMirror::Probe(const GraphicContext& graphics, std::span<const uint64_t> sizes) {
	ProbeResult result;
	result.refusal = Unsupported(graphics);
	if (!result.refusal.empty()) return result;
	const auto     start = Now();
	const uint64_t space = graphics.physical_device_properties.limits.sparseAddressSpaceSize;
	for (const auto size: sizes) {
		if (size == 0 || size > space) {
			result.trials += fmt::format("{}{}: above the {} of sparse address space", result.trials.empty() ? "" : ", ",
			                             Gib(size), Gib(space));
			continue;
		}
		std::string detail;
		const auto  outcome = ProbeOnce(graphics, size, detail);
		result.trials += fmt::format("{}{}: {}", result.trials.empty() ? "" : ", ", Gib(size), detail);
		if (outcome == Outcome::Bound) {
			result.size = size;
			break;
		}
		if (outcome == Outcome::Stuck) break;
	}
	result.ms = static_cast<double>(Now() - start) / 1e6;
	if (result.size == 0) {
		result.refusal = fmt::format("no size bound on a probe device ({})", result.trials);
	}
	return result;
}

GuestMirror::GuestMirror(GraphicContext& graphics, std::span<const uint64_t> sizes): m_graphics(graphics) {
	const auto probe = Probe(graphics, sizes);
	if (probe.size == 0) {
		m_description = fmt::format("no mirror: {}; the buffer cache works without it", probe.refusal);
		return;
	}
	m_size = probe.size;
	if (!CreateBuffer()) {
		m_size = 0;
		return;
	}
	m_shared_queue = graphics.sparse_queue == graphics.queue;
	m_bound.assign(((m_size >> m_block_bits) + 63) / 64, 0);
	m_resident = std::vector<std::atomic<uint64_t>>(m_bound.size());
	vk::PhysicalDeviceVulkan13Properties properties13 {};
	vk::PhysicalDeviceProperties2        properties {.pNext = &properties13};
	graphics.physical_device.getProperties2(&properties);
	m_description = fmt::format(
	    "sparse mirror of {} at base 0, device address 0x{:x} to 0x{:x} (probe {:.1f} ms: {}), blocks of {} KiB, {} queue of "
	    "family {}, memory type {}{}",
	    Gib(m_size), m_device_address, m_device_address + m_size, probe.ms, probe.trials, BlockSize() >> 10u,
	    m_shared_queue ? "the work" : fmt::format("its own (index {})", graphics.sparse_queue_index),
	    graphics.sparse_family, m_memory_type,
	    m_size > properties13.maxBufferSize ? fmt::format(", above maxBufferSize ({})", Gib(properties13.maxBufferSize))
	                                        : std::string {});
	m_binder = std::jthread([this](std::stop_token stop) { BinderThread(stop); });
}

GuestMirror::~GuestMirror() {
	if (m_binder.joinable()) {
		m_binder.request_stop();
		m_changed.notify_all();
		m_binder.join();
	}
	DestroyBuffer();
}

bool GuestMirror::CreateBuffer() {
	const auto                 device = m_graphics.device;
	const vk::BufferCreateInfo info {.flags = BufferFlags, .size = m_size, .usage = BufferUsage};
	if (const auto made = device.createBuffer(&info, nullptr, &m_buffer); made != vk::Result::eSuccess) {
		m_buffer      = nullptr;
		m_description = fmt::format("no mirror: the device refused a sparse buffer of {} ({})", Gib(m_size),
		                            vk::to_string(made));
		return false;
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(m_buffer, &requirements);
	const vk::BufferDeviceAddressInfo address_info {.buffer = m_buffer};
	m_device_address = device.getBufferAddress(&address_info);
	m_block_bits  = std::max<uint32_t>(
        MinBlockBits, static_cast<uint32_t>(std::bit_width(std::max<uint64_t>(requirements.alignment, 1)) - 1));
	m_memory_type = PickMemoryType(m_graphics.physical_device_memory_properties, requirements.memoryTypeBits);
	const vk::FenceCreateInfo fence_info {};
	if (!std::has_single_bit(requirements.alignment) || m_memory_type == UINT32_MAX ||
	    device.createFence(&fence_info, nullptr, &m_fence) != vk::Result::eSuccess) {
		m_description = fmt::format("no mirror: page of {} bytes, memory types 0x{:x}", requirements.alignment,
		                            requirements.memoryTypeBits);
		DestroyBuffer();
		return false;
	}
	return true;
}

void GuestMirror::DestroyBuffer() {
	const auto device = m_graphics.device;
	if (m_buffer) device.destroyBuffer(m_buffer);
	for (auto& chunk: m_chunks) {
		if (chunk.memory) device.freeMemory(chunk.memory);
	}
	if (m_fence) device.destroyFence(m_fence);
	m_buffer = nullptr;
	m_fence  = nullptr;
	m_chunks.clear();
}

vk::Buffer GuestMirror::Handle() {
	return m_buffer;
}

bool GuestMirror::IsBound(uint64_t vaddr) const noexcept {
	const auto block = vaddr >> m_block_bits;
	return block / 64 < m_bound.size() && (m_bound[block / 64] & (uint64_t {1} << (block % 64))) != 0;
}

void GuestMirror::SetBound(uint64_t first, uint64_t count, bool bound) noexcept {
	for (auto block = first; block < first + count; block++) {
		const auto bit = uint64_t {1} << (block % 64);
		m_bound[block / 64] = bound ? (m_bound[block / 64] | bit) : (m_bound[block / 64] & ~bit);
	}
}

void GuestMirror::NoteRange(uint64_t vaddr, uint64_t size) {
	if (m_size == 0 || size == 0 || UINT64_MAX - vaddr < size) return;
	const uint64_t blocks = m_size >> m_block_bits;
	const uint64_t first  = vaddr >> m_block_bits;
	uint64_t       last   = (vaddr + size - 1) >> m_block_bits;
	if (first >= blocks) {
		m_outside_blocks += last - first + 1;
		return;
	}
	if (last >= blocks) {
		m_outside_blocks += last - blocks + 1;
		last = blocks - 1;
	}
	for (auto block = first; block <= last; block++) {
		auto&      word = m_bound[block / 64];
		const auto bit  = uint64_t {1} << (block % 64);
		if ((word & bit) != 0) continue;
		word |= bit;
		m_pending_blocks++;
		if (!m_pending.empty() && m_pending.back().first + m_pending.back().count == block) {
			m_pending.back().count++;
		} else {
			m_pending.push_back({block, 1});
		}
	}
}

void GuestMirror::MakeResident(uint64_t vaddr, uint64_t size) {
	NoteRange(vaddr, size);
	if (m_size == 0 || size == 0 || UINT64_MAX - vaddr < size) return;
	const uint64_t last = std::min((vaddr + size - 1) >> m_block_bits, (m_size >> m_block_bits) - 1);
	for (auto block = vaddr >> m_block_bits; block <= last; block++) {
		if ((m_resident[block / 64].load(std::memory_order_acquire) & (uint64_t {1} << (block % 64))) == 0) {
			Flush();
			WaitIdle();
			return;
		}
	}
}

void GuestMirror::Post(Kind kind, std::vector<Run>&& runs) {
	{
		Common::ScopedLock lock(m_mutex);
		if (kind == Kind::Bind && !m_commands.empty() && m_commands.back().kind == Kind::Bind) {
			auto& queued = m_commands.back().runs;
			queued.insert(queued.end(), runs.begin(), runs.end());
		} else {
			m_commands.push_back({kind, std::move(runs)});
		}
	}
	m_changed.notify_all();
}

void GuestMirror::Submit() {
	if (m_pending.empty()) return;
	if (m_pending_blocks < FlushBlocks && Now() - m_last_post < FlushNs) return;
	Flush();
}

void GuestMirror::Flush() {
	if (m_pending.empty()) return;
	m_last_post      = Now();
	m_pending_blocks = 0;
	Post(Kind::Bind, std::move(m_pending));
	m_pending.clear();
}

void GuestMirror::Trim(std::span<const std::pair<uint64_t, uint64_t>> live) {
	if (m_size == 0) return;
	Flush();
	std::vector<uint64_t> covered(m_bound.size(), 0);
	const uint64_t        blocks = m_size >> m_block_bits;
	for (const auto& [vaddr, size]: live) {
		if (size == 0 || UINT64_MAX - vaddr < size) continue;
		const auto first = vaddr >> m_block_bits;
		const auto last  = std::min((vaddr + size - 1) >> m_block_bits, blocks - 1);
		for (auto block = first; block <= last && block < blocks; block++) {
			covered[block / 64] |= uint64_t {1} << (block % 64);
		}
	}
	std::vector<Run> runs;
	for (uint64_t word = 0; word < m_bound.size(); word++) {
		const auto unused = m_bound[word] & ~covered[word];
		for (auto bits = unused; bits != 0; bits &= bits - 1) {
			const auto block = word * 64 + static_cast<uint64_t>(std::countr_zero(bits));
			if (!runs.empty() && runs.back().first + runs.back().count == block) {
				runs.back().count++;
			} else {
				runs.push_back({block, 1});
			}
		}
		m_bound[word] &= covered[word];
	}
	if (!runs.empty()) Post(Kind::Unbind, std::move(runs));
}

void GuestMirror::WaitIdle() {
	if (m_size == 0) return;
	Common::UniqueLock lock(m_mutex);
	m_changed.wait(lock.Native(), [this]() KYTY_REQUIRES(m_mutex) { return m_commands.empty() && !m_busy; });
}

void GuestMirror::BinderThread(std::stop_token stop) {
	KYTY_PROFILER_THREAD("Thread_GuestMirror");
	for (;;) {
		Command command;
		{
			Common::UniqueLock lock(m_mutex);
			m_changed.wait(lock.Native(), stop, [this]() KYTY_REQUIRES(m_mutex) { return !m_commands.empty(); });
			if (m_commands.empty()) return;
			command = std::move(m_commands.front());
			m_commands.pop_front();
			m_busy = true;
		}
		switch (command.kind) {
			case Kind::Bind: Bind(command.runs); break;
			case Kind::Unbind: Unbind(command.runs); break;
		}
		{
			Common::ScopedLock lock(m_mutex);
			m_busy = false;
		}
		m_changed.notify_all();
	}
}

GuestMirror::Grant GuestMirror::Allocate(uint64_t wanted) {
	const auto slots = static_cast<uint32_t>(std::max<uint64_t>(ChunkBytes >> m_block_bits, 1));
	if (m_current_chunk != UINT32_MAX && m_chunks[m_current_chunk].next < slots) {
		auto&      chunk = m_chunks[m_current_chunk];
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(wanted, slots - chunk.next));
		const auto first = chunk.next;
		chunk.next += count;
		chunk.used += count;
		return {m_current_chunk, first, count};
	}
	if (!m_free_slots.empty()) {
		const auto slot = m_free_slots.back();
		m_free_slots.pop_back();
		m_chunks[slot.chunk].used++;
		return {slot.chunk, slot.index, 1};
	}
	const vk::MemoryAllocateFlagsInfo flags {.flags = vk::MemoryAllocateFlagBits::eDeviceAddress};
	const vk::MemoryAllocateInfo      info {
	         .pNext = &flags, .allocationSize = uint64_t {slots} << m_block_bits, .memoryTypeIndex = m_memory_type};
	vk::DeviceMemory             memory;
	const auto                   made = m_graphics.device.allocateMemory(&info, nullptr, &memory);
	if (made != vk::Result::eSuccess) {
		EXIT("[kyty-mirror] could not allocate %" PRIu64 " MiB for the mirror: %s\n", info.allocationSize >> 20u,
		     vk::to_string(made).c_str());
	}
	if (!m_free_chunks.empty()) {
		m_current_chunk = m_free_chunks.back();
		m_free_chunks.pop_back();
		m_chunks[m_current_chunk] = {memory, 0, 0};
	} else {
		m_current_chunk = static_cast<uint32_t>(m_chunks.size());
		m_chunks.push_back({memory, 0, 0});
	}
	m_chunk_count.fetch_add(1, std::memory_order_relaxed);
	return Allocate(wanted);
}

void GuestMirror::Bind(const std::vector<Run>& runs) {
	m_binds.clear();
	uint64_t blocks = 0;
	for (const auto& run: runs) {
		auto block = run.first;
		auto left  = run.count;
		while (left != 0) {
			const auto [chunk, index, count] = Allocate(left);
			m_binds.push_back({.resourceOffset = block << m_block_bits,
			                   .size           = uint64_t {count} << m_block_bits,
			                   .memory         = m_chunks[chunk].memory,
			                   .memoryOffset   = uint64_t {index} << m_block_bits});
			for (uint32_t i = 0; i < count; i++) m_block_slots[block + i] = {chunk, index + i};
			block += count;
			left -= count;
			blocks += count;
		}
	}
	Execute(m_binds, "bind");
	for (const auto& run: runs) {
		for (auto block = run.first; block < run.first + run.count; block++) {
			m_resident[block / 64].fetch_or(uint64_t {1} << (block % 64), std::memory_order_release);
		}
	}
	m_blocks.fetch_add(blocks, std::memory_order_relaxed);
	m_bound_blocks.fetch_add(blocks, std::memory_order_relaxed);
}

void GuestMirror::Unbind(const std::vector<Run>& runs) {
	m_binds.clear();
	uint64_t blocks = 0;
	for (const auto& run: runs) {
		m_binds.push_back({.resourceOffset = run.first << m_block_bits, .size = run.count << m_block_bits});
		for (auto block = run.first; block < run.first + run.count; block++) {
			m_resident[block / 64].fetch_and(~(uint64_t {1} << (block % 64)), std::memory_order_release);
		}
	}
	Execute(m_binds, "unbind");
	for (const auto& run: runs) {
		for (auto block = run.first; block < run.first + run.count; block++) {
			const auto found = m_block_slots.find(block);
			if (found == m_block_slots.end()) continue;
			Free(found->second);
			m_block_slots.erase(found);
			blocks++;
		}
	}
	std::vector<bool> emptied(m_chunks.size(), false);
	for (uint32_t index = 0; index < m_chunks.size(); index++) {
		auto& chunk = m_chunks[index];
		if (chunk.memory && chunk.used == 0) {
			m_graphics.device.freeMemory(chunk.memory);
			chunk          = {};
			emptied[index] = true;
			m_free_chunks.push_back(index);
			m_chunk_count.fetch_sub(1, std::memory_order_relaxed);
			if (m_current_chunk == index) m_current_chunk = UINT32_MAX;
		}
	}
	std::erase_if(m_free_slots, [&](const Slot& slot) { return emptied[slot.chunk]; });
	m_unbound_blocks.fetch_add(blocks, std::memory_order_relaxed);
	m_bound_blocks.fetch_sub(std::min(blocks, m_bound_blocks.load(std::memory_order_relaxed)), std::memory_order_relaxed);
}

void GuestMirror::Free(Slot slot) {
	m_chunks[slot.chunk].used--;
	m_free_slots.push_back(slot);
}

void GuestMirror::Execute(std::vector<vk::SparseMemoryBind>& binds, const char* what) {
	const auto device = m_graphics.device;
	for (size_t begin = 0; begin < binds.size(); begin += MaxBindsPerCall) {
		const auto                           count = std::min(MaxBindsPerCall, binds.size() - begin);
		const vk::SparseBufferMemoryBindInfo buffer_bind {
		    .buffer = m_buffer, .bindCount = static_cast<uint32_t>(count), .pBinds = binds.data() + begin};
		const vk::BindSparseInfo info {.bufferBindCount = 1, .pBufferBinds = &buffer_bind};
		const auto               start = Now();
		vk::Result               result;
		if (m_shared_queue) {
			Common::LockGuard lock(m_graphics.queue_mutex);
			result = m_graphics.sparse_queue.bindSparse(1, &info, m_fence);
		} else {
			result = m_graphics.sparse_queue.bindSparse(1, &info, m_fence);
		}
		const auto called = Now();
		if (result == vk::Result::eSuccess) {
			result = device.waitForFences(1, &m_fence, VK_TRUE, UINT64_MAX);
		}
		if (result != vk::Result::eSuccess) {
			EXIT("[kyty-mirror] vkQueueBindSparse (%s of %zu ranges at 0x%" PRIx64 ") failed: %s\n", what, count,
			     binds[begin].resourceOffset, vk::to_string(result).c_str());
		}
		(void)device.resetFences(1, &m_fence);
		const auto done = Now();
		m_calls.fetch_add(1, std::memory_order_relaxed);
		m_call_ns.fetch_add(called - start, std::memory_order_relaxed);
		m_done_ns.fetch_add(done - start, std::memory_order_relaxed);
		auto longest = m_longest_ns.load(std::memory_order_relaxed);
		while (done - start > longest &&
		       !m_longest_ns.compare_exchange_weak(longest, done - start, std::memory_order_relaxed)) {
		}
	}
}

GuestMirror::Stats GuestMirror::Totals() const {
	return {.bound_blocks   = m_bound_blocks.load(std::memory_order_relaxed),
	        .chunks         = m_chunk_count.load(std::memory_order_relaxed),
	        .calls          = m_calls.load(std::memory_order_relaxed),
	        .blocks         = m_blocks.load(std::memory_order_relaxed),
	        .unbound_blocks = m_unbound_blocks.load(std::memory_order_relaxed),
	        .call_ns        = m_call_ns.load(std::memory_order_relaxed),
	        .done_ns        = m_done_ns.load(std::memory_order_relaxed),
	        .longest_ns     = m_longest_ns.load(std::memory_order_relaxed)};
}

GuestMirror::Stats GuestMirror::TakeStats() {
	const auto totals = Totals();
	Stats      delta  = totals;
	delta.calls          = totals.calls - m_taken.calls;
	delta.blocks         = totals.blocks - m_taken.blocks;
	delta.unbound_blocks = totals.unbound_blocks - m_taken.unbound_blocks;
	delta.call_ns        = totals.call_ns - m_taken.call_ns;
	delta.done_ns        = totals.done_ns - m_taken.done_ns;
	m_taken              = totals;
	delta.longest_ns     = m_longest_ns.exchange(0, std::memory_order_relaxed);
	return delta;
}

}
