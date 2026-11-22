#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/guestMirror.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::GraphicContext;
using Libs::Graphics::GuestMirror;

constexpr const char* Name        = "kyty_shim_mirror_shadow";
constexpr int         SkippedExit = 77;
constexpr uint64_t    GiB         = GuestMirror::GiB;

uint32_t g_failures = 0;

void Require(bool value, const std::string& what) {
	if (value) return;
	std::fprintf(stderr, "%s: %s\n", Name, what.c_str());
	std::abort();
}

void Check(bool value, const std::string& what) {
	if (value) return;
	if (g_failures++ < 32) std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
}

bool Contains(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

struct Device {
	GraphicContext graphics;
	std::string    name;

	~Device() {
		if (graphics.device) graphics.device.destroy();
		if (graphics.instance) graphics.instance.destroy();
	}
};

bool Open(Device& gpu, bool sparse, std::string& why) {
	static const vk::detail::DynamicLoader loader;
	if (!loader.success()) {
		why = "the Vulkan loader is missing";
		return false;
	}
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) {
		why = "no Vulkan 1.3 instance";
		return false;
	}
	auto& graphics    = gpu.graphics;
	graphics.instance = created.value;
	dispatch.init(graphics.instance);
	const auto devices = graphics.instance.enumeratePhysicalDevices();
	Require(devices.result == vk::Result::eSuccess, "device enumeration failed");
	vk::PhysicalDevice chosen;
	for (const auto candidate: devices.value) {
		const auto properties = candidate.getProperties();
		if (properties.apiVersion < VK_API_VERSION_1_3 || properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu) {
			continue;
		}
		const auto features = candidate.getFeatures();
		if (sparse && (!features.sparseBinding || !features.sparseResidencyBuffer)) continue;
		if (chosen && properties.deviceType != vk::PhysicalDeviceType::eDiscreteGpu) continue;
		chosen = candidate;
	}
	if (!chosen) {
		why = sparse ? "no Vulkan 1.3 device outside integrated GPUs binds sparse buffers"
		             : "no Vulkan 1.3 device outside integrated GPUs";
		return false;
	}
	const auto families = chosen.getQueueFamilyProperties();
	const auto main     = std::ranges::find_if(families, [](const auto& family) {
        return static_cast<bool>(family.queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute));
    });
	Require(main != families.end(), "no graphics or compute queue");
	graphics.queue_family = static_cast<uint32_t>(main - families.begin());
	graphics.sparse_family = static_cast<uint32_t>(-1);
	if (sparse) {
		for (uint32_t family = 0; family < families.size(); family++) {
			if (family != graphics.queue_family && (families[family].queueFlags & vk::QueueFlagBits::eSparseBinding)) {
				graphics.sparse_family = family;
				break;
			}
		}
		if (graphics.sparse_family == static_cast<uint32_t>(-1)) {
			if (!(families[graphics.queue_family].queueFlags & vk::QueueFlagBits::eSparseBinding)) {
				why = "no queue family binds sparse memory";
				return false;
			}
			graphics.sparse_family = graphics.queue_family;
		}
	}
	const std::array<float, 1>               priorities {1.0f};
	std::array<vk::DeviceQueueCreateInfo, 2> queues {};
	queues[0] = vk::DeviceQueueCreateInfo {
	    .queueFamilyIndex = graphics.queue_family, .queueCount = 1, .pQueuePriorities = priorities.data()};
	uint32_t queue_count = 1;
	if (sparse && graphics.sparse_family != graphics.queue_family) {
		queues[queue_count++] = vk::DeviceQueueCreateInfo {
		    .queueFamilyIndex = graphics.sparse_family, .queueCount = 1, .pQueuePriorities = priorities.data()};
	}
	vk::PhysicalDeviceVulkan12Features features12 {.bufferDeviceAddress = VK_TRUE};
	vk::PhysicalDeviceFeatures         features {};
	features.sparseBinding         = sparse ? VK_TRUE : VK_FALSE;
	features.sparseResidencyBuffer = sparse ? VK_TRUE : VK_FALSE;
	const vk::DeviceCreateInfo device_info {.pNext                = &features12,
	                                        .queueCreateInfoCount = queue_count,
	                                        .pQueueCreateInfos    = queues.data(),
	                                        .pEnabledFeatures     = &features};
	const auto device = chosen.createDevice(device_info);
	Require(device.result == vk::Result::eSuccess, "device creation failed");
	graphics.device = device.value;
	dispatch.init(graphics.device);
	graphics.physical_device                  = chosen;
	graphics.physical_device_properties       = chosen.getProperties();
	graphics.physical_device_memory_properties = chosen.getMemoryProperties();
	graphics.queue                            = graphics.device.getQueue(graphics.queue_family, 0);
	graphics.sparse_buffers_enabled           = sparse;
	if (sparse) graphics.sparse_queue = graphics.device.getQueue(graphics.sparse_family, 0);
	gpu.name = graphics.physical_device_properties.deviceName.data();
	return true;
}

void Fallback() {
	Device      gpu;
	std::string why;
	if (!Open(gpu, false, why)) {
		std::printf("%s: fallback not checked, %s\n", Name, why.c_str());
		return;
	}
	Check(Contains(GuestMirror::Unsupported(gpu.graphics), "sparseBinding"),
	      "a device without sparse binding is refused by name: " + GuestMirror::Unsupported(gpu.graphics));
	GuestMirror mirror(gpu.graphics);
	Check(!mirror.Available() && mirror.Size() == 0, "no mirror without sparse binding");
	Check(Contains(mirror.Description(), "no mirror: the device has no sparseBinding") &&
	          Contains(mirror.Description(), "works without it"),
	      "the fallback says why: " + mirror.Description());
	mirror.NoteRange(0x10000, 0x10000);
	mirror.Flush();
	mirror.WaitIdle();
	Check(mirror.Totals().calls == 0 && !mirror.IsBound(0x10000), "the fallback binds nothing");
}

struct HostBuffer {
	vk::Device       device;
	vk::Buffer       buffer;
	vk::DeviceMemory memory;
	void*            mapped = nullptr;

	HostBuffer(const GraphicContext& graphics, uint64_t size): device(graphics.device) {
		const vk::BufferCreateInfo info {
		    .size = size, .usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst};
		Require(device.createBuffer(&info, nullptr, &buffer) == vk::Result::eSuccess, "host buffer");
		const auto requirements = device.getBufferMemoryRequirements(buffer);
		const auto& types       = graphics.physical_device_memory_properties;
		uint32_t    type        = UINT32_MAX;
		for (uint32_t index = 0; index < types.memoryTypeCount && type == UINT32_MAX; index++) {
			const auto flags = types.memoryTypes[index].propertyFlags;
			if ((requirements.memoryTypeBits & (1u << index)) && (flags & vk::MemoryPropertyFlagBits::eHostVisible) &&
			    (flags & vk::MemoryPropertyFlagBits::eHostCoherent)) {
				type = index;
			}
		}
		Require(type != UINT32_MAX, "no host-visible coherent memory");
		const vk::MemoryAllocateInfo allocate {.allocationSize = requirements.size, .memoryTypeIndex = type};
		Require(device.allocateMemory(&allocate, nullptr, &memory) == vk::Result::eSuccess, "host memory");
		Require(device.bindBufferMemory(buffer, memory, 0) == vk::Result::eSuccess, "host bind");
		Require(device.mapMemory(memory, 0, size, {}, &mapped) == vk::Result::eSuccess, "host map");
	}
	~HostBuffer() {
		device.unmapMemory(memory);
		device.destroyBuffer(buffer);
		device.freeMemory(memory);
	}
};

bool RoundTrip(const GraphicContext& graphics, vk::Buffer mirror, uint64_t offset, uint64_t size, uint32_t seed) {
	HostBuffer source(graphics, size);
	HostBuffer target(graphics, size);
	auto*      words = static_cast<uint32_t*>(source.mapped);
	for (uint64_t index = 0; index < size / 4; index++) words[index] = seed * 2654435761u + static_cast<uint32_t>(index);
	std::memset(target.mapped, 0, size);
	const auto                      device = graphics.device;
	const vk::CommandPoolCreateInfo pool_info {.queueFamilyIndex = graphics.queue_family};
	vk::CommandPool                 pool;
	Require(device.createCommandPool(&pool_info, nullptr, &pool) == vk::Result::eSuccess, "command pool");
	const vk::CommandBufferAllocateInfo allocate {.commandPool = pool, .commandBufferCount = 1};
	vk::CommandBuffer                   command;
	Require(device.allocateCommandBuffers(&allocate, &command) == vk::Result::eSuccess, "command buffer");
	const vk::CommandBufferBeginInfo begin {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
	Require(command.begin(&begin) == vk::Result::eSuccess, "begin");
	const vk::BufferCopy in {.srcOffset = 0, .dstOffset = offset, .size = size};
	command.copyBuffer(source.buffer, mirror, 1, &in);
	const vk::MemoryBarrier barrier {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                                 .dstAccessMask = vk::AccessFlagBits::eTransferRead};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, 1, &barrier, 0,
	                        nullptr, 0, nullptr);
	const vk::BufferCopy out {.srcOffset = offset, .dstOffset = 0, .size = size};
	command.copyBuffer(mirror, target.buffer, 1, &out);
	const vk::MemoryBarrier host {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                              .dstAccessMask = vk::AccessFlagBits::eHostRead};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0, nullptr,
	                        0, nullptr);
	Require(command.end() == vk::Result::eSuccess, "end");
	const vk::SubmitInfo submit {.commandBufferCount = 1, .pCommandBuffers = &command};
	Require(graphics.queue.submit(1, &submit, nullptr) == vk::Result::eSuccess, "submit");
	Require(graphics.queue.waitIdle() == vk::Result::eSuccess, "wait");
	const bool same = std::memcmp(source.mapped, target.mapped, size) == 0;
	device.destroyCommandPool(pool);
	return same;
}

int Sparse() {
	Device      gpu;
	std::string why;
	if (!Open(gpu, true, why)) {
		std::printf("%s: skipped, %s\n", Name, why.c_str());
		return SkippedExit;
	}
	auto&          graphics = gpu.graphics;
	const uint64_t space    = graphics.physical_device_properties.limits.sparseAddressSpaceSize;
	const uint64_t size     = std::min<uint64_t>(4 * GiB, space & ~(GiB - 1));
	Require(size >= GiB, "the device has less than 1 GiB of sparse address space");

	const bool                    above = space <= UINT64_MAX - GiB;
	const std::array<uint64_t, 2> probe_sizes {above ? space + GiB : size, size};
	const auto                    probe = GuestMirror::Probe(graphics, probe_sizes);
	Check(probe.size == size, "the probe binds a page at the end of a " + std::to_string(size >> 30u) +
	                              " GiB buffer: " + probe.trials + " " + probe.refusal);
	Check(!above || Contains(probe.trials, "above the"),
	      "the probe skips a size above the sparse address space: " + probe.trials);

	const auto queue = graphics.sparse_queue;
	graphics.sparse_queue = nullptr;
	Check(Contains(GuestMirror::Unsupported(graphics), "no queue family"), "a device without a sparse queue is refused");
	graphics.sparse_queue = queue;

	const std::array<uint64_t, 1> sizes {size};
	GuestMirror                   mirror(graphics, sizes);
	Require(mirror.Available(), "the mirror exists: " + mirror.Description());
	Check(mirror.Size() == size && Contains(mirror.Description(), "sparse mirror of"), mirror.Description());
	const uint64_t block = mirror.BlockSize();
	Check(block >= 0x10000 && (block & (block - 1)) == 0, "blocks are a power of two of 64 KiB or more");

	const uint64_t first = size / 2 + 3 * block + 0x1230;
	mirror.NoteRange(first, 0x100);
	Check(mirror.Totals().calls == 0, "nothing is bound before the submission");
	mirror.Flush();
	mirror.WaitIdle();
	auto totals = mirror.Totals();
	Check(totals.bound_blocks == 1 && totals.calls == 1 && totals.chunks == 1, "one block bound in one call");
	mirror.NoteRange(first + 2 * block, 0x40);
	mirror.Submit();
	mirror.WaitIdle();
	Check(mirror.Totals().calls == 1 && mirror.IsBound(first + 2 * block),
	      "a submission within a second of the last binding keeps its block for later");
	mirror.Flush();
	mirror.WaitIdle();
	totals = mirror.Totals();
	Check(totals.calls == 2 && totals.bound_blocks == 2, "a flush binds what the submissions kept");
	Check(mirror.IsBound(first) && !mirror.IsBound(first + block), "the residency map holds the block");
	Check(RoundTrip(graphics, mirror.Handle(), first & ~(block - 1), block, 1), "the bound block keeps what is written to it");

	mirror.NoteRange(first + 0x40, 0x40);
	mirror.Flush();
	mirror.WaitIdle();
	Check(mirror.Totals().calls == 2, "a block already bound is not bound again");

	mirror.NoteRange((first & ~(block - 1)) + block - 0x10, 0x20);
	mirror.Flush();
	mirror.WaitIdle();
	totals = mirror.Totals();
	Check(totals.bound_blocks == 3 && totals.calls == 3, "a range across two blocks binds the missing one");
	Check(RoundTrip(graphics, mirror.Handle(), first & ~(block - 1), 2 * block, 2), "two neighbouring blocks hold data");

	mirror.NoteRange(size + block, block);
	Check(mirror.OutsideBlocks() == 1, "a range above the mirror is counted outside");

	const auto run_base = size - 64 * (uint64_t {1} << 20u);
	for (uint64_t index = 0; index < 300; index++) mirror.NoteRange(run_base + index * 2 * block, 1);
	mirror.Flush();
	mirror.WaitIdle();
	totals = mirror.Totals();
	Check(totals.bound_blocks == 303 && totals.calls == 5,
	      "300 scattered blocks go in calls of 256 ranges at most (calls " + std::to_string(totals.calls) + ")");
	Check(RoundTrip(graphics, mirror.Handle(), run_base + 299 * 2 * block, block, 3), "the last scattered block holds data");

	const std::array<std::pair<uint64_t, uint64_t>, 1> live {{{first, 0x100}}};
	mirror.Trim(live);
	mirror.WaitIdle();
	totals = mirror.Totals();
	Check(totals.bound_blocks == 1 && totals.unbound_blocks == 302, "a trim unbinds the blocks no live range covers");
	Check(mirror.IsBound(first) && !mirror.IsBound(run_base), "the trim keeps the covered block");
	Check(totals.chunks == 1, "a trim keeps the chunk that still holds a bound block");
	Check(RoundTrip(graphics, mirror.Handle(), first & ~(block - 1), block, 4), "the kept block still holds data");

	mirror.NoteRange(run_base, 1);
	mirror.Flush();
	mirror.WaitIdle();
	Check(mirror.Totals().bound_blocks == 2, "a trimmed block is bound again when a buffer comes back");

	mirror.Trim({});
	mirror.WaitIdle();
	totals = mirror.Totals();
	Check(totals.bound_blocks == 0 && totals.chunks == 0 && !mirror.IsBound(first) && !mirror.IsBound(run_base),
	      "a trim without live buffers unbinds everything and frees every chunk");
	mirror.NoteRange(first, 0x100);
	mirror.Flush();
	mirror.WaitIdle();
	totals = mirror.Totals();
	Check(totals.bound_blocks == 1 && totals.chunks == 1, "a block is bound again in a new chunk");
	Check(RoundTrip(graphics, mirror.Handle(), first & ~(block - 1), block, 5), "the block bound again holds data");

	const auto stats = mirror.TakeStats();
	Check(stats.calls == mirror.Totals().calls && stats.done_ns >= stats.call_ns, "the first stats cover everything");
	Check(mirror.TakeStats().calls == 0, "stats are taken as differences");
	std::printf("%s: sparse checks on %s (%s)\n", Name, gpu.name.c_str(), mirror.Description().c_str());
	return 0;
}

int DefaultSizes() {
	Device      gpu;
	std::string why;
	if (!Open(gpu, true, why)) {
		std::printf("%s: skipped, %s\n", Name, why.c_str());
		return SkippedExit;
	}
	GuestMirror mirror(gpu.graphics);
	if (!mirror.Available()) {
		Check(Contains(mirror.Description(), "no mirror: no size bound on a probe device"),
		      "a device whose sparse space is too small for every default size has no mirror: " + mirror.Description());
		std::printf("%s: default sizes on %s: %s\n", Name, gpu.name.c_str(), mirror.Description().c_str());
		return 0;
	}
	constexpr uint64_t highest = 0xB4'4582'0000;
	const uint64_t     block   = mirror.BlockSize();
	const uint64_t     address = std::min(highest, mirror.Size() - block) & ~(block - 1);
	mirror.NoteRange(address, block);
	mirror.NoteRange(0x228'0000, 0x2'BE50);
	mirror.Flush();
	mirror.WaitIdle();
	Check(mirror.IsBound(address) && mirror.IsBound(0x228'0000), "blocks bind at both ends of the mirror");
	Check(RoundTrip(gpu.graphics, mirror.Handle(), address, block, 6), "the highest block holds data");
	Check(RoundTrip(gpu.graphics, mirror.Handle(), 0x228'0000 & ~(block - 1), block, 7), "the lowest block holds data");
	std::printf("%s: default sizes on %s: %s\n", Name, gpu.name.c_str(), mirror.Description().c_str());
	return 0;
}

}

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "default-sizes") == 0) {
		const int result = DefaultSizes();
		if (g_failures != 0) {
			std::fprintf(stderr, "%s: %u checks failed\n", Name, g_failures);
			return 1;
		}
		if (result == SkippedExit) return SkippedExit;
		std::printf("%s: ok\n", Name);
		return 0;
	}
	Fallback();
	const int sparse = Sparse();
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %u checks failed\n", Name, g_failures);
		return 1;
	}
	if (sparse == SkippedExit) return SkippedExit;
	std::printf("%s: ok\n", Name);
	return 0;
}
