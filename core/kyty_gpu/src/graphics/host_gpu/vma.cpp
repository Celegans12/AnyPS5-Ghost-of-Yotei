#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/alignment.h"
#include "common/assert.h"
#include "common/envDefaults.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

bool VramGcFixEnabled() {
	static const bool on = [] {
		// Default from Common::EnvDefaults; =0 turns it off.
		const char* value   = Common::EnvOrDefault("KYTY_VRAM_GC_FIX");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) {
			std::fprintf(stderr, "VRAM_GC_FIX enabled (texture collector scans past unfreeable images; GPU-written "
			                     "color images are downloaded before they are freed)\n");
		}
		return enabled;
	}();
	return on;
}

bool VramBudgetEnabled() {
	static const bool on = [] {
		// Default from Common::EnvDefaults; =0 turns it off.
		const char* value   = Common::EnvOrDefault("KYTY_VRAM_BUDGET");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) {
			std::fprintf(stderr, "VRAM_BUDGET enforcement enabled (target = min(KYTY_VRAM_BUDGET_MB cap, driver budget "
			                     "- 1 GiB), re-read every collector run; images are evicted before new ones are "
			                     "created over it)\n");
		}
		return enabled;
	}();
	return on;
}

// KYTY_VRAM_BUDGET_MB=<n>: the caches size their garbage-collection thresholds for at most n MiB.
static uint64_t VramBudgetCap() {
	static const uint64_t cap = [] {
		const char* value = std::getenv("KYTY_VRAM_BUDGET_MB"); // default 11264 (user limit: 11 GB); =0 turns the cap off
		const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 11264ull;
		if (mib != 0) std::fprintf(stderr, "VRAM_BUDGET cap=%lluMiB\n", static_cast<unsigned long long>(mib));
		return mib << 20u;
	}();
	return cap;
}

// KYTY_TRACE_PERF (PERF device_memory): every vkAllocateMemory / vkFreeMemory of the allocator, timed ("slow runs",
// 19.10: in 6 of 7 slow runs a 256 MiB VMA block was freed and allocated again every few seconds).
namespace {
struct DeviceMemoryCalls {
	std::atomic<uint64_t> allocs {0}, alloc_bytes {0}, alloc_ns {0}, alloc_max_ns {0}, big_allocs {0};
	std::atomic<uint64_t> frees {0}, free_ns {0}, big_frees {0};
};
DeviceMemoryCalls      g_memory_calls;
PFN_vkAllocateMemory   g_real_allocate = nullptr;
PFN_vkFreeMemory       g_real_free     = nullptr;
constexpr VkDeviceSize BigBytes        = VkDeviceSize {128} << 20u;
thread_local uint64_t  t_allocs        = 0; // this thread's vkAllocateMemory calls (which image creations allocated)
// Images whose creation allocated device memory (dedicated or a new block): count per format/size/usage.
std::mutex                                g_image_alloc_mutex;
std::unordered_map<std::string, uint64_t> g_image_allocs;
std::atomic<uint64_t>                     g_image_alloc_calls {0}, g_image_creates {0}, g_image_deletes {0};
// Texture relocation: the size of every live VkDeviceMemory (vkFreeMemory is not given it) and the frees since start, so a
// batch counts the blocks and bytes it gave back from the driver calls themselves.
std::mutex                                g_memory_size_mutex;
std::unordered_map<VkDeviceMemory, uint64_t> g_memory_sizes;
uint64_t                                  g_memory_frees = 0, g_memory_freed_bytes = 0;

uint64_t NanosSince(std::chrono::steady_clock::time_point start) {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

VKAPI_ATTR VkResult VKAPI_CALL TimedAllocateMemory(VkDevice device, const VkMemoryAllocateInfo* info,
                                                   const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
	const auto     start  = std::chrono::steady_clock::now();
	const VkResult result = g_real_allocate(device, info, callbacks, memory);
	const uint64_t ns     = NanosSince(start);
	auto&          c      = g_memory_calls;
	c.allocs.fetch_add(1, std::memory_order_relaxed);
	t_allocs++;
	c.alloc_bytes.fetch_add(info->allocationSize, std::memory_order_relaxed);
	c.alloc_ns.fetch_add(ns, std::memory_order_relaxed);
	if (info->allocationSize >= BigBytes) c.big_allocs.fetch_add(1, std::memory_order_relaxed);
	uint64_t max = c.alloc_max_ns.load(std::memory_order_relaxed);
	while (ns > max && !c.alloc_max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
	}
	if (result == VK_SUCCESS) {
		std::lock_guard lock(g_memory_size_mutex);
		g_memory_sizes[*memory] = info->allocationSize;
	}
	return result;
}

VKAPI_ATTR void VKAPI_CALL TimedFreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
	const auto start = std::chrono::steady_clock::now();
	g_real_free(device, memory, callbacks);
	auto& c = g_memory_calls;
	c.frees.fetch_add(1, std::memory_order_relaxed);
	c.free_ns.fetch_add(NanosSince(start), std::memory_order_relaxed);
	{
		std::lock_guard lock(g_memory_size_mutex);
		g_memory_frees++;
		if (const auto found = g_memory_sizes.find(memory); found != g_memory_sizes.end()) {
			g_memory_freed_bytes += found->second;
			g_memory_sizes.erase(found);
		}
	}
}
} // namespace

namespace {
// A deleted image (with its memory) is kept for up to 2 s (at most 32 images, 512 MiB) and handed to the next creation
// with the same create info instead of destroying one and creating another (the game's 1080p depth buffer and its R32/S8
// aliases are created and destroyed every frame). The contents of a new image are undefined either way (initial layout
// Undefined).
struct RecycleKey {
	VkImageCreateFlags    flags   = 0;
	VkImageType           type    = VK_IMAGE_TYPE_2D;
	VkFormat              format  = VK_FORMAT_UNDEFINED;
	VkExtent3D            extent  = {};
	uint32_t              levels  = 0;
	uint32_t              layers  = 0;
	VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
	VkImageUsageFlags     usage   = 0;
	bool operator==(const RecycleKey& o) const {
		return flags == o.flags && type == o.type && format == o.format && extent.width == o.extent.width &&
		       extent.height == o.extent.height && extent.depth == o.extent.depth && levels == o.levels &&
		       layers == o.layers && samples == o.samples && usage == o.usage;
	}
};
struct RecycledImage {
	RecycleKey                            key;
	VkImage                               image      = VK_NULL_HANDLE;
	VmaAllocation                         allocation = nullptr;
	uint64_t                              bytes      = 0;
	std::chrono::steady_clock::time_point parked;
};
constexpr size_t   RecycleMaxImages = 32;
constexpr uint64_t RecycleMaxBytes  = uint64_t {512} << 20u;
constexpr auto     RecycleMaxAge    = std::chrono::seconds(2);
std::mutex                                g_recycle_mutex;
std::vector<RecycledImage>                g_recycled; // oldest first
uint64_t                                  g_recycled_bytes = 0;
std::unordered_map<VkImage, RecycleKey>   g_recyclable; // live images created through CreateImage with a plain create info
std::atomic<uint64_t>                     g_recycle_hits {0}, g_recycle_parks {0}, g_recycle_destroys {0};

// Only plain create infos (no pNext chain, exclusive, optimal tiling, undefined initial layout) are recycled.
bool RecycleKeyOf(const vk::ImageCreateInfo& info, RecycleKey& key) {
	if (info.pNext != nullptr || info.tiling != vk::ImageTiling::eOptimal ||
	    info.sharingMode != vk::SharingMode::eExclusive || info.initialLayout != vk::ImageLayout::eUndefined) {
		return false;
	}
	key.flags   = static_cast<VkImageCreateFlags>(info.flags);
	key.type    = static_cast<VkImageType>(info.imageType);
	key.format  = static_cast<VkFormat>(info.format);
	key.extent  = info.extent;
	key.levels  = info.mipLevels;
	key.layers  = info.arrayLayers;
	key.samples = static_cast<VkSampleCountFlagBits>(info.samples);
	key.usage   = static_cast<VkImageUsageFlags>(info.usage);
	return true;
}

// Destroys parked images past the age or count/byte limits (under g_recycle_mutex); `all` destroys every one.
void TrimRecycled(VmaAllocator allocator, bool all) {
	const auto now = std::chrono::steady_clock::now();
	size_t     drop = 0;
	uint64_t   bytes = g_recycled_bytes;
	while (drop < g_recycled.size() &&
	       (all || now - g_recycled[drop].parked > RecycleMaxAge || g_recycled.size() - drop > RecycleMaxImages ||
	        bytes > RecycleMaxBytes)) {
		bytes -= g_recycled[drop].bytes;
		drop++;
	}
	if (drop == 0) return;
	// A command in the ring may still name these images.
	CommandRecorder::DrainBeforeDestroy();
	for (size_t i = 0; i < drop; i++) {
		vmaDestroyImage(allocator, g_recycled[i].image, g_recycled[i].allocation);
		g_recycle_destroys.fetch_add(1, std::memory_order_relaxed);
	}
	g_recycled.erase(g_recycled.begin(), g_recycled.begin() + static_cast<std::ptrdiff_t>(drop));
	g_recycled_bytes = bytes;
}
} // namespace

std::string TakeDeviceMemoryCallStats() {
	auto& c = g_memory_calls;
	char  text[256];
	std::snprintf(text, sizeof(text),
	              "allocs=%llu alloc_MiB=%llu alloc_ms=%.1f alloc_max_ms=%.2f big_allocs=%llu frees=%llu free_ms=%.1f",
	              static_cast<unsigned long long>(c.allocs.exchange(0)),
	              static_cast<unsigned long long>(c.alloc_bytes.exchange(0) >> 20u),
	              static_cast<double>(c.alloc_ns.exchange(0)) / 1e6, static_cast<double>(c.alloc_max_ns.exchange(0)) / 1e6,
	              static_cast<unsigned long long>(c.big_allocs.exchange(0)), static_cast<unsigned long long>(c.frees.exchange(0)),
	              static_cast<double>(c.free_ns.exchange(0)) / 1e6);
	std::string line = text;
	std::snprintf(text, sizeof(text), " image_creates=%llu image_deletes=%llu image_allocs=%llu top[",
	              static_cast<unsigned long long>(g_image_creates.exchange(0)),
	              static_cast<unsigned long long>(g_image_deletes.exchange(0)),
	              static_cast<unsigned long long>(g_image_alloc_calls.exchange(0)));
	line += text;
	std::vector<std::pair<uint64_t, std::string>> top;
	{
		std::lock_guard lock(g_image_alloc_mutex);
		for (const auto& [key, n]: g_image_allocs) top.emplace_back(n, key);
		g_image_allocs.clear();
	}
	std::sort(top.begin(), top.end(), std::greater<>());
	for (size_t i = 0; i < top.size() && i < 8u; ++i) {
		line += (i == 0 ? "" : " ") + top[i].second + "x" + std::to_string(top[i].first);
	}
	line += "]";
	{
		std::lock_guard lock(g_recycle_mutex);
		std::snprintf(text, sizeof(text), " recycle hits=%llu parks=%llu destroys=%llu parked=%zu/%lluMiB",
		              static_cast<unsigned long long>(g_recycle_hits.exchange(0)),
		              static_cast<unsigned long long>(g_recycle_parks.exchange(0)),
		              static_cast<unsigned long long>(g_recycle_destroys.exchange(0)), g_recycled.size(),
		              static_cast<unsigned long long>(g_recycled_bytes >> 20u));
		line += text;
	}
	return line;
}

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
	g_real_allocate                 = VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory;
	g_real_free                     = VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory;
	if (g_real_allocate != nullptr && g_real_free != nullptr) {
		functions.vkAllocateMemory = TimedAllocateMemory;
		functions.vkFreeMemory     = TimedFreeMemory;
	}

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	{
		std::lock_guard lock(g_recycle_mutex);
		TrimRecycled(allocator, true);
		g_recyclable.clear();
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

void GraphicContext::GetDeviceLocalMemoryStats(uint64_t& usage, uint64_t& budget, uint64_t& blocks,
                                               uint64_t& allocated) const {
	usage = budget = blocks = allocated = 0;
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto& properties = GetPhysicalDeviceMemoryProperties();
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		if (!(properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal)) continue;
		usage += budgets[i].usage;
		budget += budgets[i].budget;
		blocks += budgets[i].statistics.blockBytes;
		allocated += budgets[i].statistics.allocationBytes;
	}
}

void GraphicContext::GetMemoryCounts(uint64_t& device_blocks, uint64_t& device_allocations,
                                     uint64_t& host_block_bytes) const {
	device_blocks = device_allocations = host_block_bytes = 0;
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto& properties = GetPhysicalDeviceMemoryProperties();
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		if (properties.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
			device_blocks += budgets[i].statistics.blockCount;
			device_allocations += budgets[i].statistics.allocationCount;
		} else {
			host_block_bytes += budgets[i].statistics.blockBytes;
		}
	}
}

void GraphicContext::GetVramBudgetState(uint64_t& target, uint64_t& effective) const {
	uint64_t usage = 0, budget = 0, blocks = 0, allocated = 0;
	GetDeviceLocalMemoryStats(usage, budget, blocks, allocated);
	constexpr uint64_t GiB   = 1024ull * 1024 * 1024;
	const uint64_t     cap   = VramBudgetCap();
	const uint64_t     limit = budget > 2 * GiB ? budget - GiB : budget / 2;
	target                   = budget == 0 ? 0 : cap != 0 ? std::min(limit, cap) : limit;
	// Free space inside VMA blocks is reused by new allocations before any block is added: eviction cannot
	// lower it, and counting it would make the collectors evict for nothing.
	const uint64_t empty = blocks > allocated ? blocks - allocated : 0;
	effective            = usage > empty ? usage - empty : 0;
}


uint64_t GraphicContext::GetAllocationSize(VmaAllocation allocation) const {
	if (allocator == nullptr || allocation == nullptr) {
		return 0;
	}
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, allocation, &info);
	return info.size;
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	const uint64_t cap = VramBudgetCap();
	if (discrete) {
		const auto result = budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
		return cap != 0 ? std::min(result, cap) : result;
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image,
                                 const ImageRelocation* relocation) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (relocation != nullptr) {
		// Texture relocation: the old image's memory type (AUTO_PREFER_DEVICE may pick the BAR type), the fullest block with room
		// (VMA walks its blocks fullest first), no new block unless the batch allows one, and never a parked image (its
		// allocation may sit in a block being emptied).
		alloc_info.flags |= VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT;
		if (!relocation->allow_new_block) alloc_info.flags |= VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
		alloc_info.memoryTypeBits = 1u << relocation->memory_type;
	}

	vk::Image::CType native_image = VK_NULL_HANDLE;
	const uint64_t    allocs_before = t_allocs; // KYTY_TRACE_PERF: PERF device_memory
	RecycleKey        key {};
	const bool        recyclable = RecycleKeyOf(image_info, key);
	bool              reused     = false;
	if (recyclable && relocation == nullptr) {
		std::lock_guard lock(g_recycle_mutex);
		TrimRecycled(allocator, false);
		for (size_t i = g_recycled.size(); i-- > 0;) {
			if (g_recycled[i].key == key) {
				native_image     = g_recycled[i].image;
				image.allocation = g_recycled[i].allocation;
				g_recycled_bytes -= g_recycled[i].bytes;
				g_recycled.erase(g_recycled.begin() + static_cast<std::ptrdiff_t>(i));
				g_recycle_hits.fetch_add(1, std::memory_order_relaxed);
				reused = true;
				break;
			}
		}
	}
	if (!reused) {
		const auto result = static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
		                   &native_image, &image.allocation, nullptr));
		if (result != vk::Result::eSuccess) {
			image.image = native_image;
			if (relocation == nullptr) LogMemoryBudget(); // a relocation without room is expected (counted no_room)
			return false;
		}
	}
	image.image = native_image;
	if (recyclable) {
		std::lock_guard lock(g_recycle_mutex);
		g_recyclable[native_image] = key;
	}

	g_image_creates.fetch_add(1, std::memory_order_relaxed);
	if (t_allocs != allocs_before) {
		g_image_alloc_calls.fetch_add(1, std::memory_order_relaxed);
		char key[128];
		std::snprintf(key, sizeof(key), "%s/%ux%ux%u/l%u/m%u/u%x/f%x", vk::to_string(image_info.format).c_str(),
		              image_info.extent.width, image_info.extent.height, image_info.extent.depth, image_info.arrayLayers,
		              image_info.mipLevels, static_cast<uint32_t>(image_info.usage), static_cast<uint32_t>(image_info.flags));
		std::lock_guard lock(g_image_alloc_mutex);
		g_image_allocs[key]++;
	}
	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	g_image_deletes.fetch_add(1, std::memory_order_relaxed);
	{
		// Parked, not destroyed (no drain: the image stays valid).
		std::lock_guard lock(g_recycle_mutex);
		const auto      found = g_recyclable.find(static_cast<VkImage>(image.image));
		if (found != g_recyclable.end()) {
			VmaAllocationInfo info {};
			vmaGetAllocationInfo(allocator, image.allocation, &info);
			g_recycled.push_back({found->second, static_cast<VkImage>(image.image), image.allocation, info.size,
			                      std::chrono::steady_clock::now()});
			g_recycled_bytes += info.size;
			g_recyclable.erase(found);
			g_recycle_parks.fetch_add(1, std::memory_order_relaxed);
			TrimRecycled(allocator, false);
			image.image      = nullptr;
			image.allocation = nullptr;
			return;
		}
	}
	// A command in the ring may still name this image (VMA does not go through the
	// dispatcher's destroy functions).
	CommandRecorder::DrainBeforeDestroy();
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

void GraphicContext::DestroyImageNow(VulkanImage& image) {
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);
	g_image_deletes.fetch_add(1, std::memory_order_relaxed);
	{
		std::lock_guard lock(g_recycle_mutex);
		g_recyclable.erase(static_cast<VkImage>(image.image));
	}
	CommandRecorder::DrainBeforeDestroy(); // as DeleteImage: VMA does not go through the dispatcher
	vmaDestroyImage(allocator, image.image, image.allocation);
	image.image      = nullptr;
	image.allocation = nullptr;
}

void GraphicContext::TrimRecycledImages() {
	if (allocator == nullptr) return;
	std::lock_guard lock(g_recycle_mutex);
	TrimRecycled(allocator, true);
}

std::vector<GraphicContext::BlockCensus> GraphicContext::CollectBlocks(uint32_t list_pct, bool ranges) const {
	std::vector<BlockCensus> blocks;
	if (allocator == nullptr) return blocks;
	const auto& properties = GetPhysicalDeviceMemoryProperties();
	for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
		const auto flags = properties.memoryTypes[type].propertyFlags;
		if (!(flags & vk::MemoryPropertyFlagBits::eDeviceLocal) || (flags & vk::MemoryPropertyFlagBits::eHostVisible)) continue;
		VmaBlockVector* vector = allocator->m_pBlockVectors[type];
		if (vector == nullptr) continue;
		// The default pool's block list (VMA has no public API for it; the defragmentation code walks it the same way).
		VmaMutexLockRead lock(vector->GetMutex(), allocator->m_UseMutex);
		for (size_t i = 0; i < vector->GetBlockCount(); i++) {
			const VmaDeviceMemoryBlock* block    = vector->GetBlock(i);
			const VmaBlockMetadata*     metadata = block->m_pMetadata;
			auto&                       entry    = blocks.emplace_back();
			entry.memory                         = block->GetDeviceMemory();
			entry.type                           = type;
			entry.size                           = metadata->GetSize();
			entry.free                           = metadata->GetSumFreeSize();
			entry.count                          = metadata->GetAllocationCount();
			const bool list = entry.free * 100u >= entry.size * list_pct;
			if (!list && !ranges) continue;
			// TLSF walks its allocations from the end of the block to the start; GetNextFreeRegionSize is the free range
			// physically before each one, and what the walk leaves of the free sum is the free tail of the block.
			uint64_t listed_free = 0;
			for (VmaAllocHandle handle = metadata->GetAllocationListBegin(); handle != VK_NULL_HANDLE;
			     handle                = metadata->GetNextAllocation(handle)) {
				if (list) entry.allocations.push_back(static_cast<VmaAllocation>(metadata->GetAllocationUserData(handle)));
				if (!ranges) continue;
				if (const uint64_t gap = metadata->GetNextFreeRegionSize(handle); gap != 0) {
					entry.free_ranges.push_back(gap);
					listed_free += gap;
				}
			}
			if (ranges && entry.free > listed_free) entry.free_ranges.push_back(entry.free - listed_free);
		}
	}
	return blocks;
}

std::vector<VmaAllocation> GraphicContext::RecycledAllocations() const {
	std::vector<VmaAllocation> allocations;
	std::lock_guard            lock(g_recycle_mutex);
	for (const auto& parked: g_recycled) allocations.push_back(parked.allocation);
	return allocations;
}

std::vector<VmaAllocation> GraphicContext::PlugBlocks(uint32_t type, std::span<const VkDeviceMemory> avoid,
                                                      uint64_t start_size, uint64_t min_size) {
	std::vector<VmaAllocation> plugs;
	if (allocator == nullptr || avoid.empty()) return plugs;
	VmaAllocationCreateInfo info {};
	// MIN_TIME walks the blocks emptiest first: the blocks to avoid are tried before the others.
	info.flags          = VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT | VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;
	info.memoryTypeBits = 1u << type;
	constexpr uint64_t Alignment = 4096;
	for (uint64_t size = Common::AlignDown(start_size, Alignment); size >= std::max(min_size, Alignment); size /= 2u) {
		size = Common::AlignDown(size, Alignment);
		for (;;) {
			const VkMemoryRequirements requirements {size, Alignment, 1u << type};
			VmaAllocation              plug = nullptr;
			VmaAllocationInfo          placed {};
			if (vmaAllocateMemory(allocator, &requirements, &info, &plug, &placed) != VK_SUCCESS) break;
			if (std::find(avoid.begin(), avoid.end(), placed.deviceMemory) == avoid.end()) {
				vmaFreeMemory(allocator, plug); // room elsewhere at this size: the destinations may take it
				break;
			}
			plugs.push_back(plug);
		}
	}
	return plugs;
}

void GraphicContext::FreePlugs(std::vector<VmaAllocation>& plugs) {
	if (!plugs.empty()) vmaFreeMemoryPages(allocator, plugs.size(), plugs.data());
	plugs.clear();
}

bool GraphicContext::AllocationPlace(VmaAllocation allocation, VkDeviceMemory& memory, uint32_t& type) const {
	if (allocator == nullptr || allocation == nullptr) return false;
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, allocation, &info);
	memory = info.deviceMemory;
	type   = info.memoryType;
	return true;
}

void GraphicContext::DeviceMemoryFrees(uint64_t& count, uint64_t& bytes) {
	std::lock_guard lock(g_memory_size_mutex);
	count = g_memory_frees;
	bytes = g_memory_freed_bytes;
}

} // namespace Libs::Graphics
