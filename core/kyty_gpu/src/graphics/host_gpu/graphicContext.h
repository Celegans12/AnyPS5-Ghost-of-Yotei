#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include <string>
#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <atomic>
#include <map>
#include <mutex>
#include <span>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

// KYTY_VRAM_GC_FIX=1 (default off, logged once): the texture collector scans past the images it cannot free
// instead of stopping at them, and downloads GPU-written color images to guest memory before it frees them
// (TextureCache::CollectGarbageFixed).
[[nodiscard]] bool VramGcFixEnabled();
// KYTY_VRAM_BUDGET=1 (default off, logged once): the caches keep the device-local memory they hold under a target
// re-read at every collector run (GraphicContext::GetVramBudgetState), evicting old images before new ones are
// created over it; implies the KYTY_VRAM_GC_FIX collector.
[[nodiscard]] bool VramBudgetEnabled();
// KYTY_TRACE_PERF: the allocator's vkAllocateMemory / vkFreeMemory calls since the last call (counts, MiB, ms).
[[nodiscard]] std::string TakeDeviceMemoryCallStats();

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	// KYTY_TRACE_PIPELINE_STATS: VK_KHR_pipeline_executable_properties is enabled.
	bool                               pipeline_stats_enabled                = false;
	bool                               compute_subgroup_size_control_enabled = false;
	bool                               sample_rate_shading_enabled           = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	// KYTY_FIX_DCC_GPU_CLEAR_CHECK: VK_EXT_conditional_rendering is enabled.
	bool                               conditional_rendering_enabled         = false;
	// bool fp64_denorm_preserve = false; // Temporarily disabled.
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               maintenance8_enabled                  = false;
	bool                               robust_buffer_access_enabled          = false;
	bool                               occlusion_queries_enabled             = false;
	bool                               occlusion_query_precise_enabled       = false;
	bool                               supports_block_texel_view              = false;
	bool                                      mesh_shader_enabled                   = false;
	// GPU indirect draws: multiDrawIndirect, drawIndirectFirstInstance, drawIndirectCount and
	// shaderDrawParameters are enabled, so indirect draws can run from GPU memory.
	bool                               gpu_indirect_draw_enabled             = false;
	// KYTY_FIX_OCCLUSION_QUERIES=1: occlusionQueryPrecise is enabled, so host occlusion queries count exact samples.
	bool                               occlusion_query_precise               = false;
	// Descriptor indexing for bindless images (BindlessTable): runtime arrays indexed
	// non-uniformly, partially bound and updated after bind. From KytyPS5 PR #937.
	bool                               bindless_enabled                      = false;
	uint32_t                           bindless_max_sampled_images           = 0;
	// The device's update-after-bind sampled image limits: per pipeline layout and per stage (bindless_max_sampled_images
	// is the smaller one).
	uint32_t                           bindless_set_sampled_images_limit     = 0;
	uint32_t                           bindless_stage_sampled_images_limit   = 0;
	uint32_t                           bindless_max_samplers                 = 0;
	vk::DescriptorSetLayout            bindless_layout                       = nullptr;
	vk::DescriptorSet                  bindless_set                          = nullptr;
	// Frames presented so far (bindless feedback is resolved at most once per frame).
	std::atomic<uint64_t>              presented_frames {0};
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	uint32_t                           device_clock_period_ps                = 0;
	vk::DescriptorSetLayout            image_heap_layout                     = nullptr;
	vk::ShaderStageFlags               required_subgroup_size_stages         = {};
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;

	[[nodiscard]] const vk::PhysicalDeviceProperties& GetPhysicalDeviceProperties() const {
		return physical_device_properties;
	}

	[[nodiscard]] const vk::PhysicalDeviceMemoryProperties&
	GetPhysicalDeviceMemoryProperties() const {
		return physical_device_memory_properties;
	}

	[[nodiscard]] vk::FormatProperties GetFormatProperties(vk::Format format) const {
		std::scoped_lock lock(m_format_properties_mutex);
		auto [it, inserted] = m_format_properties.try_emplace(format);
		if (inserted) {
			physical_device.getFormatProperties(format, &it->second);
		}
		return it->second;
	}

	[[nodiscard]] vk::Result GetImageFormatProperties(vk::Format format, vk::ImageType type,
	                                                  vk::ImageTiling            tiling,
	                                                  vk::ImageUsageFlags        usage,
	                                                  vk::ImageCreateFlags       flags,
	                                                  vk::ImageFormatProperties* properties) const {
		using Key = std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
		                       vk::ImageCreateFlags>;
		std::scoped_lock lock(m_image_format_properties_mutex);
		auto [it, inserted] =
		    m_image_format_properties.try_emplace(Key {format, type, tiling, usage, flags});
		if (inserted) {
			it->second.first = physical_device.getImageFormatProperties(format, type, tiling, usage,
			                                                            flags, &it->second.second);
		}
		if (properties != nullptr) {
			*properties = it->second.second;
		}
		return it->second.first;
	}

	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		return subgroup_size == 64u || compute_subgroup_size_control_enabled;
	}

	[[nodiscard]] vk::DeviceSize StorageMinAlignment() const {
		const auto alignment = physical_device_properties.limits.minStorageBufferOffsetAlignment;
		return alignment != 0 ? alignment : 1;
	}

	[[nodiscard]] bool CreateAllocator();
	void               DestroyAllocator();
	void               LogMemoryBudget() const;
	[[nodiscard]] bool CanReportMemoryUsage() const noexcept { return memory_budget_ext_enabled; }
	[[nodiscard]] uint64_t GetDeviceMemoryUsage() const;
	// Device-local heaps: driver-reported usage and budget of this process, VMA block and allocation bytes.
	void GetDeviceLocalMemoryStats(uint64_t& usage, uint64_t& budget, uint64_t& blocks, uint64_t& allocated) const;
	// KYTY_TRACE_MEM_CENSUS: VkDeviceMemory blocks and VMA allocations in the device-local heaps, and VMA block
	// bytes in the other (host) heaps.
	void GetMemoryCounts(uint64_t& device_blocks, uint64_t& device_allocations, uint64_t& host_block_bytes) const;
	// Bytes of one VMA allocation (0 for none).
	[[nodiscard]] uint64_t GetAllocationSize(VmaAllocation allocation) const;
	// KYTY_VRAM_BUDGET: target = min(KYTY_VRAM_BUDGET_MB cap, driver budget - 1 GiB) for the device-local heaps, read
	// now (the driver budget moves with other processes; 0 when it reports none), and effective = the driver-reported
	// usage minus the free space inside VMA blocks (what eviction can lower).
	void GetVramBudgetState(uint64_t& target, uint64_t& effective) const;
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;

	struct HeapUsage {
		uint64_t usage        = 0; // Whole process, as reported by VK_EXT_memory_budget.
		uint64_t budget       = 0;
		uint32_t allocations  = 0;
		bool     device_local = false;
	};
	[[nodiscard]] uint32_t GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const;
	// Texture relocation: the destination of a moved texture (CreateImage): the memory type of the old one, and whether the
	// allocation may add a block (else it only takes room in existing blocks).
	struct ImageRelocation {
		uint32_t memory_type     = 0;
		bool     allow_new_block = false;
	};
	[[nodiscard]] bool CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image,
	                               const ImageRelocation* relocation = nullptr);
	void               DeleteImage(VulkanImage& image);
	// Texture relocation: destroys the image and frees its allocation now (never parked for reuse: its block must empty).
	void DestroyImageNow(VulkanImage& image);
	// Texture relocation: destroys every parked image (they hold allocations in the blocks being emptied).
	void TrimRecycledImages();
	// Texture relocation: one VMA block of a device-local, not host-visible memory type, read from VMA's block lists.
	struct BlockCensus {
		VkDeviceMemory             memory = VK_NULL_HANDLE;
		uint32_t                   type   = 0;
		uint64_t                   size   = 0;
		uint64_t                   free   = 0;
		uint64_t                   count  = 0; // allocations in the block
		std::vector<VmaAllocation> allocations;
		std::vector<uint64_t>      free_ranges; // with `ranges`: the sizes of its free ranges
	};
	// list_pct: only blocks at least that empty (percent) list their allocations (the walk over every allocation is
	// most of the census); ranges: every block lists its free ranges (the room check before a move).
	[[nodiscard]] std::vector<BlockCensus> CollectBlocks(uint32_t list_pct = 0, bool ranges = false) const;
	// Texture relocation: the allocations of parked images (they go within 2 s, or at TrimRecycledImages).
	[[nodiscard]] std::vector<VmaAllocation> RecycledAllocations() const;
	// Texture relocation: VMA places a new allocation in the first block (fullest first) with a free range it fits, which for a
	// large texture is often the emptiest block - the one being emptied, or the empty block VMA keeps. Plugs fill the free
	// ranges of those blocks (memory type `type`, `avoid`) with raw allocations, largest first by halving sizes down to
	// `min_size`, so the destinations go elsewhere; a plug that lands in another block is freed at once. FreePlugs after
	// the destinations exist (plugs are never bound or used).
	[[nodiscard]] std::vector<VmaAllocation> PlugBlocks(uint32_t type, std::span<const VkDeviceMemory> avoid,
	                                                    uint64_t start_size, uint64_t min_size);
	void FreePlugs(std::vector<VmaAllocation>& plugs);
	// The device memory and memory type an allocation lives in (false without one).
	[[nodiscard]] bool AllocationPlace(VmaAllocation allocation, VkDeviceMemory& memory, uint32_t& type) const;
	// Texture relocation: vkFreeMemory calls of the allocator since start and the bytes they freed (counted only with the switch).
	static void DeviceMemoryFrees(uint64_t& count, uint64_t& bytes);

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	mutable std::mutex                                 m_format_properties_mutex;
	mutable std::map<vk::Format, vk::FormatProperties> m_format_properties;
	mutable std::mutex                                 m_image_format_properties_mutex;
	mutable std::map<std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
	                            vk::ImageCreateFlags>,
	                 std::pair<vk::Result, vk::ImageFormatProperties>>
	    m_image_format_properties;
};

struct VulkanImageState {
	vk::PipelineStageFlags2 pl_stage    = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        access_mask = vk::AccessFlagBits2::eNone;
	vk::ImageLayout         layout      = vk::ImageLayout::eUndefined;
};

struct VulkanImage {
	VulkanImage() = default;
	KYTY_CLASS_NO_COPY(VulkanImage);

	vk::Format                    format      = vk::Format::eUndefined;
	vk::ImageType                 image_type  = vk::ImageType::e2D;
	vk::Extent3D                  extent      = {1, 1, 1};
	uint32_t                      layers      = 1;
	uint32_t                      mip_levels  = 1;
	uint32_t                      samples     = 1;
	vk::ImageUsageFlags           usage       = {};
	vk::ImageCreateFlags          flags       = {};
	vk::Image                     image       = nullptr;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresource_states;
	VmaAllocation                allocation = nullptr;
};



} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
