#ifndef KYTY_SHIM_TESTS_TEST_VULKAN_DEVICE_H_
#define KYTY_SHIM_TESTS_TEST_VULKAN_DEVICE_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string_view>

namespace KytyShimTests {

struct TestVulkanDevice {
	vk::detail::DynamicLoader loader;
	Libs::Graphics::GraphicContext graphics;

	const char* Initialize(bool require_tiling = false, bool sparse = false) {
		if (!loader.success()) {
			return "Vulkan loader unavailable";
		}
		const auto get_instance = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
		if (get_instance == nullptr) {
			return "Vulkan loader unavailable";
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance);
		const vk::ApplicationInfo app {.pApplicationName = "Kyty shim test",
		                               .apiVersion = Libs::Graphics::VULKAN_TARGET_API_VERSION};
		const vk::InstanceCreateInfo instance_info {.pApplicationInfo = &app};
		if (vk::createInstance(&instance_info, nullptr, &graphics.instance) != vk::Result::eSuccess) {
			return "could not create a Vulkan 1.3 instance";
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(graphics.instance);
		const auto devices = graphics.instance.enumeratePhysicalDevices();
		if (devices.result != vk::Result::eSuccess) {
			return "Vulkan device enumeration failed";
		}
		for (const auto device : devices.value) {
			if (InitializeDevice(device, require_tiling, sparse)) {
				return nullptr;
			}
		}
		return sparse ? "no Vulkan device supports the scheduler fixture with sparse buffers"
		              : "no Vulkan device supports the scheduler fixture";
	}

	~TestVulkanDevice() {
		if (graphics.allocator != nullptr) {
			graphics.DestroyAllocator();
		}
		if (graphics.device != nullptr) {
			graphics.device.destroy();
		}
		if (graphics.instance != nullptr) {
			graphics.instance.destroy();
		}
	}

private:
	bool InitializeDevice(vk::PhysicalDevice physical, bool require_tiling, bool sparse) {
		vk::PhysicalDeviceVulkan12Properties limits {};
		vk::PhysicalDeviceProperties2 properties {.pNext = &limits};
		physical.getProperties2(&properties);
		if (properties.properties.apiVersion < Libs::Graphics::VULKAN_TARGET_API_VERSION ||
		    std::min({limits.maxDescriptorSetUpdateAfterBindSampledImages,
		              limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
		              limits.maxPerStageUpdateAfterBindResources}) < Libs::Graphics::ShaderRecompiler::IR::ImageHeapCapacity) {
			return false;
		}
		const auto families = physical.getQueueFamilyProperties();
		const auto queue_flags = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
		const auto family = std::ranges::find_if(families, [queue_flags](const auto& properties) {
			return (properties.queueFlags & queue_flags) == queue_flags;
		});
		if (family == families.end()) {
			return false;
		}
		const auto extensions = physical.enumerateDeviceExtensionProperties(nullptr);
		const std::array required_extensions {
		    VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME};
		if (extensions.result != vk::Result::eSuccess ||
		    !std::ranges::all_of(required_extensions, [&extensions](const char* extension) {
			    return std::ranges::any_of(extensions.value, [extension](const auto& properties) {
				    return std::string_view(properties.extensionName.data()) == extension;
			    });
		    })) {
			return false;
		}
		vk::PhysicalDeviceRobustness2FeaturesEXT supported_robustness {};
		vk::PhysicalDeviceVulkan12Features supported12 {.pNext = &supported_robustness};
		vk::PhysicalDeviceVulkan11Features supported11 {.pNext = &supported12};
		vk::PhysicalDeviceVulkan13Features supported13 {.pNext = &supported11};
		vk::PhysicalDeviceFeatures2 supported {.pNext = &supported13};
		physical.getFeatures2(&supported);
		if (sparse && (!supported.features.sparseBinding || !supported.features.sparseResidencyBuffer)) {
			return false;
		}
		if (!supported.features.shaderInt64 || !supported.features.sampleRateShading ||
		    !supported_robustness.nullDescriptor ||
		    (require_tiling && (!supported11.storageBuffer16BitAccess ||
		                        !supported12.storageBuffer8BitAccess)) ||
		    !supported12.descriptorBindingSampledImageUpdateAfterBind ||
		    !supported12.descriptorBindingUpdateUnusedWhilePending ||
		    !supported12.descriptorBindingPartiallyBound || !supported12.hostQueryReset ||
		    !supported12.timelineSemaphore || !supported12.bufferDeviceAddress ||
		    !supported13.synchronization2 || !supported13.dynamicRendering) {
			return false;
		}
		vk::PhysicalDeviceRobustness2FeaturesEXT robustness {.nullDescriptor = VK_TRUE};
		vk::PhysicalDeviceVulkan12Features features12 {
		    .pNext = &robustness,
		    .storageBuffer8BitAccess = require_tiling,
		    .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
		    .descriptorBindingUpdateUnusedWhilePending = VK_TRUE,
		    .descriptorBindingPartiallyBound = VK_TRUE,
		    .hostQueryReset = VK_TRUE,
		    .timelineSemaphore = VK_TRUE,
		    .bufferDeviceAddress = VK_TRUE};
		vk::PhysicalDeviceVulkan11Features features11 {
		    .pNext = &features12, .storageBuffer16BitAccess = require_tiling};
		const vk::PhysicalDeviceVulkan13Features features13 {
		    .pNext = &features11, .synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE};
		vk::PhysicalDeviceFeatures core_features {
		    .sampleRateShading = VK_TRUE, .shaderInt64 = VK_TRUE};
		core_features.sparseBinding         = sparse ? VK_TRUE : VK_FALSE;
		core_features.sparseResidencyBuffer = sparse ? VK_TRUE : VK_FALSE;
		const std::array<float, 2> priorities {1.0f, 1.0f};
		const auto main_family = static_cast<uint32_t>(family - families.begin());
		std::array<vk::DeviceQueueCreateInfo, 3> queue_infos {};
		queue_infos[0] = vk::DeviceQueueCreateInfo {.queueFamilyIndex = main_family, .queueCount = 1, .pQueuePriorities = priorities.data()};
		uint32_t queue_count = 1;
		uint32_t readback_family = static_cast<uint32_t>(-1);
		uint32_t sparse_family = static_cast<uint32_t>(-1);
		uint32_t sparse_index = 0;
		if (sparse) {
			for (uint32_t index = 0; index < families.size(); index++) {
				const auto flags = families[index].queueFlags;
				if (index != main_family && families[index].queueCount != 0 && (flags & vk::QueueFlagBits::eTransfer) &&
				    !(flags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))) {
					readback_family = index;
					queue_infos[queue_count++] = vk::DeviceQueueCreateInfo {
					    .queueFamilyIndex = index, .queueCount = 1, .pQueuePriorities = priorities.data()};
					break;
				}
			}
			for (uint32_t index = 0; index < families.size(); index++) {
				if (index != main_family && index != readback_family && families[index].queueCount != 0 &&
				    (families[index].queueFlags & vk::QueueFlagBits::eSparseBinding)) {
					sparse_family = index;
					queue_infos[queue_count++] = vk::DeviceQueueCreateInfo {
					    .queueFamilyIndex = index, .queueCount = 1, .pQueuePriorities = priorities.data()};
					break;
				}
			}
			if (sparse_family == static_cast<uint32_t>(-1)) {
				if (!(families[main_family].queueFlags & vk::QueueFlagBits::eSparseBinding)) {
					return false;
				}
				sparse_family = main_family;
				if (families[main_family].queueCount >= 2) {
					queue_infos[0].queueCount = 2;
					sparse_index = 1;
				}
			}
		}
		const vk::DeviceCreateInfo device_info {
		    .pNext = &features13, .queueCreateInfoCount = queue_count, .pQueueCreateInfos = queue_infos.data(),
		    .enabledExtensionCount = static_cast<uint32_t>(required_extensions.size()),
		    .ppEnabledExtensionNames = required_extensions.data(),
		    .pEnabledFeatures = &core_features};
		if (physical.createDevice(&device_info, nullptr, &graphics.device) != vk::Result::eSuccess) {
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(graphics.device);
		graphics.physical_device = physical;
		graphics.physical_device_properties = properties.properties;
		graphics.physical_device_memory_properties = physical.getMemoryProperties();
		graphics.queue_family = main_family;
		graphics.device.getQueue(graphics.queue_family, 0, &graphics.queue);
		graphics.readback_family = readback_family;
		if (readback_family != static_cast<uint32_t>(-1)) {
			graphics.device.getQueue(readback_family, 0, &graphics.readback_queue);
		}
		graphics.sparse_buffers_enabled = sparse;
		graphics.sparse_family = sparse_family;
		graphics.sparse_queue_index = sparse_index;
		if (sparse) {
			graphics.device.getQueue(sparse_family, sparse_index, &graphics.sparse_queue);
		}
		graphics.occlusion_queries_enabled = true;
		return true;
	}
};

}

#endif
