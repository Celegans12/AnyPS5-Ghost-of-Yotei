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

	const char* Initialize(bool require_tiling = false) {
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
			if (InitializeDevice(device, require_tiling)) {
				return nullptr;
			}
		}
		return "no Vulkan device supports the scheduler fixture";
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
	bool InitializeDevice(vk::PhysicalDevice physical, bool require_tiling) {
		vk::PhysicalDeviceVulkan12Properties limits {};
		vk::PhysicalDeviceVulkan11Properties properties11 {.pNext = &limits};
		vk::PhysicalDeviceProperties2 properties {.pNext = &properties11};
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
		const vk::PhysicalDeviceFeatures core_features {
		    .sampleRateShading = VK_TRUE,
		    .shaderStorageImageMultisample = supported.features.shaderStorageImageMultisample,
		    .shaderInt64 = VK_TRUE};
		const float priority = 1.0f;
		const vk::DeviceQueueCreateInfo queue_info {
		    .queueFamilyIndex = static_cast<uint32_t>(family - families.begin()),
		    .queueCount = 1, .pQueuePriorities = &priority};
		const vk::DeviceCreateInfo device_info {
		    .pNext = &features13, .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
		    .enabledExtensionCount = static_cast<uint32_t>(required_extensions.size()),
		    .ppEnabledExtensionNames = required_extensions.data(),
		    .pEnabledFeatures = &core_features};
		if (physical.createDevice(&device_info, nullptr, &graphics.device) != vk::Result::eSuccess) {
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(graphics.device);
		graphics.physical_device = physical;
		graphics.physical_device_properties = properties.properties;
		graphics.subgroup_supported_stages = properties11.subgroupSupportedStages;
		graphics.subgroup_supported_operations = properties11.subgroupSupportedOperations;
		if (!core_features.shaderStorageImageMultisample) {
			graphics.physical_device_properties.limits.storageImageSampleCounts = vk::SampleCountFlagBits::e1;
		}
		graphics.physical_device_memory_properties = physical.getMemoryProperties();
		graphics.queue_family = queue_info.queueFamilyIndex;
		graphics.device.getQueue(graphics.queue_family, 0, &graphics.queue);
		graphics.occlusion_queries_enabled = true;
		return true;
	}
};

}

#endif
