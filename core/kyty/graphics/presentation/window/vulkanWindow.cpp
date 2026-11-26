#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#endif
#include "graphics/host_gpu/renderer/upscaler/dlss.h"
#include "graphics/host_gpu/renderer/upscaler/fsr.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/presentation/window/windowInternal.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"
#include "libs/controller.h"
#include "loader/systemContent.h"

#if !defined(ANYPS5_KYTY)
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <vector>
#include <vulkan/vk_platform.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace Libs::Graphics {

struct VulkanExtensions {
	bool enable_validation_layers = false;

	std::vector<const char*>             required_extensions;
	std::vector<vk::ExtensionProperties> available_extensions;
	std::vector<vk::LayerProperties>     available_layers;
};

static constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

vk::PhysicalDeviceVulkan11Features WindowContext::RequiredVulkan11Features() noexcept {
	return {.storageBuffer16BitAccess = VK_TRUE, .shaderDrawParameters = VK_TRUE};
}

vk::PhysicalDeviceVulkan12Features WindowContext::RequiredVulkan12Features() noexcept {
	return {.samplerMirrorClampToEdge                     = VK_TRUE,
	        .storageBuffer8BitAccess                      = VK_TRUE,
	        .shaderBufferInt64Atomics                     = VK_TRUE,
	        .shaderSampledImageArrayNonUniformIndexing    = VK_TRUE,
	        .shaderStorageBufferArrayNonUniformIndexing   = VK_TRUE,
	        .shaderStorageImageArrayNonUniformIndexing    = VK_TRUE,
	        .descriptorBindingSampledImageUpdateAfterBind = VK_TRUE,
	        .descriptorBindingUpdateUnusedWhilePending    = VK_TRUE,
	        .descriptorBindingPartiallyBound              = VK_TRUE,
	        .uniformBufferStandardLayout                  = VK_TRUE,
	        .timelineSemaphore                            = VK_TRUE,
	        .bufferDeviceAddress                          = VK_TRUE,
	        .shaderOutputViewportIndex                    = VK_TRUE,
	        .shaderOutputLayer                            = VK_TRUE};
}

vk::PhysicalDeviceVulkan13Features WindowContext::RequiredVulkan13Features() noexcept {
	return {.synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE};
}

static bool HasExtension(const std::vector<vk::ExtensionProperties>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const auto& ext) { return strcmp(ext.extensionName, name) == 0; });
}

static bool HasExtension(const std::vector<const char*>& extensions, const char* name) {
	return std::any_of(extensions.begin(), extensions.end(),
	                   [name](const char* ext) { return strcmp(ext, name) == 0; });
}

static bool HasLayer(const std::vector<vk::LayerProperties>& layers, const char* name) {
	return std::any_of(layers.begin(), layers.end(),
	                   [name](const auto& layer) { return strcmp(layer.layerName, name) == 0; });
}

static void GetSurfaceCapabilities(vk::PhysicalDevice physical_device, vk::SurfaceKHR surface,
                                   SurfaceCapabilities& r) {
	RequireVulkanSuccess(physical_device.getSurfaceCapabilitiesKHR(surface, &r.capabilities),
	                     "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

	r.formats = EnumerateVulkan<vk::SurfaceFormatKHR>(
	    "vkGetPhysicalDeviceSurfaceFormatsKHR", [&](uint32_t* count, vk::SurfaceFormatKHR* values) {
		    return physical_device.getSurfaceFormatsKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.formats.empty());

	r.present_modes = EnumerateVulkan<vk::PresentModeKHR>(
	    "vkGetPhysicalDeviceSurfacePresentModesKHR",
	    [&](uint32_t* count, vk::PresentModeKHR* values) {
		    return physical_device.getSurfacePresentModesKHR(surface, count, values);
	    });
	EXIT_NOT_IMPLEMENTED(r.present_modes.empty());
}

static bool CheckFormat(vk::PhysicalDevice device, vk::Format format,
                        vk::FormatFeatureFlags features) {
	return (device.getFormatProperties(format).optimalTilingFeatures & features) == features;
}

static uint32_t VulkanFindQueueFamily(vk::PhysicalDevice device, vk::SurfaceKHR surface) {
	EXIT_IF(device == nullptr);
#if !defined(ANYPS5_KYTY)
	EXIT_IF(surface == nullptr);
#endif

	uint32_t queue_family_count = 0;
	device.getQueueFamilyProperties(&queue_family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
	device.getQueueFamilyProperties(&queue_family_count, queue_families.data());

	const auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
	for (uint32_t family = 0; family < queue_family_count; family++) {
		const auto& properties             = queue_families[family];
		vk::Bool32  presentation_supported = VK_FALSE;
#if defined(ANYPS5_KYTY)
		if (surface == nullptr) {
			presentation_supported = VK_TRUE;
		} else
#endif
		RequireVulkanSuccess(device.getSurfaceSupportKHR(family, surface, &presentation_supported),
		                     "vkGetPhysicalDeviceSurfaceSupportKHR");

		LOGF("\tqueue family: %s [count = %u], [present = %s]\n",
		     vk::to_string(properties.queueFlags).c_str(), properties.queueCount,
		     (presentation_supported == VK_TRUE ? "true" : "false"));
		if (properties.queueCount != 0 && (properties.queueFlags & required) == required &&
		    presentation_supported == VK_TRUE) {
			LOGF("\tselected universal queue family %u\n", family);
			return family;
		}
	}
	return static_cast<uint32_t>(-1);
}

static void VulkanFindPhysicalDevice(vk::Instance instance, vk::SurfaceKHR surface,
                                     const std::vector<const char*>& device_extensions,
                                     vk::PhysicalDevice& out_device, uint32_t& out_queue_family,
                                     std::string& out_rejections) {
	EXIT_IF(instance == nullptr);
#if !defined(ANYPS5_KYTY)
	EXIT_IF(surface == nullptr);
#endif

	auto devices = EnumerateVulkan<vk::PhysicalDevice>(
	    "vkEnumeratePhysicalDevices", [&](uint32_t* count, vk::PhysicalDevice* values) {
		    return instance.enumeratePhysicalDevices(count, values);
	    });
	EXIT_NOT_IMPLEMENTED(devices.empty());

	const std::string requested = Common::ToLower(Config::GetGpuName());
	if (!requested.empty()) {
		std::string                     names;
		std::vector<vk::PhysicalDevice> matching;
		for (const auto& device: devices) {
			vk::PhysicalDeviceProperties device_properties {};
			device.getProperties(&device_properties);
			names += names.empty() ? "" : ", ";
			names += device_properties.deviceName.data();
			if (Common::ToLower(device_properties.deviceName.data()).find(requested) !=
			    std::string::npos) {
				matching.push_back(device);
			}
		}
		if (matching.empty()) {
			EXIT("ANYPS5_GPU=\"%s\" matches the name of no Vulkan device; devices: %s\n",
			     Config::GetGpuName().c_str(), names.c_str());
		}
		devices = std::move(matching);
	} else if (vk::PhysicalDevice headset; Vr::OpenXrPhysicalDevice(instance, headset)) {
		devices = {headset};
	}

	vk::PhysicalDevice best_device       = nullptr;
	uint32_t           best_queue_family = static_cast<uint32_t>(-1);

	for (const auto& device: devices) {
		bool skip_device = false;

		vk::PhysicalDeviceProperties device_properties {};
		device.getProperties(&device_properties);

		const auto reject = [&](const std::string& reason) {
			LOGF("%s\n", reason.c_str());
			out_rejections +=
			    skip_device ? "; " : fmt::format("\n  {}: ", device_properties.deviceName.data());
			out_rejections += reason;
			skip_device = true;
		};

		LOGF("Vulkan device: %s\n", device_properties.deviceName.data());
		if (device_properties.apiVersion < VULKAN_TARGET_API_VERSION) {
			reject(fmt::format("Vulkan {}.{} is required, but device supports only {}.{}.{}",
			                   VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION),
			                   VK_VERSION_MAJOR(device_properties.apiVersion),
			                   VK_VERSION_MINOR(device_properties.apiVersion),
			                   VK_VERSION_PATCH(device_properties.apiVersion)));
			continue;
		}

		vk::PhysicalDeviceImageViewMinLodFeaturesEXT  min_lod {};
		vk::PhysicalDeviceDepthClipEnableFeaturesEXT  clip_enable {.pNext = &min_lod};
		vk::PhysicalDeviceDepthClipControlFeaturesEXT clip_control {.pNext = &clip_enable};
		vk::PhysicalDeviceRobustness2FeaturesEXT      robustness2 {.pNext = &clip_control};
		vk::PhysicalDeviceVulkan11Features            features11 {.pNext = &robustness2};
		vk::PhysicalDeviceVulkan12Features            features12 {.pNext = &features11};
		vk::PhysicalDeviceVulkan13Features            features13 {.pNext = &features12};
		vk::PhysicalDeviceFeatures2                   device_features2 {.pNext = &features13};
		device.getFeatures2(&device_features2);

		const auto queue_family = VulkanFindQueueFamily(device, surface);
		if (queue_family == static_cast<uint32_t>(-1)) {
			reject("No universal graphics, compute, and presentation queue");
		}

		const auto check_feature = [&](vk::Bool32 supported, const char* name) {
			if (supported != VK_TRUE) {
				reject(fmt::format("{} is not supported", name));
			}
		};

		check_feature(min_lod.minLod, "image view minLod");

		check_feature(clip_control.depthClipControl, "depthClipControl");
#if defined(__APPLE__)
		if (clip_enable.depthClipEnable != VK_TRUE) {
			LOGF("depthClipEnable is not supported\n");
		}
#else
		check_feature(clip_enable.depthClipEnable, "depthClipEnable");
#endif
#if !defined(__APPLE__)
		check_feature(device_features2.features.depthClamp, "depthClamp");
#endif

		check_feature(features12.samplerMirrorClampToEdge, "samplerMirrorClampToEdge");
		check_feature(features11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
		check_feature(features11.shaderDrawParameters, "shaderDrawParameters");
		check_feature(features12.storageBuffer8BitAccess, "storageBuffer8BitAccess");
		check_feature(features12.timelineSemaphore, "timelineSemaphore");
		check_feature(features12.shaderOutputLayer, "shaderOutputLayer");
		check_feature(features12.shaderOutputViewportIndex, "shaderOutputViewportIndex");
		check_feature(features12.bufferDeviceAddress, "bufferDeviceAddress");
		check_feature(features12.shaderBufferInt64Atomics, "shaderBufferInt64Atomics");
		check_feature(features12.shaderSampledImageArrayNonUniformIndexing,
		              "shaderSampledImageArrayNonUniformIndexing");
		check_feature(features12.shaderStorageImageArrayNonUniformIndexing,
		              "shaderStorageImageArrayNonUniformIndexing");
		check_feature(features12.shaderStorageBufferArrayNonUniformIndexing,
		              "shaderStorageBufferArrayNonUniformIndexing");
		check_feature(features12.descriptorBindingSampledImageUpdateAfterBind,
		              "descriptorBindingSampledImageUpdateAfterBind");
		check_feature(features12.descriptorBindingUpdateUnusedWhilePending,
		              "descriptorBindingUpdateUnusedWhilePending");
		check_feature(features12.descriptorBindingPartiallyBound,
		              "descriptorBindingPartiallyBound");
		check_feature(features12.uniformBufferStandardLayout, "uniformBufferStandardLayout");
		check_feature(features13.robustImageAccess, "robustImageAccess");
		check_feature(robustness2.nullDescriptor, "nullDescriptor");
		check_feature(features13.dynamicRendering, "dynamicRendering");
		check_feature(features13.synchronization2, "synchronization2");
		check_feature(device_features2.features.sampleRateShading, "sampleRateShading");
		check_feature(device_features2.features.depthBiasClamp, "depthBiasClamp");
		check_feature(device_features2.features.shaderClipDistance, "shaderClipDistance");
		check_feature(device_features2.features.shaderCullDistance, "shaderCullDistance");
		check_feature(device_features2.features.largePoints, "largePoints");
		check_feature(device_features2.features.multiViewport, "multiViewport");
		check_feature(device_features2.features.fillModeNonSolid, "fillModeNonSolid");
		check_feature(device_features2.features.fragmentStoresAndAtomics,
		              "fragmentStoresAndAtomics");

		check_feature(device_features2.features.samplerAnisotropy, "samplerAnisotropy");
		check_feature(device_features2.features.robustBufferAccess, "robustBufferAccess");
#if defined(__APPLE__)
		if (device_features2.features.depthBounds != VK_TRUE) {
			LOGF("depthBounds is not supported\n");
		}
#else
		check_feature(device_features2.features.depthBounds, "depthBounds");
#endif
		check_feature(device_features2.features.shaderStorageImageWriteWithoutFormat,
		              "shaderStorageImageWriteWithoutFormat");

		check_feature(device_features2.features.shaderImageGatherExtended,
		              "shaderImageGatherExtended");
		check_feature(device_features2.features.shaderResourceMinLod, "shaderResourceMinLod");

		check_feature(device_features2.features.independentBlend, "independentBlend");
		check_feature(device_features2.features.tessellationShader, "tessellationShader");
		check_feature(device_features2.features.drawIndirectFirstInstance,
		              "drawIndirectFirstInstance");
		check_feature(device_features2.features.multiDrawIndirect, "multiDrawIndirect");

		if (!skip_device) {
			auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
			    "vkEnumerateDeviceExtensionProperties",
			    [&](uint32_t* count, vk::ExtensionProperties* values) {
				    return device.enumerateDeviceExtensionProperties(nullptr, count, values);
			    });
			EXIT_NOT_IMPLEMENTED(available_extensions.empty());

			for (const char* ext: device_extensions) {
				if (!HasExtension(available_extensions, ext)) {
					reject(fmt::format("{} is not supported", ext));
				}
			}

			if (skip_device) {
				for (const auto& ext: available_extensions) {
					LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
					     ext.specVersion);
				}
			}
		}

		SurfaceCapabilities candidate_capabilities;
#if defined(ANYPS5_KYTY)
		if (!skip_device && surface != nullptr) {
#else
		if (!skip_device) {
#endif
			GetSurfaceCapabilities(device, surface, candidate_capabilities);

			if (!(candidate_capabilities.capabilities.supportedUsageFlags &
			      vk::ImageUsageFlagBits::eTransferDst)) {
				reject("Surface cannot be destination of blit");
			}
		}

		const auto check_format = [&](vk::Format format, vk::FormatFeatureFlags features,
		                              const char* usage) {
			if (!skip_device && !CheckFormat(device, format, features)) {
				reject(fmt::format("Format vk::Format::e{} cannot be used as {}",
				                   vk::to_string(format), usage));
			}
		};

		const auto depth_features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;
		const auto texture_features =
		    vk::FormatFeatureFlagBits::eSampledImage | vk::FormatFeatureFlagBits::eTransferDst;
		const auto storage_features =
		    vk::FormatFeatureFlagBits::eStorageImage | vk::FormatFeatureFlagBits::eTransferDst;

		check_format(vk::Format::eD32Sfloat, depth_features, "depth buffer");
		check_format(vk::Format::eD32SfloatS8Uint, depth_features, "depth buffer");
		check_format(vk::Format::eD16Unorm, depth_features, "depth buffer");
		check_format(vk::Format::eBc3SrgbBlock, texture_features, "texture");
		check_format(vk::Format::eR8G8B8A8Srgb, texture_features, "texture");
		check_format(vk::Format::eR8Unorm, texture_features, "texture");
		check_format(vk::Format::eR8G8Unorm, texture_features, "texture");

		for (const auto [srgb, unorm]:
		     {std::pair {vk::Format::eR8G8B8A8Srgb, vk::Format::eR8G8B8A8Unorm},
		      std::pair {vk::Format::eB8G8R8A8Srgb, vk::Format::eB8G8R8A8Unorm}}) {
			if (!skip_device && !CheckFormat(device, srgb, storage_features)) {
				LOGF("Format vk::Format::e%s cannot be used as storage image\n",
				     vk::to_string(srgb).c_str());
				check_format(unorm, storage_features, "storage image");
			}
		}

		if (!skip_device && device_properties.limits.maxSamplerAnisotropy < 16.0f) {
			reject("maxSamplerAnisotropy < 16.0f");
		}

		if (skip_device) {
			continue;
		}

		if (best_device == nullptr ||
		    (requested.empty() &&
		     device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)) {
			best_device       = device;
			best_queue_family = queue_family;
		}
	}

	if (best_device == nullptr && !requested.empty()) {
		out_rejections.insert(0, fmt::format(" only among the devices whose name contains "
		                                     "ANYPS5_GPU=\"{}\";",
		                                     Config::GetGpuName()));
	}
	out_device       = best_device;
	out_queue_family = best_queue_family;
}

static vk::Device VulkanCreateDevice(GraphicContext&          graphics,
                                     std::vector<const char*> device_extensions) {
	const auto physical_device = graphics.physical_device;
	const auto queue_family    = graphics.queue_family;
	EXIT_IF(physical_device == nullptr);
	EXIT_IF(queue_family == static_cast<uint32_t>(-1));

	const auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
	    "vkEnumerateDeviceExtensionProperties",
	    [&](uint32_t* count, vk::ExtensionProperties* values) {
		    return physical_device.enumerateDeviceExtensionProperties(nullptr, count, values);
	    });
	const auto add = [&](const char* name) {
		const bool available = HasExtension(available_extensions, name);
		if (available) device_extensions.push_back(name);
		return available;
	};
	vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
	                   vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceRobustness2FeaturesEXT,
	                   vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
	                   vk::PhysicalDeviceMeshShaderFeaturesEXT,
	                   vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT,
	                   vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT,
	                   vk::PhysicalDeviceProvokingVertexFeaturesEXT,
	                   vk::PhysicalDeviceMaintenance8FeaturesKHR,
	                   vk::PhysicalDeviceShaderClockFeaturesKHR,
	                   vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT,
	                   vk::PhysicalDeviceConditionalRenderingFeaturesEXT,
	                   vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR,
	                   vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR,
	                   vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
	                   vk::PhysicalDeviceRayQueryFeaturesKHR,
	                   vk::PhysicalDeviceCooperativeMatrixFeaturesKHR,
	                   vk::PhysicalDeviceShaderFloat8FeaturesEXT>
	    supported;
	auto& [supported2, supported12, supported13, robustness2, workgroup_layout, mesh,
	       feedback_layout, feedback_dynamic, provoking_vertex, maintenance8, shader_clock,
	       image_atomic_int64, conditional_rendering, barycentric, compute_derivatives,
	       acceleration_structure, ray_query, cooperative_matrix, float8] = supported;
	graphics.memory_budget_ext_enabled = add(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
	if (BufferCache::CommandMemoryInPlaceRequested()) {
		graphics.host_import_enabled = add(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
		if (graphics.host_import_enabled) {
			vk::PhysicalDeviceExternalMemoryHostPropertiesEXT host {};
			vk::PhysicalDeviceProperties2                     host_properties {};
			host_properties.pNext = &host;
			physical_device.getProperties2(&host_properties);
			graphics.host_import_alignment = host.minImportedHostPointerAlignment;
		}
	}
	add(VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME);
	const bool extended_dynamic_state = add(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME);
	if (!add(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
	}
	if (!add(VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceMeshShaderFeaturesEXT>();
	}
	if (!HasExtension(available_extensions,
	                  VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) ||
	    !add(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME) ||
	    !add(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
		supported.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
	}
	if (!add(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
	}
	if (!add(VK_KHR_MAINTENANCE_8_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
	}
	if (!add(VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceShaderClockFeaturesKHR>();
	}
	if (!add(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>();
	}
	if (!add(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceConditionalRenderingFeaturesEXT>();
	}
	if (!add(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
	}
	if (!add(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR>();
	}
	const bool ray_query_extensions =
	    Config::NativeRtEnabled() &&
	    HasExtension(available_extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) &&
	    HasExtension(available_extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
	    HasExtension(available_extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
	if (ray_query_extensions) {
		add(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		add(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		add(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	} else {
		supported.unlink<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
		supported.unlink<vk::PhysicalDeviceRayQueryFeaturesKHR>();
	}
	const bool fsr = Fsr::Enabled();
	if (!fsr || !add(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
	}
	if (!fsr || !add(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
		supported.unlink<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
	}
	Vr::OpenXrAppendExtensions(device_extensions, available_extensions, true);
	Dlss::AppendExtensions(device_extensions, available_extensions, true);
	physical_device.getFeatures2(&supported2);
	const bool barycentric_supported = barycentric.fragmentShaderBarycentric == VK_TRUE;
	const bool ray_query_enabled =
	    acceleration_structure.accelerationStructure && ray_query.rayQuery;
	if (!barycentric_supported) {
		Log::WriteToConsoleAndLog(
		    "WARNING: Your GPU is not fully supported (fragmentShaderBarycentric is unavailable). "
		    "Some games may render incorrectly or crash.\n");
	}
	graphics.shader_image_int64_atomics_enabled = image_atomic_int64.shaderImageInt64Atomics;
#if defined(ANYPS5_KYTY)
	if (PipelineLibrariesWanted() && extended_dynamic_state &&
	    HasExtension(available_extensions, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) &&
	    HasExtension(available_extensions, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
		vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>
		    library_features;
		physical_device.getFeatures2(&library_features.get<vk::PhysicalDeviceFeatures2>());
		vk::StructureChain<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT,
		                   vk::PhysicalDeviceDriverProperties>
		    library_properties;
		physical_device.getProperties2(&library_properties.get<vk::PhysicalDeviceProperties2>());
		const auto& library_limits =
		    library_properties.get<vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT>();
		graphics.pipeline_library_enabled =
		    library_properties.get<vk::PhysicalDeviceDriverProperties>().driverID != vk::DriverId::eMesaRadv &&
		    library_features.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>().graphicsPipelineLibrary ==
		        VK_TRUE &&
		    library_limits.graphicsPipelineLibraryFastLinking == VK_TRUE &&
		    library_limits.graphicsPipelineLibraryIndependentInterpolationDecoration == VK_TRUE;
		if (graphics.pipeline_library_enabled) {
			add(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
			add(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
		}
	}
	LOGF("Vulkan graphics pipeline libraries: %s\n", graphics.pipeline_library_enabled ? "true" : "false");
	if (HasExtension(available_extensions, VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
		vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT>
		    interlock_features;
		physical_device.getFeatures2(&interlock_features.get<vk::PhysicalDeviceFeatures2>());
		graphics.fragment_shader_pixel_interlock_enabled =
		    interlock_features.get<vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT>()
		        .fragmentShaderPixelInterlock == VK_TRUE;
		if (graphics.fragment_shader_pixel_interlock_enabled) {
			add(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME);
		}
	}
	LOGF("Vulkan fragment shader pixel interlock: %s\n",
	     graphics.fragment_shader_pixel_interlock_enabled ? "true" : "false");
#endif
	graphics.sampler_filter_minmax_enabled      = supported12.samplerFilterMinmax;

	auto features12                     = WindowContext::RequiredVulkan12Features();
	features12.shaderFloat16           = supported12.shaderFloat16;
	graphics.shader_float16_enabled    = supported12.shaderFloat16 != VK_FALSE;
	LOGF("Vulkan half precision: %s\n", graphics.shader_float16_enabled ? "native" : "software");
	features12.shaderSharedInt64Atomics = supported12.shaderSharedInt64Atomics;
	features12.samplerFilterMinmax      = supported12.samplerFilterMinmax;
	if (!features12.shaderSharedInt64Atomics || !workgroup_layout.workgroupMemoryExplicitLayout) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "WARNING: Native 64-bit LDS atomics are unavailable: shaderSharedInt64Atomics={}, "
		    "workgroupMemoryExplicitLayout={}. Continuing with the selected Vulkan device.\n",
		    features12.shaderSharedInt64Atomics != VK_FALSE,
		    workgroup_layout.workgroupMemoryExplicitLayout != VK_FALSE));
	}
	graphics.mesh_shader_enabled = mesh.meshShader && mesh.taskShader;

	vk::StructureChain<
	    vk::PhysicalDeviceProperties2, vk::PhysicalDeviceFloatControlsProperties,
	    vk::PhysicalDeviceRobustness2PropertiesEXT, vk::PhysicalDeviceVulkan11Properties,
	    vk::PhysicalDeviceSubgroupSizeControlProperties, vk::PhysicalDevicePushDescriptorProperties,
	    vk::PhysicalDeviceDriverProperties, vk::PhysicalDeviceMeshShaderPropertiesEXT>
	    properties;
	auto& [properties2, float_controls, robustness2_props, properties11, subgroup_control,
	       push_descriptor, driver_properties, mesh_properties] = properties;
	if (!graphics.mesh_shader_enabled) {
		properties.unlink<vk::PhysicalDeviceMeshShaderPropertiesEXT>();
	}
	physical_device.getProperties2(&properties2);
	EXIT_NOT_IMPLEMENTED(!(properties11.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eVote));
	graphics.subgroup_supported_stages     = properties11.subgroupSupportedStages;
	graphics.subgroup_supported_operations = properties11.subgroupSupportedOperations;
	graphics.physical_device_properties = properties2.properties;
	graphics.driver_id                  = driver_properties.driverID;
	graphics.max_push_descriptors       = push_descriptor.maxPushDescriptors;
	graphics.mesh_shader_properties     = mesh_properties;
	LOGF("Select device: %s\n", graphics.physical_device_properties.deviceName.data());

	const auto subgroup_stages = subgroup_control.requiredSubgroupSizeStages;
	graphics.subgroup_size     = properties11.subgroupSize;
	graphics.min_subgroup_size = subgroup_control.minSubgroupSize;
	graphics.max_subgroup_size = subgroup_control.maxSubgroupSize;
	graphics.compute_subgroup_size_control_enabled =
	    supported13.subgroupSizeControl == VK_TRUE &&
	    (subgroup_stages & vk::ShaderStageFlagBits::eCompute) &&
	    subgroup_control.minSubgroupSize <= 64 && subgroup_control.maxSubgroupSize >= 64;

	LOGF("Vulkan subgroup: default=%u min=%u max=%u size_control_stages=0x%08x subgroup_stages=0x%08x size_control=%s "
	     "wave64=%s\n",
	     graphics.subgroup_size, graphics.min_subgroup_size, graphics.max_subgroup_size,
	     static_cast<vk::ShaderStageFlags::MaskType>(subgroup_stages),
	     static_cast<vk::ShaderStageFlags::MaskType>(graphics.subgroup_supported_stages),
	     graphics.compute_subgroup_size_control_enabled ? "true" : "false",
	     graphics.SupportsComputeWave64() ? "true" : "false");
	graphics.provoking_vertex_last_enabled    = provoking_vertex.provokingVertexLast;
	graphics.attachment_feedback_loop_enabled = feedback_layout.attachmentFeedbackLoopLayout &&
	                                            feedback_dynamic.attachmentFeedbackLoopDynamicState;
	graphics.maintenance8_enabled             = maintenance8.maintenance8;
	graphics.ray_query_enabled                = ray_query_enabled;
	graphics.robust_buffer_access_enabled =
	    robustness2.robustBufferAccess2 &&
	    robustness2_props.robustStorageBufferAccessSizeAlignment == 1u;
	graphics.r8_srgb_views   = graphics.SamplesFiltered(vk::Format::eR8Srgb);
	graphics.r8g8_srgb_views = graphics.SamplesFiltered(vk::Format::eR8G8Srgb);
	constexpr auto blended_target = vk::FormatFeatureFlagBits::eColorAttachment |
	                                vk::FormatFeatureFlagBits::eColorAttachmentBlend;
	graphics.r4g4b4a4_render_targets =
	    (graphics.GetFormatProperties(vk::Format::eR4G4B4A4UnormPack16).optimalTilingFeatures &
	     blended_target) == blended_target;
	if (shader_clock.shaderDeviceClock) {
		graphics.device_clock_period_ps = static_cast<uint32_t>(
		    std::lround(graphics.physical_device_properties.limits.timestampPeriod * 1000.0f));
	}
	graphics.conditional_rendering_enabled = conditional_rendering.conditionalRendering;
	LOGF("Vulkan maintenance8 support: %s\n",
	     graphics.maintenance8_enabled ? "true" : "false");
	LOGF("Vulkan depth feedback support: %s\n",
	     graphics.attachment_feedback_loop_enabled ? "true" : "false");
	if (graphics.mesh_shader_enabled) {
		LOGF("Vulkan MeshEXT: invocations=%u vertices=%u primitives=%u shared=%u\n",
		     graphics.mesh_shader_properties.maxMeshWorkGroupInvocations,
		     graphics.mesh_shader_properties.maxMeshOutputVertices,
		     graphics.mesh_shader_properties.maxMeshOutputPrimitives,
		     graphics.mesh_shader_properties.maxMeshSharedMemorySize);
	}
	EXIT_NOT_IMPLEMENTED(supported2.features.shaderInt64 != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported2.features.vertexPipelineStoresAndAtomics != VK_TRUE);
	EXIT_NOT_IMPLEMENTED(supported2.features.dualSrcBlend != VK_TRUE);
	vk::PhysicalDeviceFeatures device_features {};
	device_features.fragmentStoresAndAtomics = VK_TRUE;
	device_features.samplerAnisotropy        = VK_TRUE;
	device_features.robustBufferAccess       = VK_TRUE;
#if !defined(__APPLE__)
	device_features.depthBounds = VK_TRUE;
	device_features.depthClamp  = VK_TRUE;
#endif
	device_features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
	device_features.shaderStorageImageMultisample =
	    supported2.features.shaderStorageImageMultisample;
	device_features.shaderSampledImageArrayDynamicIndexing =
	    supported2.features.shaderSampledImageArrayDynamicIndexing;
	if (supported2.features.shaderSampledImageArrayDynamicIndexing != VK_TRUE) {
		LOGF("WARNING: dynamic sampled image indexing is not supported; bindless image "
		     "tables will not work\n");
	}
	device_features.shaderImageGatherExtended            = VK_TRUE;
	device_features.shaderResourceMinLod                 = VK_TRUE;
	device_features.independentBlend                     = VK_TRUE;
	device_features.dualSrcBlend                         = VK_TRUE;
	device_features.tessellationShader                   = VK_TRUE;
	device_features.drawIndirectFirstInstance            = VK_TRUE;
	device_features.multiDrawIndirect                    = VK_TRUE;
	device_features.sampleRateShading                    = VK_TRUE;
	device_features.depthBiasClamp                       = VK_TRUE;
	device_features.shaderClipDistance                   = VK_TRUE;
	device_features.shaderCullDistance                   = VK_TRUE;
	device_features.largePoints                          = VK_TRUE;
	device_features.multiViewport                        = VK_TRUE;
	device_features.fillModeNonSolid                      = VK_TRUE;
	device_features.vertexPipelineStoresAndAtomics       = VK_TRUE;
	if (!device_features.shaderStorageImageMultisample) {
		graphics.physical_device_properties.limits.storageImageSampleCounts =
		    vk::SampleCountFlagBits::e1;
	}
	graphics.occlusion_queries_enabled =
	    Config::OcclusionQueriesEnabled() && supported12.hostQueryReset;
	graphics.occlusion_query_precise_enabled =
	    graphics.occlusion_queries_enabled && supported2.features.occlusionQueryPrecise;
	features12.hostQueryReset             = graphics.occlusion_queries_enabled;
	device_features.occlusionQueryPrecise = graphics.occlusion_query_precise_enabled;
	LOGF("Vulkan ray query: %s\n", ray_query_enabled ? "true" : "false");
	LOGF("Vulkan occlusion queries: %s, precise: %s\n",
	     graphics.occlusion_queries_enabled ? "true" : "false",
	     graphics.occlusion_query_precise_enabled ? "true" : "false");
	device_features.multiDrawIndirect         = supported2.features.multiDrawIndirect;
	device_features.drawIndirectFirstInstance = supported2.features.drawIndirectFirstInstance;
	features12.drawIndirectCount              = supported12.drawIndirectCount;
	graphics.indirect_draws_enabled = device_features.multiDrawIndirect == VK_TRUE &&
	                                  device_features.drawIndirectFirstInstance == VK_TRUE &&
	                                  features12.drawIndirectCount == VK_TRUE;
	device_features.shaderInt64 = VK_TRUE;
	device_features.shaderFloat64 =
	    supported2.features.shaderFloat64 && float_controls.shaderSignedZeroInfNanPreserveFloat64;

	auto features13                = WindowContext::RequiredVulkan13Features();
	features13.robustImageAccess   = supported13.robustImageAccess;
	features13.subgroupSizeControl = graphics.compute_subgroup_size_control_enabled;
	if (fsr) {
		device_features.shaderInt16            = supported2.features.shaderInt16;
		features12.shaderFloat16               = supported12.shaderFloat16;
		features12.shaderInt8                  = supported12.shaderInt8;
		features12.separateDepthStencilLayouts = supported12.separateDepthStencilLayouts;
		features13.shaderIntegerDotProduct     = supported13.shaderIntegerDotProduct;
		features13.computeFullSubgroups        = supported13.computeFullSubgroups;
	}
	features13.pipelineCreationCacheControl = supported13.pipelineCreationCacheControl;
	graphics.pipeline_cache_lookup_enabled  = supported13.pipelineCreationCacheControl == VK_TRUE;

	LOGF("Vulkan robustness: robustImageAccess=%s robustImageAccess2=%s buffer bounds=%s\n",
	     features13.robustImageAccess == VK_TRUE ? "true" : "false",
	     robustness2.robustImageAccess2 == VK_TRUE ? "true" : "false",
	     graphics.robust_buffer_access_enabled ? "robustBufferAccess2" : "shader checks");

	graphics.native_ray_tracing_enabled = ray_query_enabled;
	LOGF("Vulkan native ray tracing: %s\n", graphics.native_ray_tracing_enabled ? "true" : "false");
	RayTracing::SetDeviceSupport(graphics.native_ray_tracing_enabled);
	const float                              queue_priority = 1.0f;
	std::array<vk::DeviceQueueCreateInfo, 2> queue_create_infos {};
	queue_create_infos[0] = {
	    .queueFamilyIndex = queue_family, .queueCount = 1, .pQueuePriorities = &queue_priority};
	uint32_t   queue_create_count = 1;
	const auto families           = physical_device.getQueueFamilyProperties();
	graphics.readback_family      = static_cast<uint32_t>(-1);
	for (uint32_t family = 0; family < families.size(); family++) {
		const auto flags = families[family].queueFlags;
		if (family != queue_family && families[family].queueCount != 0 &&
		    (flags & vk::QueueFlagBits::eTransfer) &&
		    !(flags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute))) {
			graphics.readback_family = family;
			queue_create_infos[queue_create_count++] = {
			    .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &queue_priority};
			break;
		}
	}
	const auto                 extension_count = static_cast<uint32_t>(device_extensions.size());
	const vk::DeviceCreateInfo create_info {.queueCreateInfoCount    = queue_create_count,
	                                        .pQueueCreateInfos       = queue_create_infos.data(),
	                                        .enabledExtensionCount   = extension_count,
	                                        .ppEnabledExtensionNames = device_extensions.data(),
	                                        .pEnabledFeatures        = &device_features};
	provoking_vertex.transformFeedbackPreservesProvokingVertex = VK_FALSE;
	image_atomic_int64.sparseImageInt64Atomics                 = VK_FALSE;
	vk::StructureChain create {
	    create_info, WindowContext::RequiredVulkan11Features(), features12, features13, robustness2,
	    vk::PhysicalDeviceDepthClipControlFeaturesEXT {.depthClipControl = VK_TRUE},
	    vk::PhysicalDeviceImageViewMinLodFeaturesEXT {.minLod = VK_TRUE},
	    vk::PhysicalDeviceDepthClipEnableFeaturesEXT {.depthClipEnable = VK_TRUE},
	    vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR {
	        .workgroupMemoryExplicitLayout = workgroup_layout.workgroupMemoryExplicitLayout},
	    barycentric,
	    vk::PhysicalDeviceMeshShaderFeaturesEXT {.taskShader = VK_TRUE, .meshShader = VK_TRUE},
	    feedback_layout, feedback_dynamic, provoking_vertex, image_atomic_int64, maintenance8,
	    shader_clock, conditional_rendering, compute_derivatives,
	    vk::PhysicalDeviceAccelerationStructureFeaturesKHR {.accelerationStructure = VK_TRUE},
	    vk::PhysicalDeviceRayQueryFeaturesKHR {.rayQuery = VK_TRUE}, cooperative_matrix, float8,
	    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {.extendedDynamicState = VK_TRUE}};
	if (!workgroup_layout.workgroupMemoryExplicitLayout) {
		create.unlink<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
	}
#if defined(__APPLE__)
	create.unlink<vk::PhysicalDeviceDepthClipEnableFeaturesEXT>();
	create.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
#else
	if (!barycentric_supported) {
		create.unlink<vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR>();
	}
#endif
	if (!graphics.mesh_shader_enabled) create.unlink<vk::PhysicalDeviceMeshShaderFeaturesEXT>();
	if (!graphics.attachment_feedback_loop_enabled) {
		create.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT>();
		create.unlink<vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT>();
	}
	if (!graphics.provoking_vertex_last_enabled) {
		create.unlink<vk::PhysicalDeviceProvokingVertexFeaturesEXT>();
	}
	if (!graphics.shader_image_int64_atomics_enabled) {
		create.unlink<vk::PhysicalDeviceShaderImageAtomicInt64FeaturesEXT>();
	}
	if (!graphics.maintenance8_enabled) create.unlink<vk::PhysicalDeviceMaintenance8FeaturesKHR>();
	if (graphics.device_clock_period_ps == 0u) {
		create.unlink<vk::PhysicalDeviceShaderClockFeaturesKHR>();
	}
	if (!graphics.conditional_rendering_enabled) {
		create.unlink<vk::PhysicalDeviceConditionalRenderingFeaturesEXT>();
	}
	if (!compute_derivatives.computeDerivativeGroupQuads) {
		create.unlink<vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR>();
	}
	if (!extended_dynamic_state) {
		create.unlink<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
	}
	if (!cooperative_matrix.cooperativeMatrix) {
		create.unlink<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>();
	}
	if (!float8.shaderFloat8) create.unlink<vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
	if (!ray_query_enabled) {
		create.unlink<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>();
		create.unlink<vk::PhysicalDeviceRayQueryFeaturesKHR>();
	}

#if defined(ANYPS5_KYTY)
	vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT library_enabled {.graphicsPipelineLibrary = VK_TRUE};
	if (graphics.pipeline_library_enabled) {
		auto& device_info     = create.get<vk::DeviceCreateInfo>();
		library_enabled.pNext = const_cast<void*>(device_info.pNext);
		device_info.pNext     = &library_enabled;
	}
	vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT interlock_enabled {.fragmentShaderPixelInterlock = VK_TRUE};
	if (graphics.fragment_shader_pixel_interlock_enabled) {
		auto& device_info       = create.get<vk::DeviceCreateInfo>();
		interlock_enabled.pNext = const_cast<void*>(device_info.pNext);
		device_info.pNext       = &interlock_enabled;
	}
#endif
	vk::Device device = nullptr;
	RequireVulkanSuccess(physical_device.createDevice(&create.get(), nullptr, &device),
	                     "vkCreateDevice");
	return device;
}

static void VulkanGetExtensions(VulkanExtensions& r) {
#if defined(ANYPS5_KYTY)
	EXIT_NOT_IMPLEMENTED(r.required_extensions.empty());
#else
	uint32_t required_extensions_count = 0;

	const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&required_extensions_count);
	EXIT_NOT_IMPLEMENTED(extensions == nullptr);
	EXIT_NOT_IMPLEMENTED(required_extensions_count == 0);
	r.required_extensions.assign(extensions, extensions + required_extensions_count);
#endif

	r.available_extensions =
	    EnumerateVulkan<vk::ExtensionProperties>(
	        "vkEnumerateInstanceExtensionProperties",
	        [](uint32_t* count, vk::ExtensionProperties* values) {
		        return vk::enumerateInstanceExtensionProperties(nullptr, count, values);
	        });

	r.enable_validation_layers = Config::VulkanValidationEnabled();
	if (Config::HdrEnabled() &&
	    HasExtension(r.available_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) &&
	    !HasExtension(r.required_extensions, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
	}

	if (HasExtension(r.available_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
		r.required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	} else {
		r.enable_validation_layers = false;
	}

	for (const char* ext: r.required_extensions) {
		LOGF("Vulkan required extension: %s\n", ext);
	}

	for (const auto& ext: r.available_extensions) {
		LOGF("Vulkan available extension: %s, version = %u\n", ext.extensionName.data(),
		     ext.specVersion);
	}

	r.available_layers = EnumerateVulkan<vk::LayerProperties>(
	    "vkEnumerateInstanceLayerProperties", [](uint32_t* count, vk::LayerProperties* values) {
		    return vk::enumerateInstanceLayerProperties(count, values);
	    });

	for (const auto& l: r.available_layers) {
		LOGF("Vulkan available layer: %s, specVersion = %u, implVersion = %u, %s\n",
		     l.layerName.data(), l.specVersion, l.implementationVersion, l.description.data());
	}

	if (r.enable_validation_layers && !HasLayer(r.available_layers, kValidationLayer)) {
		LOGF("no validation layer: %s\n", kValidationLayer);
		r.enable_validation_layers = false;
	}

	if (r.enable_validation_layers) {
		auto available_extensions = EnumerateVulkan<vk::ExtensionProperties>(
		    "vkEnumerateInstanceExtensionProperties",
		    [](uint32_t* count, vk::ExtensionProperties* values) {
			    return vk::enumerateInstanceExtensionProperties(kValidationLayer, count, values);
		    });

		for (const auto& ext: available_extensions) {
			LOGF("VK_LAYER_KHRONOS_validation available extension: %s, version = %u\n",
			     ext.extensionName.data(), ext.specVersion);
		}

		if (HasExtension(available_extensions, VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
		} else {
			r.enable_validation_layers = false;
		}
	}
}

static std::vector<const char*> RequiredDeviceExtensions() {
	std::vector<const char*> device_extensions = {
	    VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME,
	    VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
	    VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, "VK_KHR_maintenance1"};

#if defined(__APPLE__)
	device_extensions.push_back("VK_KHR_portability_subset");
#else
	device_extensions.push_back(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
#endif

	if (Config::SpirvDebugPrintfEnabled()) {
		device_extensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
	}
	return device_extensions;
}

static VKAPI_ATTR vk::Bool32 VKAPI_CALL VulkanDebugMessengerCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT      message_severity,
    vk::DebugUtilsMessageTypeFlagsEXT             message_types,
    const vk::DebugUtilsMessengerCallbackDataEXT* callback_data, void* ) {
	EXIT_IF(callback_data == nullptr);
	EXIT_IF(callback_data->pMessage == nullptr);

	const char*     severity_str   = nullptr;
	fmt::text_style severity_style = Log::Color::Default;
	bool            skip           = false;
	bool            error          = false;
	bool            debug_printf   = false;
	switch (message_severity) {
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
			severity_str   = "V";
			severity_style = Log::Color::BrightWhite;
			skip           = true;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
			if ((message_types & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) &&
			    Config::SpirvDebugPrintfEnabled() && callback_data->pMessageIdName != nullptr &&
			    strcmp(callback_data->pMessageIdName, "UNASSIGNED-DEBUG-PRINTF") == 0) {
				debug_printf   = true;
				severity_style = Log::Color::BrightYellow;
				skip           = true;
			} else {
				severity_str   = "I";
				severity_style = Log::Color::Default;
				skip           = true;
			}
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
			severity_str   = "W";
			severity_style = Log::Color::Red;
			break;
		case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
			severity_str   = "E";
			severity_style = Log::Color::BrightRed;
			error = static_cast<bool>(message_types &
			                          vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
			break;
		default: severity_str = "?";
	}

	if (error) {
		EXIT("[Vulkan][%s][%u]: %s\n", severity_str, static_cast<uint32_t>(message_types),
		     callback_data->pMessage);
	}

	if (!skip) {
		LOGF_COLOR(severity_style, "[Vulkan][%s][%u]: %s\n", severity_str,
		           static_cast<uint32_t>(message_types), callback_data->pMessage);
	}

	if (debug_printf) {
		auto strs = Common::Split(std::string(callback_data->pMessage), '|');
		if (!strs.empty()) {
			LOGF_COLOR(severity_style, "%s\n", strs[strs.size() - 1].c_str());
		}
	}

	return VK_FALSE;
}

static void VulkanCheckInstanceVersion() {
	uint32_t version = VK_API_VERSION_1_0;

	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion != nullptr) {
		auto result = vk::enumerateInstanceVersion(&version);
		if (result != vk::Result::eSuccess) {
			EXIT("Could not query Vulkan loader version: %s\n", vk::to_string(result).c_str());
		}
	}

	LOGF("Vulkan loader version: %u.%u.%u\n", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
	     VK_VERSION_PATCH(version));
	if (version < VULKAN_TARGET_API_VERSION) {
		EXIT("Vulkan %u.%u is required, but loader supports only %u.%u.%u\n",
		     VK_VERSION_MAJOR(VULKAN_TARGET_API_VERSION),
		     VK_VERSION_MINOR(VULKAN_TARGET_API_VERSION), VK_VERSION_MAJOR(version),
		     VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
	}
}

void WindowContext::CreateVulkan() {
#if !defined(ANYPS5_KYTY)
	EXIT_IF(window == nullptr);
#endif
	EXIT_IF(graphic_ctx.instance != nullptr);
	EXIT_IF(graphic_ctx.physical_device != nullptr);
	EXIT_IF(graphic_ctx.device != nullptr);
	EXIT_IF(surface != nullptr);

#if defined(ANYPS5_KYTY)
	auto get_instance_proc_addr = host_get_instance_proc_addr;
	if (get_instance_proc_addr == nullptr) {
		EXIT("Could not load Vulkan\n");
	}
#else
	auto get_instance_proc_addr =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
	if (get_instance_proc_addr == nullptr) {
		EXIT("Could not load Vulkan: %s\n", SDL_GetError());
	}
#endif
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);

	Vr::OpenXrInit();
	Dlss::Load();
	Fsr::Load();

	VulkanExtensions r;
#if defined(ANYPS5_KYTY)
	r.required_extensions = host_extensions;
#endif
	VulkanGetExtensions(r);
	Vr::OpenXrAppendExtensions(r.required_extensions, r.available_extensions, false);
	Dlss::AppendExtensions(r.required_extensions, r.available_extensions, false);
	VulkanCheckInstanceVersion();

	const vk::ApplicationInfo app_info {.pApplicationName   = "Kyty",
	                                    .applicationVersion = 1,
	                                    .pEngineName        = "Kyty",
	                                    .engineVersion      = 1,
	                                    .apiVersion         = VULKAN_TARGET_API_VERSION};

	if (Config::SpirvDebugPrintfEnabled() && Config::GpuAssistedValidationEnabled()) {
		EXIT("--spirv-debug-printf and --gpu-assisted-validation are mutually exclusive\n");
	}

	vk::ValidationFeatureEnableEXT enabled_features[2]    = {};
	uint32_t                       enabled_features_count = 0;
	if (Config::SpirvDebugPrintfEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eDebugPrintf;
	}
	if (Config::GpuAssistedValidationEnabled()) {
		enabled_features[enabled_features_count++] = vk::ValidationFeatureEnableEXT::eGpuAssisted;
		Log::WriteToConsoleAndLog(
		    "Vulkan GPU-assisted validation is enabled; expect a large slowdown\n");
	}

	vk::ValidationFeaturesEXT validation_features {};
	validation_features.enabledValidationFeatureCount = enabled_features_count;
	validation_features.pEnabledValidationFeatures    = enabled_features;

	vk::DebugUtilsMessengerCreateInfoEXT dbg_create_info {};
	dbg_create_info.pNext           = &validation_features;
	dbg_create_info.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
	                                  vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	dbg_create_info.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
	                                  vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
	dbg_create_info.pfnUserCallback = VulkanDebugMessengerCallback;

	vk::InstanceCreateInfo inst_info {};
	inst_info.pNext                   = (r.enable_validation_layers ? &dbg_create_info : nullptr);
#if defined(__APPLE__)
	if (HasExtension(r.available_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
		if (!HasExtension(r.required_extensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
			r.required_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		}
		inst_info.flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
		LOGF("Vulkan instance: enabled %s\n", VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
	}
#endif
	inst_info.pApplicationInfo        = &app_info;
	inst_info.enabledExtensionCount   = static_cast<uint32_t>(r.required_extensions.size());
	inst_info.ppEnabledExtensionNames = r.required_extensions.data();
	inst_info.enabledLayerCount       = r.enable_validation_layers ? 1u : 0u;
	inst_info.ppEnabledLayerNames     = &kValidationLayer;

	const vk::Result result = vk::createInstance(&inst_info, nullptr, &graphic_ctx.instance);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eErrorIncompatibleDriver:
			EXIT("Unable to find a compatible Vulkan Driver");
		default: EXIT("Could not create a Vulkan instance (for unknown reasons)");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.instance);

	if (r.enable_validation_layers) {
		dbg_create_info.pNext = nullptr;
		if (graphic_ctx.instance.createDebugUtilsMessengerEXT(
		        &dbg_create_info, nullptr, &graphic_ctx.debug_messenger) != vk::Result::eSuccess) {
			EXIT("Could not create debug messenger");
		}
	}

#if !defined(ANYPS5_KYTY)
	RecreateSurface();
#endif

	auto device_extensions = RequiredDeviceExtensions();

	std::string rejected_devices;
	VulkanFindPhysicalDevice(graphic_ctx.instance, surface, device_extensions,
	                         graphic_ctx.physical_device, graphic_ctx.queue_family,
	                         rejected_devices);

	if (graphic_ctx.physical_device == nullptr) {
		EXIT("Could not find suitable device:%s", rejected_devices.c_str());
	}

	graphic_ctx.physical_device.getMemoryProperties(&graphic_ctx.physical_device_memory_properties);
	const vk::PhysicalDeviceImageFormatInfo2 block_texel_view_info {
	    .format = vk::Format::eBc1RgbaUnormBlock,
	    .type = vk::ImageType::e2D,
	    .tiling = vk::ImageTiling::eOptimal,
	    .usage = vk::ImageUsageFlagBits::eSampled,
	    .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
	};
	const auto block_texel_view_props =
	    graphic_ctx.physical_device.getImageFormatProperties2(block_texel_view_info);
	graphic_ctx.supports_block_texel_view = block_texel_view_props.result == vk::Result::eSuccess;
	LOGF("Block Texel View support: %s\n", graphic_ctx.supports_block_texel_view ? "Yes" : "No");

	graphic_ctx.device = VulkanCreateDevice(graphic_ctx, std::move(device_extensions));
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphic_ctx.device);
	graphic_ctx.device.getQueue(graphic_ctx.queue_family, 0, &graphic_ctx.queue);
	EXIT_IF(graphic_ctx.queue == nullptr);
	if (graphic_ctx.readback_family != static_cast<uint32_t>(-1)) {
		graphic_ctx.device.getQueue(graphic_ctx.readback_family, 0, &graphic_ctx.readback_queue);
	}
	Dlss::Init(graphic_ctx, VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
	           VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);
	Fsr::Init(graphic_ctx, VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr);

	if (!graphic_ctx.CreateAllocator()) {
		EXIT("Could not create Vulkan memory allocator");
	}

	ShaderRecompiler::IR::SetComputeIndirectFloors(256, 8);
	render_context = std::make_unique<RenderContext>(graphic_ctx);
	LibKernel::Memory::InstallGpuResources(render_context.get());
	presenter = std::make_unique<Presenter>(*this);
	Vr::OpenXrStartSession(graphic_ctx);
}

VulkanDeviceIdentity GetVulkanDeviceIdentity(const GraphicContext& graphics) {
	const auto&          properties = graphics.GetPhysicalDeviceProperties();
	VulkanDeviceIdentity identity {.vendor_id      = properties.vendorID,
	                               .device_id      = properties.deviceID,
	                               .driver_version = properties.driverVersion,
	                               .queue_family   = graphics.queue_family};
	std::memcpy(identity.cache_uuid.data(), properties.pipelineCacheUUID.data(),
	            identity.cache_uuid.size());
	return identity;
}

bool CreateHeadlessVulkan(GraphicContext& graphics, const VulkanDeviceIdentity* identity,
                          std::string& error) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS || KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static const vk::detail::DynamicLoader loader;
	if (!loader.success()) {
		error = "Vulkan loader not found";
		return false;
	}
	const auto get_instance_proc_addr = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	if (get_instance_proc_addr == nullptr) {
		error = "vkGetInstanceProcAddr not found";
		return false;
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance_proc_addr);
	VulkanCheckInstanceVersion();

	const vk::ApplicationInfo app_info {.pApplicationName   = "Kyty",
	                                    .applicationVersion = 1,
	                                    .pEngineName        = "Kyty",
	                                    .engineVersion      = 1,
	                                    .apiVersion         = VULKAN_TARGET_API_VERSION};
	const auto available = EnumerateVulkan<vk::ExtensionProperties>(
	    "vkEnumerateInstanceExtensionProperties", [](uint32_t* count, vk::ExtensionProperties* values) {
		    return vk::enumerateInstanceExtensionProperties(nullptr, count, values);
	    });
	std::vector<const char*> extensions;
	for (const char* name: {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_win32_surface"}) {
		if (HasExtension(available, name)) {
			extensions.push_back(name);
		}
	}
	vk::InstanceCreateInfo inst_info {};
	inst_info.pApplicationInfo        = &app_info;
	inst_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	inst_info.ppEnabledExtensionNames = extensions.data();
	if (vk::createInstance(&inst_info, nullptr, &graphics.instance) != vk::Result::eSuccess) {
		error = "vkCreateInstance failed";
		return false;
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphics.instance);

	const auto devices = EnumerateVulkan<vk::PhysicalDevice>(
	    "vkEnumeratePhysicalDevices", [&](uint32_t* count, vk::PhysicalDevice* values) {
		    return graphics.instance.enumeratePhysicalDevices(count, values);
	    });
	for (const auto& device: devices) {
		vk::PhysicalDeviceProperties properties {};
		device.getProperties(&properties);
		if (identity == nullptr) {
			if (graphics.physical_device == nullptr ||
			    properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
				graphics.physical_device = device;
			}
			continue;
		}
		if (properties.vendorID == identity->vendor_id && properties.deviceID == identity->device_id &&
		    properties.driverVersion == identity->driver_version &&
		    std::memcmp(properties.pipelineCacheUUID.data(), identity->cache_uuid.data(),
		                identity->cache_uuid.size()) == 0) {
			graphics.physical_device = device;
			break;
		}
	}
	if (graphics.physical_device == nullptr) {
		error = "the emulator's GPU was not found";
		return false;
	}
	if (identity != nullptr) {
		graphics.queue_family = identity->queue_family;
	} else {
		const auto families = graphics.physical_device.getQueueFamilyProperties();
		const auto required = vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute;
		for (uint32_t family = 0; family < families.size(); family++) {
			if ((families[family].queueFlags & required) == required) {
				graphics.queue_family = family;
				break;
			}
		}
	}
	graphics.physical_device.getMemoryProperties(&graphics.physical_device_memory_properties);
	graphics.device = VulkanCreateDevice(graphics, RequiredDeviceExtensions());
	VULKAN_HPP_DEFAULT_DISPATCHER.init(graphics.device);
	return true;
#else
	(void)graphics;
	(void)identity;
	error = "not supported on this platform";
	return false;
#endif
}

void WindowContext::RefreshSurfaceCapabilities() {
	EXIT_IF(graphic_ctx.physical_device == nullptr || surface == nullptr);
	GetSurfaceCapabilities(graphic_ctx.physical_device, surface, surface_capabilities);
}

void WindowContext::RecreateSurface() {
	EXIT_IF(window == nullptr || graphic_ctx.instance == nullptr);
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
#if defined(ANYPS5_KYTY)
	vk::SurfaceKHR::CType native_surface =
	    host_create_surface(host_context, static_cast<vk::Instance::CType>(graphic_ctx.instance));
	if (native_surface == VK_NULL_HANDLE) {
		EXIT("Could not create the Vulkan surface\n");
	}
#else
	vk::SurfaceKHR::CType native_surface = VK_NULL_HANDLE;
	if (!SDL_Vulkan_CreateSurface(window, static_cast<vk::Instance::CType>(graphic_ctx.instance),
	                              nullptr, &native_surface)) {
		EXIT("Could not create the Vulkan surface: %s\n", SDL_GetError());
	}
#endif
	surface = native_surface;
}

WindowContext::~WindowContext() {
	Vr::OpenXrShutdown();
	ShutdownSystemOverlayInput();
	presenter.reset();
	LibKernel::Memory::InstallGpuResources(nullptr);
	render_context.reset();
	Dlss::Shutdown();
	Fsr::Shutdown();

	if (graphic_ctx.device != nullptr) {
		RequireVulkanSuccess(graphic_ctx.device.waitIdle(), "wait for Vulkan device shutdown");
		graphic_ctx.DestroyAllocator();
		graphic_ctx.device.destroy(nullptr);
		graphic_ctx.device = nullptr;
		graphic_ctx.queue  = nullptr;
	}
	if (surface != nullptr) {
		graphic_ctx.instance.destroySurfaceKHR(surface, nullptr);
		surface = nullptr;
	}
	if (graphic_ctx.debug_messenger != nullptr) {
		graphic_ctx.instance.destroyDebugUtilsMessengerEXT(graphic_ctx.debug_messenger, nullptr);
		graphic_ctx.debug_messenger = nullptr;
	}
	if (graphic_ctx.instance != nullptr) {
		graphic_ctx.instance.destroy(nullptr);
		graphic_ctx.instance        = nullptr;
		graphic_ctx.physical_device = nullptr;
	}
#if !defined(ANYPS5_KYTY)
	if (window != nullptr) {
		SDL_DestroyWindow(window);
		window = nullptr;
	}
	SDL_QuitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
#endif
}

}
