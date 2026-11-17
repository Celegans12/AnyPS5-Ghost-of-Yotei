#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_OFFTHREADENCODING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_OFFTHREADENCODING_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>

namespace AgcDriver::Graphics::OffThreadEncoding {

bool Enabled();

PFN_vkGetDeviceProcAddr Wrap(VkDevice device, PFN_vkGetDeviceProcAddr resolver);

bool Active();

VkCommandBuffer Acquire(VkDevice device, std::uint32_t queueFamily);

void Begin(VkCommandBuffer commands, VkCommandBufferUsageFlags flags);

VkResult End(VkCommandBuffer commands);

void Retire(VkCommandBuffer commands) noexcept;

void Forget(VkDevice device) noexcept;

struct Statistics {
    std::uint64_t batches = 0;
    std::uint64_t records = 0;
    std::uint64_t bytes = 0;
    std::uint64_t drains = 0;
    std::uint64_t drainNs = 0;
    std::uint64_t directCalls = 0;
    std::uint64_t encodeNs = 0;
};

Statistics Counts();

}

#endif
