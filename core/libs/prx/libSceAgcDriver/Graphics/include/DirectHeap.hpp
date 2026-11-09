#ifndef AGC_GRAPHICS_DIRECTHEAP_HPP
#define AGC_GRAPHICS_DIRECTHEAP_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics {

class DirectHeap {
public:
    static constexpr std::uint32_t FirstImageKind = 1u;
    static constexpr std::uint32_t LastSampledKind = 28u;
    static constexpr std::uint32_t SamplerKind = 57u;
    static constexpr std::uint32_t SampledSlots = 8192u;
    static constexpr std::uint32_t SamplerSlots = 4096u;
    static constexpr std::uint32_t NoSlot = 0xffffffffu;

    explicit DirectHeap(const Context& context);
    ~DirectHeap();
    DirectHeap(const DirectHeap&) = delete;
    DirectHeap& operator=(const DirectHeap&) = delete;

    static bool Supported(const Context& context);
    VkDescriptorSetLayout Layout() const { return layout; }
    VkDescriptorSet Set() const { return set; }
    std::uint32_t ImageSlot(std::uint32_t kind, VkImageView view, VkImageLayout imageLayout, std::shared_ptr<void> keep);
    std::uint32_t SamplerSlot(VkSampler sampler, std::shared_ptr<void> keep);
    std::uint64_t Writes() const { return writes; }

private:
    struct Kind {
        std::unordered_map<VkImageView, std::uint32_t> slots;
        std::vector<std::shared_ptr<void>> kept;
        std::uint32_t next = 1u;
    };
    Context context;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::mutex mutex;
    std::vector<Kind> kinds;
    std::unordered_map<VkSampler, std::uint32_t> samplerSlots;
    std::vector<std::shared_ptr<void>> keptSamplers;
    std::uint32_t nextSampler = 2u;
    std::uint64_t writes = 0;
};

DirectHeap& DirectHeapFor(const Context& context);

}

#endif
