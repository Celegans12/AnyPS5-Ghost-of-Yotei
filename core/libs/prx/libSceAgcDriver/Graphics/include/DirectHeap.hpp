#ifndef AGC_GRAPHICS_DIRECTHEAP_HPP
#define AGC_GRAPHICS_DIRECTHEAP_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <cstdint>
#include <span>
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
    std::uint64_t Recycled() const { return recycled; }

private:
    struct Kind {
        struct Slot {
            std::uint32_t index = 0;
            std::weak_ptr<void> owner;
        };
        std::unordered_map<VkImageView, Slot> slots;
        std::vector<std::uint32_t> free;
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
    std::uint64_t recycled = 0;
};

DirectHeap& DirectHeapFor(const Context& context);

struct DirectSlotRequest {
    static constexpr std::uint32_t Words = 10u;
    static constexpr std::uint32_t Sampler = 1u;
    static constexpr std::uint32_t Point = 2u;
    static constexpr std::uint32_t Inactive = 4u;
};

bool ResolveDirectStages(const Context& context, std::span<const CompiledShader> shaders, std::vector<std::shared_ptr<ShaderRecompiler::RecompileResult>>& patched, std::vector<CompiledShader>& out);

}

#endif
