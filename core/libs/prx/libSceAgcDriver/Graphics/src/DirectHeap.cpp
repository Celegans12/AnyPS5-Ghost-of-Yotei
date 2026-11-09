#include "prx/libSceAgcDriver/Graphics/include/DirectHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "RuntimeAbi.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>

namespace AgcDriver::Graphics {

namespace {

std::uint32_t sampledSlotsFor(const Context& context) {
    const auto kinds = DirectHeap::LastSampledKind - DirectHeap::FirstImageKind + 1u;
    const auto limit = context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindSampledImages;
    return limit == 0u ? DirectHeap::SampledSlots : std::min(DirectHeap::SampledSlots, limit / kinds);
}

std::uint32_t samplerSlotsFor(const Context& context) {
    const auto limit = context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindSamplers;
    return limit == 0u ? DirectHeap::SamplerSlots : std::min(DirectHeap::SamplerSlots, limit);
}

}

bool DirectHeap::Supported(const Context& context) {
    return context.descriptorIndexing && context.runtimeDescriptorArray && context.partiallyBound && context.sampledImageUpdateAfterBind && context.samplerTableUpdateAfterBind;
}

DirectHeap::DirectHeap(const Context& context) : context(context), kinds(LastSampledKind + 1u) {
    if (!Supported(context)) throw std::runtime_error("direct heap: the device lacks update-after-bind descriptor indexing");
    const auto sampledSlots = sampledSlotsFor(context);
    const auto samplerSlots = samplerSlotsFor(context);
    constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorBindingFlags> flags;
    for (auto kind = FirstImageKind; kind <= LastSampledKind; ++kind) {
        bindings.push_back({kind, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampledSlots, stages, nullptr});
        flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    }
    bindings.push_back({SamplerKind, VK_DESCRIPTOR_TYPE_SAMPLER, samplerSlots, stages, nullptr});
    flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flagInfo.bindingCount = static_cast<std::uint32_t>(flags.size());
    flagInfo.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &flagInfo};
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreateDescriptorSetLayout direct heap");
    const std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampledSlots * (LastSampledKind - FirstImageKind + 1u)}, {VK_DESCRIPTOR_TYPE_SAMPLER, samplerSlots}}};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1u;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
    poolInfo.pPoolSizes = sizes.data();
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool direct heap");
    VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = pool;
    allocateInfo.descriptorSetCount = 1u;
    allocateInfo.pSetLayouts = &layout;
    Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocateInfo, &set), "vkAllocateDescriptorSets direct heap");
    std::vector<VkDescriptorImageInfo> placeholders;
    std::vector<VkWriteDescriptorSet> writesOut;
    placeholders.reserve(LastSampledKind);
    for (auto kind = FirstImageKind; kind <= LastSampledKind; ++kind) {
        constexpr auto bindingCount = static_cast<std::uint32_t>(ShaderRecompiler::RuntimeAbi::Binding::Count);
        const auto view = context.nullDescriptors ? VK_NULL_HANDLE : context.placeholderImages->View(kind % bindingCount);
        placeholders.push_back({VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = kind;
        write.dstArrayElement = 0u;
        write.descriptorCount = 1u;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &placeholders.back();
        writesOut.push_back(write);
    }
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writesOut.size()), writesOut.data(), 0, nullptr);
}

DirectHeap::~DirectHeap() {
    if (pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (layout != VK_NULL_HANDLE) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, layout, nullptr);
}

std::uint32_t DirectHeap::ImageSlot(std::uint32_t kind, VkImageView view, VkImageLayout imageLayout, std::shared_ptr<void> keep) {
    if (kind < FirstImageKind || kind > LastSampledKind || view == VK_NULL_HANDLE) return NoSlot;
    std::lock_guard lock(mutex);
    auto& entry = kinds[kind];
    if (const auto found = entry.slots.find(view); found != entry.slots.end()) return found->second;
    if (entry.next >= sampledSlotsFor(context)) return NoSlot;
    const auto slot = entry.next++;
    const VkDescriptorImageInfo info{VK_NULL_HANDLE, view, imageLayout};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = kind;
    write.dstArrayElement = slot;
    write.descriptorCount = 1u;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, 1u, &write, 0, nullptr);
    ++writes;
    entry.slots.emplace(view, slot);
    entry.kept.push_back(std::move(keep));
    return slot;
}

std::uint32_t DirectHeap::SamplerSlot(VkSampler sampler, std::shared_ptr<void> keep) {
    if (sampler == VK_NULL_HANDLE) return NoSlot;
    std::lock_guard lock(mutex);
    if (const auto found = samplerSlots.find(sampler); found != samplerSlots.end()) return found->second;
    if (nextSampler >= samplerSlotsFor(context)) return NoSlot;
    const auto slot = nextSampler++;
    const VkDescriptorImageInfo info{sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = SamplerKind;
    write.dstArrayElement = slot;
    write.descriptorCount = 1u;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = &info;
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, 1u, &write, 0, nullptr);
    ++writes;
    samplerSlots.emplace(sampler, slot);
    keptSamplers.push_back(std::move(keep));
    return slot;
}

DirectHeap& DirectHeapFor(const Context& context) {
    static std::mutex mutex;
    static std::map<VkDevice, std::unique_ptr<DirectHeap>> heaps;
    std::lock_guard lock(mutex);
    auto& heap = heaps[context.device];
    if (heap == nullptr) heap = std::make_unique<DirectHeap>(context);
    return *heap;
}

}
