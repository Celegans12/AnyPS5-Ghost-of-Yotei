#include "prx/libSceAgcDriver/Graphics/include/DirectHeap.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PlaceholderImages.hpp"
#include "RuntimeAbi.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "Recompiler.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

std::uint32_t sampledSlotsFor(const Context& context) {
    const auto kinds = DirectHeap::LastSampledKind - DirectHeap::FirstImageKind + 1u;
    const auto limit = context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindSampledImages;
    return limit == 0u ? DirectHeap::SampledSlots : std::min(DirectHeap::SampledSlots, limit / kinds);
}

std::uint32_t storageSlotsFor(const Context& context) {
    const auto kinds = DirectHeap::LastStorageKind - DirectHeap::LastSampledKind;
    const auto limit = context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindStorageImages;
    return limit == 0u ? DirectHeap::StorageSlots : std::min(DirectHeap::StorageSlots, limit / kinds);
}

std::uint32_t samplerSlotsFor(const Context& context) {
    const auto limit = context.descriptorIndexingLimits.maxDescriptorSetUpdateAfterBindSamplers;
    return limit == 0u ? DirectHeap::SamplerSlots : std::min(DirectHeap::SamplerSlots, limit);
}

}

bool DirectHeap::Supported(const Context& context) {
    return context.descriptorIndexing && context.runtimeDescriptorArray && context.partiallyBound && context.sampledImageUpdateAfterBind;
}

DirectHeap::DirectHeap(const Context& context) : context(context), kinds(LastStorageKind + 1u) {
    if (!Supported(context)) throw std::runtime_error("direct heap: the device lacks update-after-bind descriptor indexing");
    const auto sampledSlots = sampledSlotsFor(context);
    const auto samplerSlots = samplerSlotsFor(context);
    constexpr VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorBindingFlags> flags;
    for (auto kind = FirstImageKind; kind <= LastSampledKind; ++kind) {
        bindings.push_back({ShaderRecompiler::RuntimeAbi::DirectHeapBindingBase + kind, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampledSlots, stages, nullptr});
        flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    }
    storage = context.storageImageUpdateAfterBind;
    const auto storageSlots = storage ? storageSlotsFor(context) : 0u;
    for (auto kind = LastSampledKind + 1u; storage && kind <= LastStorageKind; ++kind) {
        bindings.push_back({ShaderRecompiler::RuntimeAbi::DirectHeapBindingBase + kind, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageSlots, stages, nullptr});
        flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    }
    bindings.push_back({ShaderRecompiler::RuntimeAbi::DirectHeapBindingBase + SamplerKind, VK_DESCRIPTOR_TYPE_SAMPLER, samplerSlots, stages, nullptr});
    flags.push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT);
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flagInfo.bindingCount = static_cast<std::uint32_t>(flags.size());
    flagInfo.pBindingFlags = flags.data();
    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, &flagInfo};
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreateDescriptorSetLayout direct heap");
    std::vector<VkDescriptorPoolSize> sizes{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, sampledSlots * (LastSampledKind - FirstImageKind + 1u)}, {VK_DESCRIPTOR_TYPE_SAMPLER, samplerSlots}};
    if (storage) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageSlots * (LastStorageKind - LastSampledKind)});
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
        write.dstBinding = ShaderRecompiler::RuntimeAbi::DirectHeapBindingBase + kind;
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
    const bool storageKind = kind > LastSampledKind;
    if (kind < FirstImageKind || kind > LastStorageKind || (storageKind && !storage) || view == VK_NULL_HANDLE) return NoSlot;
    std::lock_guard lock(mutex);
    auto& entry = kinds[kind];
    std::uint32_t slot = NoSlot;
    if (const auto found = entry.slots.find(view); found != entry.slots.end()) {
        if (!found->second.owner.expired()) return found->second.index;
        slot = found->second.index;
        entry.slots.erase(found);
        ++recycled;
    }
    if (slot == NoSlot && entry.free.empty() && entry.next >= (storageKind ? storageSlotsFor(context) : sampledSlotsFor(context))) {
        for (auto it = entry.slots.begin(); it != entry.slots.end();) {
            if (it->second.owner.expired()) {
                entry.free.push_back(it->second.index);
                it = entry.slots.erase(it);
            } else {
                ++it;
            }
        }
    }
    if (slot == NoSlot && !entry.free.empty()) {
        slot = entry.free.back();
        entry.free.pop_back();
        ++recycled;
    }
    if (slot == NoSlot) {
        if (entry.next >= (storageKind ? storageSlotsFor(context) : sampledSlotsFor(context))) return NoSlot;
        slot = entry.next++;
    }
    const VkDescriptorImageInfo info{VK_NULL_HANDLE, view, imageLayout};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = ShaderRecompiler::RuntimeAbi::DirectHeapBindingBase + kind;
    write.dstArrayElement = slot;
    write.descriptorCount = 1u;
    write.descriptorType = storageKind ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, 1u, &write, 0, nullptr);
    ++writes;
    entry.slots[view] = {slot, keep};
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
    write.dstBinding = ShaderRecompiler::RuntimeAbi::DirectHeapBindingBase + SamplerKind;
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
    static std::map<VkDevice, std::string> failed;
    if (const auto found = failed.find(context.device); found != failed.end()) throw std::runtime_error(found->second);
    auto& heap = heaps[context.device];
    if (heap == nullptr) {
        try {
            heap = std::make_unique<DirectHeap>(context);
        } catch (const std::exception& error) {
            heaps.erase(context.device);
            failed[context.device] = error.what();
            throw;
        }
    }
    return *heap;
}


namespace {

// A slot lookup's result, reused while nothing it depends on can have moved: the same collect
// epoch (no ordering point where the CPU could have written guest memory), proof serial and pending
// storage serial, the same rule as ShaderResources::Revalidate's epoch proof. Per worker thread,
// since collect epochs are per thread. APS5_DIRECT_NO_SLOT_MEMO=1 looks every slot up.
struct SlotMemoKey {
    std::uint32_t kind = 0;
    std::uint32_t flags = 0;
    std::array<std::uint32_t, 8> words{};
    bool operator<(const SlotMemoKey& other) const { return std::tie(kind, flags, words) < std::tie(other.kind, other.flags, other.words); }
};

struct SlotMemoValue {
    std::weak_ptr<Texture> texture;
    std::weak_ptr<StorageTexture> storage;
    std::uint32_t slot = DirectHeap::NoSlot;
};

struct SlotMemo {
    std::uint64_t epoch = 0;
    std::uint64_t proof = 0;
    std::uint64_t pending = 0;
    std::map<SlotMemoKey, SlotMemoValue> entries;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
};

bool SlotMemoEnabled() {
    static const bool enabled = std::getenv("APS5_DIRECT_NO_SLOT_MEMO") == nullptr;
    return enabled;
}

SlotMemo& ThreadSlotMemo() {
    thread_local SlotMemo memo;
    if (!SlotMemoEnabled()) return memo;
    const auto epoch = GuestMemory::CollectEpoch();
    const auto proof = GuestMemory::ProofSerial();
    const auto pending = StorageTexture::PendingSerial();
    if (epoch == 0 || epoch != memo.epoch || proof != memo.proof || pending != memo.pending) {
        memo.entries.clear();
        memo.epoch = epoch;
        memo.proof = proof;
        memo.pending = pending;
    }
    return memo;
}

}

bool ResolveDirectStages(const Context& context, std::span<const CompiledShader> shaders, std::vector<std::shared_ptr<ShaderRecompiler::RecompileResult>>& patched, std::vector<CompiledShader>& out, std::vector<std::shared_ptr<StorageTexture>>* written) {
    if (std::ranges::none_of(shaders, [](const CompiledShader& shader) { return shader.program != nullptr && !shader.program->directSlotRequests.empty(); })) return false;
    struct Totals {
        std::uint64_t stages = 0, images = 0, missing = 0, samplers = 0, full = 0, memoHits = 0;
        double us = 0;
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    };
    static std::mutex totalsMutex;
    static Totals totals;
    const auto started = std::chrono::steady_clock::now();
    auto& heap = DirectHeapFor(context);
    auto* recorder = Recorder::Active();
    if (recorder != nullptr && recorder->Device() != context.device) recorder = nullptr;
    std::uint64_t images = 0, missing = 0, samplers = 0, full = 0;
    auto& memo = ThreadSlotMemo();
    const auto memoHitsBefore = memo.hits;
    out.assign(shaders.begin(), shaders.end());
    for (auto& shader : out) {
        const auto* program = shader.program;
        if (program == nullptr || program->directSlotRequests.empty()) continue;
        auto copy = std::make_shared<ShaderRecompiler::RecompileResult>(*program);
        const auto& requests = program->directSlotRequests;
        const auto count = requests.size() / DirectSlotRequest::Words;
        std::vector<std::uint32_t> slots(count, 0u);
        for (std::size_t index = 0; index < count; ++index) {
            const auto* request = requests.data() + index * DirectSlotRequest::Words;
            const auto kind = request[0];
            const auto flags = request[1];
            const std::span<const std::uint32_t> words(request + 2, 8u);
            if ((flags & DirectSlotRequest::Inactive) != 0u) continue;
            std::uint32_t slot = DirectHeap::NoSlot;
            // Image kinds come in groups of seven shapes (1D, 1D array, 2D, 2D array, MSAA, MSAA
            // array, 3D) from kind 1; kinds 22..28 are the depth-comparison group.
            const bool shape2D = kind >= DirectHeap::FirstImageKind && (kind - DirectHeap::FirstImageKind) % 7u == 2u;
            const bool comparison = kind >= 22u && kind <= 28u;
            bool firstLayer = false;
            // Written storage lookups have side effects (depth plane flushes, retiring covered
            // depth surfaces), so only read-only images and samplers come from the memo.
            const bool memoizable = SlotMemoEnabled() && (flags & DirectSlotRequest::Sampler) == 0u && (flags & DirectSlotRequest::Written) == 0u;
            SlotMemoKey memoKey;
            if (memoizable) {
                memoKey.kind = kind;
                memoKey.flags = flags;
                std::copy(words.begin(), words.end(), memoKey.words.begin());
                if (const auto found = memo.entries.find(memoKey); found != memo.entries.end()) {
                    auto texture = found->second.texture.lock();
                    auto storage = found->second.storage.lock();
                    if ((texture != nullptr && !texture->Released()) || storage != nullptr) {
                        ++images;
                        ++memo.hits;
                        if (recorder != nullptr) {
                            if (texture != nullptr) recorder->Keep(texture);
                            else recorder->Keep(storage);
                        }
                        slots[index] = found->second.slot;
                        continue;
                    }
                    memo.entries.erase(found);
                }
                ++memo.misses;
            }
            if ((flags & DirectSlotRequest::Storage) != 0u) {
                ++images;
                const bool writes = (flags & DirectSlotRequest::Written) != 0u;
                std::uint32_t level = 0;
                std::shared_ptr<StorageTexture> storage;
                try {
                    storage = DirectStorageTexture(context, words, flags >> DirectSlotRequest::MipShift, writes, level, shape2D, firstLayer);
                } catch (const std::exception&) {
                    storage = nullptr;
                }
                if (storage == nullptr) {
                    ++missing;
                    continue;
                }
                const auto view = (flags & DirectSlotRequest::Atomic64) != 0u ? storage->Atomic64View(level, firstLayer) : (flags & DirectSlotRequest::Atomic) != 0u ? storage->AtomicView(level, firstLayer) : storage->StorageView(level, firstLayer, (flags & DirectSlotRequest::UintView) != 0u);
                slot = heap.ImageSlot(kind, view, VK_IMAGE_LAYOUT_GENERAL, storage);
                if (recorder != nullptr) recorder->Keep(storage);
                if (writes && written != nullptr) written->push_back(storage);
                if (memoizable && slot != DirectHeap::NoSlot) memo.entries[memoKey] = {{}, storage, slot};
            } else if ((flags & DirectSlotRequest::Sampler) != 0u) {
                ++samplers;
                if (context.samplerCache == nullptr) continue;
                auto sampler = context.samplerCache->Get(context, words.first(4u), false, false, true, false);
                slot = heap.SamplerSlot(sampler->Handle(), sampler);
                if (recorder != nullptr) recorder->Keep(sampler);
            } else {
                ++images;
                std::shared_ptr<Texture> texture;
                try {
                    texture = DirectSampledTexture(context, words, shape2D, comparison, firstLayer);
                } catch (const std::exception&) {
                    texture = nullptr;
                }
                if (texture == nullptr) {
                    ++missing;
                    continue;
                }
                const auto view = firstLayer && texture->FirstLayerView() != VK_NULL_HANDLE ? texture->FirstLayerView() : texture->View();
                slot = heap.ImageSlot(kind, view, texture->Layout(), texture);
                if (recorder != nullptr) recorder->Keep(texture);
                if (memoizable && slot != DirectHeap::NoSlot) memo.entries[memoKey] = {texture, {}, slot};
            }
            if (slot == DirectHeap::NoSlot) {
                ++full;
                continue;
            }
            slots[index] = slot;
        }
        bool written = false;
        for (auto& binding : copy->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::ShaderData) continue;
            if (binding.guestDescriptor.size() < copy->directSlotDword + slots.size()) binding.guestDescriptor.resize(copy->directSlotDword + slots.size(), 0u);
            std::copy(slots.begin(), slots.end(), binding.guestDescriptor.begin() + copy->directSlotDword);
            written = true;
        }
        if (!written) {
            const auto offset = static_cast<std::size_t>(copy->directSlotDword) * sizeof(std::uint32_t);
            const auto bytes = slots.size() * sizeof(std::uint32_t);
            if (copy->pushConstants.size() >= offset + bytes) std::memcpy(copy->pushConstants.data() + offset, slots.data(), bytes);
        }
        copy->directSlotRequests.clear();
        shader.program = copy.get();
        patched.push_back(std::move(copy));
    }
    std::lock_guard lock(totalsMutex);
    ++totals.stages;
    totals.images += images;
    totals.missing += missing;
    totals.samplers += samplers;
    totals.full += full;
    totals.memoHits += memo.hits - memoHitsBefore;
    totals.us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
    const auto now = std::chrono::steady_clock::now();
    if (now - totals.last >= std::chrono::seconds(10)) {
        std::fprintf(stderr, "[direct-resolve] 10 s: %llu draws, avg %.2f us; images %llu (%llu missing, %llu from the slot memo), samplers %llu, heap full %llu; heap writes %llu, recycled %llu\n", static_cast<unsigned long long>(totals.stages), totals.stages != 0 ? totals.us / static_cast<double>(totals.stages) : 0.0, static_cast<unsigned long long>(totals.images), static_cast<unsigned long long>(totals.missing), static_cast<unsigned long long>(totals.memoHits), static_cast<unsigned long long>(totals.samplers), static_cast<unsigned long long>(totals.full), static_cast<unsigned long long>(heap.Writes()), static_cast<unsigned long long>(heap.Recycled()));
        totals = {};
        totals.last = now;
    }
    return true;
}

}
