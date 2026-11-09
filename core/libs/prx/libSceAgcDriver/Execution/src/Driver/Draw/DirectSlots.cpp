#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectSlots.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DirectHeap.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "IntermediateRepresentation/IrMetadata/BufferFormat.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include <array>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

namespace {

struct TextureMemoEntry {
    std::array<std::uint32_t, 8> words{};
    std::uint64_t epoch = 0;
    std::weak_ptr<Graphics::Texture> texture;
};

std::uint64_t hashWords(const std::array<std::uint32_t, 8>& words) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto word : words) hash = (hash ^ word) * 1099511628211ull;
    return hash;
}

std::uint32_t pointFilterWord(std::uint32_t filter) {
    const bool mipmapped = ((filter >> 26u) & 3u) != 0u;
    return (filter & ~(0xffu << 20u)) | (1u << 24u) | (mipmapped ? 1u << 26u : 0u);
}

}

bool FillDirectSlots(const Graphics::Context& context, const ShaderRecompiler::CompiledShaderInfo& compiled, const ShaderRecompiler::ResourceSnapshot& snapshot, std::vector<std::uint32_t>& slots, std::vector<std::shared_ptr<void>>& keep, DirectSlotStatistics& statistics, std::string& reason) {
    using Kind = ShaderRecompiler::DescriptorBindingKind;
    static std::mutex memoMutex;
    static std::unordered_map<std::uint64_t, TextureMemoEntry> memo;
    auto& heap = Graphics::DirectHeapFor(context);
    const auto& info = compiled.info;
    const auto& layout = compiled.bindings;
    slots.clear();
    slots.reserve(layout.directSlotCount);
    const auto epoch = GuestMemory::ThreadCollectEpoch();
    thread_local std::vector<std::uint32_t> modeOf;
    modeOf.assign(info.images.size(), 0xffffffffu);
    for (const auto& binding : layout.descriptors) {
        const bool samplerHeap = binding.kind == Kind::Samplers;
        const bool imageHeap = ShaderRecompiler::ImageBindingResourceClass(binding.kind) != ShaderRecompiler::ImageResourceClass::None;
        if (!samplerHeap && !imageHeap) continue;
        if (imageHeap && static_cast<std::uint32_t>(binding.kind) > Graphics::DirectHeap::LastSampledKind) {
            reason = "storage image kind";
            return false;
        }
        for (std::size_t element = 0; element < binding.resources.size(); ++element) {
            const auto resource = binding.resources[element];
            if (samplerHeap) {
                ++statistics.samplers;
                if (resource >= snapshot.samplers.size() || context.samplerCache == nullptr) {
                    reason = "sampler without a snapshot";
                    return false;
                }
                std::array<std::uint32_t, 4> words{};
                std::copy_n(snapshot.samplers[resource].dwords.begin(), 4u, words.begin());
                if ((element & 1u) != 0u) words[2] = pointFilterWord(words[2]);
                auto sampler = context.samplerCache->Get(context, words, false, false, true, false);
                const auto slot = heap.SamplerSlot(sampler->Handle(), sampler);
                if (slot == Graphics::DirectHeap::NoSlot) {
                    reason = "sampler heap full";
                    return false;
                }
                slots.push_back(slot);
                continue;
            }
            ++statistics.images;
            if (resource >= snapshot.images.size() || resource >= info.images.size() || resource >= info.runtimeImageModes.size()) {
                reason = "image without a snapshot";
                return false;
            }
            const auto& modes = info.runtimeImageModes[resource];
            const auto& descriptor = snapshot.images[resource].dwords;
            if (descriptor[0] == 0u && (descriptor[1] & 0xffu) == 0u) {
                ++statistics.inactiveImages;
                slots.push_back(0u);
                continue;
            }
            auto mode = modeOf[resource];
            if (mode == 0xffffffffu) {
                try {
                    mode = modeOf[resource] = ShaderRecompiler::ResourceMaterializer::RuntimeImageMode(info.images[resource], snapshot.images[resource], modes);
                } catch (const std::exception& error) {
                    reason = std::string("image mode: ") + error.what();
                    return false;
                }
            }
            if (mode >= modes.size() || ShaderRecompiler::DescriptorBindingForImage(modes[mode]) != binding.kind) {
                ++statistics.inactiveImages;
                slots.push_back(0u);
                continue;
            }
            static const bool noTextures = std::getenv("APS5_DIRECT_NO_TEXTURES") != nullptr;
            if (noTextures) {
                slots.push_back(0u);
                continue;
            }
            auto words = snapshot.images[resource].dwords;
            if (modes[mode].conversionFormat != ShaderRecompiler::IrBufferFormat::Invalid || modes[mode].depthBits) words[3] = (words[3] & ~0xfffu) | ShaderRecompiler::ShaderImageIdentitySwizzle;
            const auto hash = hashWords(words);
            std::shared_ptr<Graphics::Texture> texture;
            {
                std::lock_guard lock(memoMutex);
                if (const auto found = memo.find(hash); found != memo.end() && found->second.epoch == epoch && found->second.words == words) {
                    texture = found->second.texture.lock();
                    if (texture != nullptr) ++statistics.memoHits;
                }
            }
            if (texture == nullptr) {
                ++statistics.textureLookups;
                try {
                    texture = Graphics::CachedSampledTexture(context, std::span<const std::uint32_t>(words.data(), 8u), VK_IMAGE_VIEW_TYPE_2D);
                } catch (const std::exception& error) {
                    ++statistics.failures;
                    reason = std::string("texture: ") + error.what();
                    return false;
                }
                std::lock_guard lock(memoMutex);
                if (memo.size() > 65536u) memo.clear();
                memo[hash] = {words, epoch, texture};
            }
            const auto slot = heap.ImageSlot(static_cast<std::uint32_t>(binding.kind), texture->View(), texture->Layout(), texture);
            if (slot == Graphics::DirectHeap::NoSlot) {
                reason = "image heap full";
                return false;
            }
            keep.push_back(texture);
            slots.push_back(slot);
        }
    }
    if (slots.size() != layout.directSlotCount) {
        reason = "slot count disagrees with the layout";
        return false;
    }
    return true;
}

}
