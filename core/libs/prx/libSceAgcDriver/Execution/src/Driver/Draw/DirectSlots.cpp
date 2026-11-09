#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DirectSlots.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DirectHeap.hpp"
#include "IntermediateRepresentation/IrMetadata/BufferFormat.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/ResourceMaterializer.hpp"
#include <algorithm>
#include <array>
#include <exception>

namespace AgcDriver::DriverDetail {

namespace {

std::uint32_t pointFilterWord(std::uint32_t filter) {
    const bool mipmapped = ((filter >> 26u) & 3u) != 0u;
    return (filter & ~(0xffu << 20u)) | (1u << 24u) | (mipmapped ? 1u << 26u : 0u);
}

void pushRequest(std::vector<std::uint32_t>& requests, std::uint32_t kind, std::uint32_t flags, const std::array<std::uint32_t, 8>& words) {
    requests.push_back(kind);
    requests.push_back(flags);
    requests.insert(requests.end(), words.begin(), words.end());
}

}

bool BuildDirectSlotRequests(const ShaderRecompiler::CompiledShaderInfo& compiled, const ShaderRecompiler::ResourceSnapshot& snapshot, std::vector<std::uint32_t>& requests, DirectSlotStatistics& statistics, std::string& reason) {
    using Kind = ShaderRecompiler::DescriptorBindingKind;
    using Request = Graphics::DirectSlotRequest;
    const auto& info = compiled.info;
    const auto& layout = compiled.bindings;
    requests.clear();
    requests.reserve(static_cast<std::size_t>(layout.directSlotCount) * Request::Words);
    thread_local std::vector<std::uint32_t> modeOf;
    modeOf.assign(info.images.size(), 0xffffffffu);
    std::size_t slots = 0;
    for (const auto& binding : layout.descriptors) {
        const bool samplerHeap = binding.kind == Kind::Samplers;
        const bool imageHeap = ShaderRecompiler::ImageBindingResourceClass(binding.kind) != ShaderRecompiler::ImageResourceClass::None;
        if (!samplerHeap && !imageHeap) continue;
        const auto kind = static_cast<std::uint32_t>(binding.kind);
        if (imageHeap && kind > Graphics::DirectHeap::LastSampledKind) {
            reason = "storage image kind";
            return false;
        }
        for (std::size_t element = 0; element < binding.resources.size(); ++element) {
            ++slots;
            const auto resource = binding.resources[element];
            std::array<std::uint32_t, 8> words{};
            if (samplerHeap) {
                ++statistics.samplers;
                if (resource >= snapshot.samplers.size()) {
                    reason = "sampler without a snapshot";
                    return false;
                }
                std::copy_n(snapshot.samplers[resource].dwords.begin(), 4u, words.begin());
                std::uint32_t flags = Request::Sampler;
                if ((element & 1u) != 0u) {
                    words[2] = pointFilterWord(words[2]);
                    flags |= Request::Point;
                }
                pushRequest(requests, kind, flags, words);
                continue;
            }
            ++statistics.images;
            if (resource >= snapshot.images.size() || resource >= info.images.size() || resource >= info.runtimeImageModes.size()) {
                reason = "image without a snapshot";
                return false;
            }
            const auto& descriptor = snapshot.images[resource].dwords;
            if (descriptor[0] == 0u && (descriptor[1] & 0xffu) == 0u) {
                ++statistics.inactiveImages;
                pushRequest(requests, kind, Request::Inactive, words);
                continue;
            }
            const auto& modes = info.runtimeImageModes[resource];
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
                pushRequest(requests, kind, Request::Inactive, words);
                continue;
            }
            words = descriptor;
            if (modes[mode].conversionFormat != ShaderRecompiler::IrBufferFormat::Invalid || modes[mode].depthBits) words[3] = (words[3] & ~0xfffu) | ShaderRecompiler::ShaderImageIdentitySwizzle;
            pushRequest(requests, kind, 0u, words);
        }
    }
    if (slots != layout.directSlotCount) {
        reason = "slot count disagrees with the layout";
        return false;
    }
    return true;
}

}
