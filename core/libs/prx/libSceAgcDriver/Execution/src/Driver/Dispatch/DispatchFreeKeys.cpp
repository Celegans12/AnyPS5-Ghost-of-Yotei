#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t MaxFreeShapes = 65536;

std::uint64_t WordsBelow(std::size_t count) {
    return count >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << count) - 1u;
}

}

std::uint64_t PatchableUserWords(const ShaderRecompiler::RecompileResult& compiled) {
    if (std::any_of(compiled.bindings.begin(), compiled.bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; })) return 0;
    std::uint64_t words = 0;
    for (const auto word : compiled.shaderDataUserWords) {
        if (word < 64) words |= std::uint64_t{1} << word;
    }
    return words;
}

std::uint64_t FreeDispatchUserWords(const ShaderRecompiler::SrtUserDataUse& use, std::uint64_t registerReads, std::uint64_t patchable, std::uint64_t relocatable, std::size_t userCount) {
    const auto walked = use.asBase | use.asValue;
    if (walked == 0 || userCount > 64) return 0;
    const auto words = WordsBelow(userCount);
    return (words & ~(walked | registerReads)) | (relocatable & ~(registerReads & ~patchable) & ~use.asValue & use.asBase & words);
}

std::uint64_t MixDispatchUserWords(std::uint64_t key, std::span<const std::uint32_t> userData, std::uint64_t freeWords) {
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    for (std::size_t i = 0; i < userData.size(); ++i) {
        const auto word = userData[i];
        if (i < 64 && (freeWords >> i & 1u) != 0) mix(0x100u | (word & 3u) | (word == 0 ? 4u : 0u));
        else mix(word);
    }
    if (freeWords != 0) {
        mix(0xf4ee0000f4ee0000ull);
        mix(freeWords);
    }
    return key;
}

UserWordsMatch MatchDispatchUserWords(const DispatchVariant& variant, std::span<const std::uint32_t> userData, std::vector<std::pair<std::uint64_t, std::uint64_t>>& runs, std::uint64_t& moved) {
    moved = 0;
    if (variant.userData.size() != userData.size() || userData.size() > 64) return UserWordsMatch::Differs;
    for (std::size_t i = 0; i < userData.size(); ++i) {
        const auto from = variant.userData[i];
        const auto to = userData[i];
        if (from == to) continue;
        const auto bit = std::uint64_t{1} << i;
        if ((variant.exactUserWords & bit) != 0) return UserWordsMatch::Differs;
        if ((variant.relocatableWords & bit) == 0) continue;
        if (from == 0 || to == 0 || ((from ^ to) & 3u) != 0) return UserWordsMatch::Differs;
        moved |= bit;
    }
    if (moved == 0) return UserWordsMatch::Same;
    runs = variant.runs;
    for (std::size_t run = 0; run < runs.size() && run < variant.runBases.size(); ++run) {
        const auto base = variant.runBases[run];
        if (base < 0 || (moved >> base & 1u) == 0) continue;
        const auto delta = static_cast<std::uint64_t>(userData[static_cast<std::size_t>(base)] & ~std::uint32_t{3}) - static_cast<std::uint64_t>(variant.userData[static_cast<std::size_t>(base)] & ~std::uint32_t{3});
        runs[run].first += delta;
        runs[run].second += delta;
    }
    return UserWordsMatch::Moved;
}

std::shared_ptr<const ShaderRecompiler::RecompileResult> PatchDispatchUserWords(const ShaderRecompiler::RecompileResult& compiled, std::span<const std::uint32_t> userData, std::uint64_t moved, bool& pushes, bool& data) {
    pushes = false;
    data = false;
    auto patched = std::make_shared<ShaderRecompiler::RecompileResult>(compiled);
    const auto& forwarded = compiled.shaderDataUserWords;
    for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
        const auto word = forwarded[slot];
        if (word >= 64 || word >= userData.size() || (moved >> word & 1u) == 0) continue;
        const auto value = userData[word];
        const auto at = static_cast<std::size_t>(slot) * sizeof(std::uint32_t);
        if (at + sizeof(std::uint32_t) <= patched->pushConstants.size()) {
            std::memcpy(patched->pushConstants.data() + at, &value, sizeof(value));
            pushes = true;
        }
        for (auto& binding : patched->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::ShaderData || slot >= binding.guestDescriptor.size()) continue;
            binding.guestDescriptor[slot] = value;
            data = true;
        }
    }
    return patched;
}

std::shared_ptr<const Recipe> MovedRecipe(const Recipe& recipe, const ShaderRecompiler::RecompileResult& compiled) {
    auto moved = std::make_shared<Recipe>(recipe);
    if (!recipe.pushes) return moved;
    const auto resources = recipe.templateRef.lock();
    if (resources == nullptr) return nullptr;
    const std::array<Graphics::CompiledShader, 1> shaders{{{ShaderRecompiler::ShaderStage::Compute, &compiled, 0}}};
    moved->pushBytes = Graphics::AssemblePushConstants(shaders);
    resources->PatchPushConstants(moved->pushBytes);
    return moved;
}

bool Driver::freeDispatchKeys() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::AddressFreeDispatchKeys) && !stampValidate();
}

bool Driver::verifyFreeDispatchKeys() {
    static const bool verify = std::getenv("APS5_VERIFY_FREE_DISPATCH_KEYS") != nullptr;
    return verify;
}

std::uint64_t Driver::freeShapeIdentity(std::uint64_t address, const ShaderSnapshot* shader, std::size_t userCount) {
    std::uint64_t identity = 0xcbf29ce484222325ull;
    for (const auto value : {address, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(shader)), static_cast<std::uint64_t>(userCount)}) {
        identity ^= value;
        identity *= 0x100000001b3ull;
    }
    return identity;
}

std::uint64_t Driver::dispatchFreeWords(std::uint64_t address, const ShaderSnapshot* shader, std::size_t userCount) {
    std::lock_guard lock(dispatchCacheMutex);
    const auto found = dispatchFreeShapes.find(freeShapeIdentity(address, shader, userCount));
    return found != dispatchFreeShapes.end() ? found->second : 0;
}

void Driver::learnDispatchFreeWords(std::uint64_t address, const ShaderSnapshot* shader, std::size_t userCount, std::uint64_t freeWords) {
    const auto identity = freeShapeIdentity(address, shader, userCount);
    const auto found = dispatchFreeShapes.find(identity);
    if (found == dispatchFreeShapes.end()) {
        if (dispatchFreeShapes.size() >= MaxFreeShapes) dispatchFreeShapes.clear();
        dispatchFreeShapes.emplace(identity, freeWords);
        return;
    }
    if ((found->second & ~freeWords) == 0) return;
    found->second &= freeWords;
    ++entryCounters.freeShapesNarrowed;
}

}
