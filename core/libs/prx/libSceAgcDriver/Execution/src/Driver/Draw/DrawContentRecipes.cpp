#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include <cstdlib>
#include <list>
#include <mutex>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::size_t MaxContentRecipes = 8192;

struct HeldRecipe {
    std::shared_ptr<const DrawContentKey> key;
    std::shared_ptr<const DrawRecipe> recipe;
    std::list<std::uint64_t>::iterator order;
    std::uint64_t touched = 0;
};

struct ContentRecipeTable {
    AgcDriver::Mutex mutex;
    std::unordered_map<std::uint64_t, HeldRecipe> held;
    std::list<std::uint64_t> order;
    std::uint64_t lookups = 0;
    std::uint64_t evicted = 0;
};

ContentRecipeTable& contentRecipes() {
    static ContentRecipeTable* const table = new ContentRecipeTable();
    return *table;
}

}

bool ContentRecipes() {
    static const bool enabled = std::getenv("APS5_NO_CONTENT_RECIPES") == nullptr;
    return enabled;
}

std::shared_ptr<const DrawContentKey> ContentKeyOf(std::uint64_t shape, const Pm4::DrawParameters& draw, std::span<const Graphics::CompiledShader> stages) {
    auto key = std::make_shared<DrawContentKey>();
    auto& words = key->words;
    words.reserve(256);
    words.push_back(static_cast<std::uint32_t>(shape));
    words.push_back(static_cast<std::uint32_t>(shape >> 32u));
    if (!Graphics::AppendDrawContentKey(stages, draw, words)) return nullptr;
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto word : words) hash = (hash ^ word) * 0x100000001b3ull;
    key->hash = hash;
    return key;
}

std::shared_ptr<const DrawRecipe> FindContentRecipe(const DrawContentKey& key) {
    auto& table = contentRecipes();
    std::lock_guard lock(table.mutex);
    ++table.lookups;
    const auto found = table.held.find(key.hash);
    if (found == table.held.end() || found->second.key->words != key.words) return nullptr;
    if (table.lookups - found->second.touched > MaxContentRecipes / 8) {
        table.order.splice(table.order.begin(), table.order, found->second.order);
        found->second.touched = table.lookups;
    }
    return found->second.recipe;
}

void AttachContentRecipe(const std::shared_ptr<const DrawContentKey>& key, std::shared_ptr<const DrawRecipe> recipe) {
    auto& table = contentRecipes();
    std::lock_guard lock(table.mutex);
    const auto [found, added] = table.held.try_emplace(key->hash);
    if (added) {
        table.order.push_front(key->hash);
        found->second.order = table.order.begin();
    } else {
        table.order.splice(table.order.begin(), table.order, found->second.order);
    }
    found->second.key = key;
    found->second.recipe = std::move(recipe);
    found->second.touched = table.lookups;
    VulkanDevice::NoteContentRecipeAttach();
    while (table.held.size() > MaxContentRecipes) {
        table.held.erase(table.order.back());
        table.order.pop_back();
        ++table.evicted;
    }
}

ContentRecipeCounts ContentRecipesHeld() {
    auto& table = contentRecipes();
    std::lock_guard lock(table.mutex);
    return {table.held.size(), table.evicted};
}

}
