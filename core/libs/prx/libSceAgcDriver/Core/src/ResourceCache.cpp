#include "prx/libSceAgcDriver/Core/include/ResourceCache.hpp"
#include "Optimization/BlockLayout.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace AgcDriver::Core {

namespace {

constexpr std::uint64_t Golden = 0x9e3779b97f4a7c15ull;
constexpr std::uint64_t LowSeed = 0x243f6a8885a308d3ull;
constexpr std::uint64_t HighSeed = 0x13198a2e03707344ull;

std::uint64_t mixLow(std::uint64_t value) {
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= value >> 33u;
    return value;
}

std::uint64_t mixHigh(std::uint64_t value) {
    value ^= value >> 30u;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27u;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31u;
    return value;
}

std::uint64_t combine(std::uint64_t seed, std::uint64_t value) {
    return mixLow(seed ^ (value + Golden + (seed << 6u) + (seed >> 2u)));
}

void absorb(std::array<std::uint64_t, 2>& digest, std::uint64_t item) {
    digest[0] = mixLow(digest[0] ^ item);
    digest[1] = mixHigh(digest[1] + std::rotl(item, 29) + Golden);
}

std::uint64_t hashOf(const BufferKey& key) {
    std::uint64_t hash = combine(Golden, key.address);
    hash = combine(hash, key.bytes);
    return combine(hash, (std::uint64_t{key.alignment} << 32u) | key.flags);
}

std::uint64_t hashOf(const ImageSetKey& key) {
    return combine(combine(combine(Golden, key.layout), key.digest[0]), key.digest[1]);
}

template <typename Key, typename Value>
class Table {
public:
    explicit Table(std::size_t slots) : entries(std::bit_ceil(std::max<std::size_t>(slots, 1u))) {}

    Hit Find(const Key& key, const Proof& now, Value& value) const {
        const Entry& entry = entries[slotOf(key)];
        if (!entry.used || entry.key != key) return Hit::Miss;
        value = entry.value;
        return entry.proof == now ? Hit::Fresh : Hit::Stale;
    }

    void Keep(const Key& key, const Proof& proof, Value value) {
        Entry& entry = entries[slotOf(key)];
        entry.key = key;
        entry.proof = proof;
        entry.value = std::move(value);
        entry.used = true;
    }

private:
    struct Entry {
        Key key;
        Proof proof;
        Value value;
        bool used = false;
    };

    std::size_t slotOf(const Key& key) const {
        return static_cast<std::size_t>(hashOf(key)) & (entries.size() - 1u);
    }

    std::vector<Entry> entries;
};

}

struct BufferCache::State {
    explicit State(std::size_t slots) : table(slots) {}
    Table<BufferKey, BufferValue> table;
};

BufferCache::BufferCache(std::size_t slots) : state(std::make_unique<State>(slots)) {}

BufferCache::~BufferCache() = default;

Hit BufferCache::Find(const BufferKey& key, const Proof& now, BufferValue& value) const {
    return state->table.Find(key, now, value);
}

void BufferCache::Keep(const BufferKey& key, const Proof& proof, BufferValue value) {
    state->table.Keep(key, proof, std::move(value));
}

ImageSetKey ImageSetKeyOf(std::uint64_t layoutId, const ShaderRecompiler::BlockLayout& layout, std::span<const std::uint32_t> words) {
    ImageSetKey key{.layout = layoutId, .digest = {LowSeed, HighSeed}};
    absorb(key.digest, layout.imageKey.size());
    for (const auto& entry : layout.imageKey) {
        const bool present = entry.word < words.size();
        const std::uint32_t masked = present ? words[entry.word] & entry.mask : 0u;
        absorb(key.digest, (std::uint64_t{entry.word} << 33u) | (std::uint64_t{present} << 32u) | masked);
    }
    return key;
}

struct ImageSetCache::State {
    explicit State(std::size_t slots) : table(slots) {}
    Table<ImageSetKey, ImageSetValue> table;
};

ImageSetCache::ImageSetCache(std::size_t slots) : state(std::make_unique<State>(slots)) {}

ImageSetCache::~ImageSetCache() = default;

Hit ImageSetCache::Find(const ImageSetKey& key, const Proof& now, ImageSetValue& value) const {
    return state->table.Find(key, now, value);
}

void ImageSetCache::Keep(const ImageSetKey& key, const Proof& proof, ImageSetValue value) {
    state->table.Keep(key, proof, std::move(value));
}

}
