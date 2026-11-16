#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_RESOURCECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_RESOURCECACHE_HPP

#include "Optimization/BlockLayout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::Core {

struct Proof { std::uint64_t content = 0; std::uint64_t registration = 0; std::uint64_t mapping = 0; std::uint64_t images = 0; bool operator==(const Proof&) const = default; };
struct BufferKey { std::uint64_t address = 0; std::uint64_t bytes = 0; std::uint32_t alignment = 0; std::uint32_t flags = 0; bool operator==(const BufferKey&) const = default; };
struct BufferValue { std::uint64_t deviceAddress = 0; std::uint32_t bytes = 0; std::uint64_t handle = 0; std::shared_ptr<void> owner; };

enum class Hit : std::uint8_t { Fresh, Stale, Miss };

class BufferCache {
public:
    explicit BufferCache(std::size_t slots = 1u << 16u);
    ~BufferCache();
    BufferCache(const BufferCache&) = delete;
    BufferCache& operator=(const BufferCache&) = delete;
    [[nodiscard]] Hit Find(const BufferKey& key, const Proof& now, BufferValue& value) const;
    void Keep(const BufferKey& key, const Proof& proof, BufferValue value);

private:
    struct State;
    std::unique_ptr<State> state;
};

struct ImageSetKey { std::uint64_t layout = 0; std::array<std::uint64_t, 2> digest{}; bool operator==(const ImageSetKey&) const = default; };

[[nodiscard]] ImageSetKey ImageSetKeyOf(std::uint64_t layoutId, const ShaderRecompiler::BlockLayout& layout, std::span<const std::uint32_t> words);

struct ImageSetValue { std::uint64_t set = 0; std::shared_ptr<void> owner; std::uint64_t lastSerial = 0; };

class ImageSetCache {
public:
    explicit ImageSetCache(std::size_t slots = 1u << 12u);
    ~ImageSetCache();
    ImageSetCache(const ImageSetCache&) = delete;
    ImageSetCache& operator=(const ImageSetCache&) = delete;
    [[nodiscard]] Hit Find(const ImageSetKey& key, const Proof& now, ImageSetValue& value) const;
    ImageSetValue Keep(const ImageSetKey& key, const Proof& proof, ImageSetValue value);

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
