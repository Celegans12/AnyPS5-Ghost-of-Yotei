#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKRING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKRING_HPP

#include <cstddef>
#include <cstdint>
#include <memory>

namespace AgcDriver::Core {

struct RingChunk { std::byte* bytes = nullptr; std::uint64_t deviceAddress = 0; std::shared_ptr<void> owner; };

struct RingSource {
    RingChunk (*allocate)(void* context, std::size_t bytes) = nullptr;
    bool (*executed)(void* context, std::uint64_t serial) = nullptr;
    void* context = nullptr;
};

struct RingSlice { std::byte* bytes = nullptr; std::uint64_t deviceAddress = 0; std::uint32_t chunk = 0; std::uint32_t lap = 0; };

class BlockRing {
public:
    static constexpr std::size_t ChunkBytes = std::size_t{4} << 20u;
    static constexpr std::size_t MaxChunks = 64;
    explicit BlockRing(RingSource source);
    ~BlockRing();
    BlockRing(const BlockRing&) = delete;
    BlockRing& operator=(const BlockRing&) = delete;
    [[nodiscard]] RingSlice Take(std::size_t bytes, std::size_t alignment);
    void Posted(const RingSlice& slice);
    void Emitted(const RingSlice& slice, std::uint64_t serial);
    [[nodiscard]] bool Live(const RingSlice& slice) const;
    struct Counts { std::uint64_t chunks = 0, laps = 0, full = 0, bytes = 0; };
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
