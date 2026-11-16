#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DRAWMEMO_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DRAWMEMO_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace AgcDriver::Core {

struct ReplayKey {
    std::uint64_t variant = 0, images = 0, inputs = 0, targets = 0;
    bool operator==(const ReplayKey&) const = default;
};

struct DrawProof {
    std::array<GuestMemory::ContentToken, 6> ranges{};
    std::array<std::uint64_t, 6> rangeAddress{};
    std::array<std::uint32_t, 6> rangeBytes{};
    std::array<std::uint64_t, 9> targetGeneration{};
    std::uint64_t depthGeneration = 0, imageGeneration = 0, mappingSerial = 0, shaderGeneration = 0, device = 0;
};

enum class ReplayMiss : std::uint8_t { None, Absent, Range, PendingGpuWrite, Target, Depth, Image, Mapping, Shader, Device };

constexpr std::size_t ReplayMissKinds = static_cast<std::size_t>(ReplayMiss::Device) + 1u;

struct ReplayEntry {
    ReplayKey key;
    DrawProof proof;
    std::array<std::uint64_t, 16> resolved{};
    std::uint64_t lastBatch = 0;
};

struct ReplayNow {
    std::array<std::uint64_t, 9> targetGeneration{};
    std::uint64_t depthGeneration = 0, imageGeneration = 0, mappingSerial = 0, shaderGeneration = 0, device = 0;
    bool (*stillCurrent)(void* context, std::uint64_t address, std::size_t bytes, const GuestMemory::ContentToken& token) = nullptr;
    bool (*pendingGpuWrite)(void* context, std::uint64_t address, std::size_t bytes) = nullptr;
    void* context = nullptr;
};

struct ReplayRelease {
    void (*release)(void* context, const ReplayEntry& entry) noexcept = nullptr;
    void* context = nullptr;
};

class DrawMemo {
public:
    static constexpr std::size_t Slots = 65536;
    static constexpr std::size_t ResolvedWords = 16;
    explicit DrawMemo(ReplayRelease release = {});
    ~DrawMemo();
    DrawMemo(const DrawMemo&) = delete;
    DrawMemo& operator=(const DrawMemo&) = delete;
    [[nodiscard]] ReplayEntry* Find(const ReplayKey& key);
    [[nodiscard]] ReplayMiss Fresh(const ReplayEntry& entry, const ReplayNow& now);
    ReplayEntry* Store(const ReplayKey& key, const DrawProof& proof, std::span<const std::uint64_t> resolved, std::uint64_t batch);
    void Retire(std::uint64_t finishedBatch);
    [[nodiscard]] static std::size_t SlotOf(const ReplayKey& key);
    struct Counts {
        std::uint64_t finds = 0, freshChecks = 0, stores = 0, evicted = 0, replaced = 0, refreshed = 0, waiting = 0, released = 0, refused = 0;
        std::array<std::uint64_t, ReplayMissKinds> outcomes{};
    };
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
