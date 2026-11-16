#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostAllocator.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferCache.hpp"
#include "prx/libSceAgcDriver/Core/include/Back.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PassState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#include "Cover_frag_spv.h"
#include "Cover_vert_spv.h"
#include "VertexRange_vert_spv.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using namespace AgcDriver::Graphics;
constexpr std::size_t Unit = 65536;

class Block {
public:
    Block() {
#ifdef _WIN32
        data = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Unit));
        Require(data != nullptr, "cannot reserve recipe test memory");
        GuestArena::GuestArenaCommit_nid_postfix(data, Bytes, PAGE_READWRITE, Bytes);
#else
        void* raw = mmap(nullptr, Bytes + Unit, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(raw != MAP_FAILED, "cannot map recipe test memory");
        const auto begin = reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned = (begin + Unit - 1) & ~(static_cast<std::uintptr_t>(Unit) - 1);
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + Bytes != begin + Bytes + Unit) munmap(reinterpret_cast<void*>(aligned + Bytes), begin + Unit - aligned);
        data = reinterpret_cast<std::byte*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(data, Bytes);
#endif
        std::fill_n(data, Bytes, std::byte{0});
        GuestAllocations::Mutation().Add(data, Bytes, true, true, true);
    }
    ~Block() {
        GuestAllocations::Mutation().Remove(data);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(data, Bytes);
        GuestArena::GuestArenaRelease_nid_postfix(data, Bytes);
#else
        munmap(data, Bytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(data, Bytes);
#endif
    }
    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    std::uint64_t Address(std::size_t offset = 0) const { return reinterpret_cast<std::uintptr_t>(data + offset); }

private:
    static constexpr std::size_t Bytes = 3 * Unit;
    std::byte* data = nullptr;
};

void CheckPixels(const ColorTarget& target, const std::array<std::uint8_t, 4>& expected) {
    std::vector<std::byte> pixels(64 * 64 * 4);
    ReadColorTarget(target, pixels);
    for (std::size_t pixel = 0; pixel < pixels.size() / 4; ++pixel) {
        for (std::size_t channel = 0; channel < 4; ++channel) {
            const auto actual = std::to_integer<std::uint8_t>(pixels[pixel * 4 + channel]);
            Require(actual == expected[channel], "recipe pixel " + std::to_string(pixel) + " channel " + std::to_string(channel) + " is " + std::to_string(actual) + ", expected " + std::to_string(expected[channel]));
        }
    }
}

State TargetState(const Block& block) {
    State state{};
    state.stages.path = ShaderPath::Vertex;
    state.color = {block.Address(), {64, 64}, VK_FORMAT_R8G8B8A8_UNORM, Unit, 0xe4u, ColorTileMode::RenderTarget, 4};
    state.colors = {state.color};
    state.hasColorTarget = true;
    state.renderExtent = {64, 64};
    state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    state.viewport = {0, 64, 64, -64, 0, 1};
    state.scissor = {{0, 0}, {64, 64}};
    state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    state.blend.colorWriteMask = 15;
    state.blends = {state.blend};
    return state;
}

template <typename TSpirv>
ShaderRecompiler::RecompileResult Program(const TSpirv& spirv, std::uint64_t variant) {
    ShaderRecompiler::RecompileResult result;
    result.spirv = std::vector<std::uint32_t>(std::begin(spirv), std::end(spirv));
    result.variantId = variant;
    return result;
}

enum Counted : std::size_t { Replayed, Captured, Unstorable, Verified, Mismatches, KeptOnce, Lean, Recaptured, Refreshed };

std::vector<std::byte> Pixels(const ColorTarget& target) {
    std::vector<std::byte> pixels(64 * 64 * 4);
    ReadColorTarget(target, pixels);
    return pixels;
}

struct Held {
    Held() : replay(NewCoreReplay()) {}
    ~Held() { DeleteCoreReplay(replay); }
    Held(const Held&) = delete;
    Held& operator=(const Held&) = delete;
    CoreReplay* replay;
};

void CheckReplay(AgcDriver::VulkanDevice& device, const Block& block) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const auto offsets = block.Address(Unit + 4096);
    std::array<float, 20> words{};
    words[4] = -8.0f;
    AgcDriver::GuestMemory::Write(offsets, std::as_bytes(std::span(words)));
    auto vertex = Program(VERTEXRANGE_VERT_SPV, 34);
    vertex.vertexAttributes = {ShaderRecompiler::VertexAttribute{0, 4, {{static_cast<std::uint32_t>(offsets), (static_cast<std::uint32_t>(offsets >> 32u) & 0xffffu) | (32u << 16u), 3u, 77u << 12u}}, 0}};
    auto fragment = Program(COVER_FRAG_SPV, 35);
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    const auto state = TargetState(block);
    const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 600, 0, false};
    std::vector<std::byte> cleared(64 * 64 * 4);
    AgcDriver::GuestMemory::Write(block.Address(), cleared);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, shaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, "the draw of the replay test did not produce a recipe");
    const auto full = Pixels(state.color);
    constexpr std::uint64_t Identity = 7;
    constexpr int Repeats = 4;
    Held held;
    const auto start = CoreReplayCounted();
    AgcDriver::GuestMemory::Write(block.Address(), cleared);
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Require(DrawCoreReplayed(context, state, triangle, shaders, recipe, false, *held.replay, false, false, Identity).recorded, "the captured draw refused its recipe");
    }
    Require(CoreReplayStorable(*held.replay), "a draw with one copied vertex range was not storable");
    Require(CoreReplayRanges(*held.replay) == 1, "the captured draw does not hold its one input range");
    Require(CoreReplayMatches(*held.replay, context.device, Identity, recipe.get(), triangle, shaders), "the captured draw does not match itself");
    Require(!CoreReplayMatches(*held.replay, context.device, Identity + 1, recipe.get(), triangle, shaders), "the captured draw matches another variant");
    auto moved = triangle;
    moved.firstVertex = 1;
    Require(!CoreReplayMatches(*held.replay, context.device, Identity, recipe.get(), moved, shaders), "the captured draw matches other parameters");
    Require(Pixels(state.color) == full, "the captured draw and the full path drew different pixels");
    for (const bool verify : {false, true}) {
        AgcDriver::GuestMemory::Write(block.Address(), cleared);
        const auto before = CoreReplayCounted();
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            for (int index = 0; index < Repeats; ++index) Require(DrawCoreReplayed(context, state, triangle, shaders, recipe, false, *held.replay, true, verify, Identity).recorded, "a replayed draw refused its recipe");
        }
        const auto after = CoreReplayCounted();
        Require(after[Replayed] == before[Replayed] + Repeats, "the replayed draws were not counted");
        Require(after[Mismatches] == before[Mismatches], "a replayed draw differs from the full path");
        Require(!verify || after[Verified] >= before[Verified] + Repeats, "the verified replay checked no draw");
        Require(Pixels(state.color) == full, "a replayed draw and the full path drew different pixels");
    }
    Require(CoreReplayCounted()[Captured] == start[Captured] + 1, "the capture was not counted once");
    auto changed = words;
    changed[0] = 0.5f;
    AgcDriver::GuestMemory::Write(offsets, std::as_bytes(std::span(changed)));
    bool caught = false;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        try {
            static_cast<void>(DrawCoreReplayed(context, state, triangle, shaders, recipe, false, *held.replay, true, true, Identity));
        } catch (const std::runtime_error&) {
            caught = true;
        }
    }
    Require(caught, "the verified replay accepted held inputs whose guest bytes changed");
    AgcDriver::GuestMemory::Write(offsets, std::as_bytes(std::span(words)));
    device.WaitIdle();
}

void CheckBack(AgcDriver::VulkanDevice& device, const Block& block) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const auto offsets = block.Address(Unit + 8192);
    std::array<float, 20> words{};
    words[4] = -8.0f;
    AgcDriver::GuestMemory::Write(offsets, std::as_bytes(std::span(words)));
    auto vertex = Program(VERTEXRANGE_VERT_SPV, 36);
    vertex.vertexAttributes = {ShaderRecompiler::VertexAttribute{0, 4, {{static_cast<std::uint32_t>(offsets), (static_cast<std::uint32_t>(offsets >> 32u) & 0xffffu) | (32u << 16u), 3u, 77u << 12u}}, 0}};
    auto fragment = Program(COVER_FRAG_SPV, 37);
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    const auto state = TargetState(block);
    const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 600, 0, false};
    std::vector<std::byte> cleared(64 * 64 * 4);
    AgcDriver::GuestMemory::Write(block.Address(), cleared);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, shaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, "the draw of the back end replay test did not produce a recipe");
    const auto full = Pixels(state.color);
    AgcDriver::Core::SetBackModeForTest(1);
    for (const int mode : {1, 2}) {
        AgcDriver::Core::SetBackReplayForTest(mode);
        AgcDriver::GuestMemory::Write(block.Address(), cleared);
        const auto before = CoreReplayCounted();
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            for (int index = 0; index < 6; ++index) Require(AgcDriver::Core::RecordBack(device, state, triangle, shaders, recipe, 11), "the back end refused a draw it replays");
        }
        const auto after = CoreReplayCounted();
        Require(after[Mismatches] == before[Mismatches], "the back end replayed a draw that differs from the full path");
        Require(after[Replayed] + after[Captured] + after[Unstorable] + after[Refreshed] == before[Replayed] + before[Captured] + before[Unstorable] + before[Refreshed] + 6, "the back end did not offer each draw to the replay");
        Require(Pixels(state.color) == full, "the replaying back end and the full path drew different pixels");
    }
    AgcDriver::Core::SetBackReplayForTest(-1);
    AgcDriver::Core::SetBackModeForTest(-1);
    device.WaitIdle();
}

}

int main() {
    try {
        Block block;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        CheckReplay(*device, block);
        CheckBack(*device, block);
        std::cout << "the core back end replays the held inputs of a draw, draws the pixels of the full path and its verified mode refuses changed inputs\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
