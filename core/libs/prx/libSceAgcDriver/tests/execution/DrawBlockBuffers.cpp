#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawBlock.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#include "BlockAbi.hpp"
#include "Cover_vert_spv.h"
#include "DrawRecipeBuffers_spv.h"
#include "DrawBlockBuffersLow_spv.h"
#include "DrawBlockBuffersHigh_spv.h"
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
#include <cstring>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver {

struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};

}

namespace {

using namespace AgcDriver::Graphics;
using AgcDriver::Core::DrawBlockHost;
constexpr std::size_t Unit = 65536;
constexpr std::size_t Units = 4;

class Block {
public:
    Block() {
#ifdef _WIN32
        data = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Unit));
        Require(data != nullptr, "cannot reserve draw block test memory");
        GuestArena::GuestArenaCommit_nid_postfix(data, Bytes, PAGE_READWRITE, Bytes);
#else
        void* raw = mmap(nullptr, Bytes + Unit, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(raw != MAP_FAILED, "cannot map draw block test memory");
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
    static constexpr std::size_t Bytes = Units * Unit;
    std::byte* data = nullptr;
};

using Pixel = std::array<std::uint8_t, 4>;

std::vector<std::byte> ReadPixels(const ColorTarget& target) {
    std::vector<std::byte> pixels(64 * 64 * 4);
    ReadColorTarget(target, pixels);
    return pixels;
}

void CheckPixels(const ColorTarget& target, const Pixel& expected, const std::string& what) {
    const auto pixels = ReadPixels(target);
    for (std::size_t pixel = 0; pixel < pixels.size() / 4; ++pixel) {
        for (std::size_t channel = 0; channel < 4; ++channel) {
            const auto actual = std::to_integer<std::uint8_t>(pixels[pixel * 4 + channel]);
            Require(actual == expected[channel], what + ": pixel " + std::to_string(pixel) + " channel " + std::to_string(channel) + " is " + std::to_string(actual) + ", expected " + std::to_string(expected[channel]));
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

ShaderRecompiler::DescriptorBinding BufferBinding(ShaderRecompiler::DescriptorRole role) {
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = role;
    binding.descriptorSet = 0;
    binding.binding = role == ShaderRecompiler::DescriptorRole::GuestBuffers ? 0u : 1u;
    binding.count = 1;
    binding.bufferWritten = {false};
    return binding;
}

std::vector<std::uint32_t> Descriptor(std::uint64_t address, std::uint32_t bytes) {
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, bytes, 0x31000000u};
}

std::vector<std::uint32_t> Tint(const std::array<float, 4>& tint) {
    std::vector<std::uint32_t> words;
    for (const auto value : tint) words.push_back(std::bit_cast<std::uint32_t>(value));
    return words;
}

struct Scene {
    ShaderRecompiler::RecompileResult vertex;
    ShaderRecompiler::RecompileResult classic;
    ShaderRecompiler::RecompileResult block;
    std::uint32_t pushOffset = 0;
};

Scene MakeScene(const Context& context) {
    Scene scene;
    scene.pushOffset = context.graphicsPipelineLibrary ? PipelinePushSlotBytes : 0u;
    scene.vertex = Program(COVER_VERT_SPV, 21);
    scene.classic = Program(DRAW_RECIPE_BUFFERS_SPV, 22);
    scene.classic.bindings = {BufferBinding(ShaderRecompiler::DescriptorRole::GuestBuffers)};
    scene.block = scene.pushOffset != 0 ? Program(DRAW_BLOCK_BUFFERS_HIGH_SPV, 23) : Program(DRAW_BLOCK_BUFFERS_LOW_SPV, 23);
    scene.block.bindings = {BufferBinding(ShaderRecompiler::DescriptorRole::GuestBuffers), BufferBinding(ShaderRecompiler::DescriptorRole::FlattenedSrt)};
    scene.block.blockModule = true;
    scene.block.blockAddressDword = 0;
    scene.block.blockLayout.words = 4;
    scene.block.blockLayout.buffers.resize(1);
    scene.block.blockLayout.buffers.front().resource = 0;
    scene.block.blockLayout.buffers.front().descriptorWords = {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
    scene.block.pushConstants.assign(sizeof(std::uint64_t), std::byte{0});
    return scene;
}

void Select(Scene& scene, std::uint64_t address, std::uint32_t bytes, const std::array<float, 4>& tint) {
    scene.classic.bindings[0].guestDescriptor = Descriptor(address, bytes);
    scene.block.bindings[0].guestDescriptor = Descriptor(address, bytes);
    scene.block.bindings[1].guestDescriptor = Tint(tint);
}

std::array<CompiledShader, 2> Shaders(const Scene& scene, bool block) {
    return {CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &scene.vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, block ? &scene.block : &scene.classic, block ? scene.pushOffset : 0u}};
}

void CheckDraws(AgcDriver::VulkanDevice& device, const Block& block, bool verify) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    Require(context.recorder != nullptr && context.descriptorCache != nullptr, "draw block tests need a recorder and a descriptor cache");
    ForceDrawBlocks(true, verify);
    struct Restore {
        ~Restore() { UnforceDrawBlocks(); }
    } restore;
    auto scene = MakeScene(context);
    const auto state = TargetState(block);
    const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
    const std::array<float, 4> red{1, 0, 0, 1};
    const std::array<float, 4> green{0, 1, 0, 1};
    const std::array<float, 4> white{1, 1, 1, 1};
    const std::array<float, 4> half{1, 1, 0, 1};
    const auto first = block.Address(Unit);
    const auto second = block.Address(2 * Unit);
    const auto large = block.Address(3 * Unit);
    AgcDriver::GuestMemory::Write(first, std::as_bytes(std::span(red)));
    AgcDriver::GuestMemory::Write(second, std::as_bytes(std::span(green)));
    const auto clear = [&] {
        const std::vector<std::byte> cleared(64 * 64 * 4);
        AgcDriver::GuestMemory::Write(block.Address(), cleared);
    };
    const auto draw = [&](bool blocks, std::shared_ptr<const AgcDriver::DrawRecipe>* recipe = nullptr) {
        const auto shaders = Shaders(scene, blocks);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Draw(context, state, triangle, shaders, {}, recipe);
        }
        WaitForOptimizedPipelines(context.device);
    };
    const auto compare = [&](std::uint64_t address, std::uint32_t bytes, const Pixel& expected, const std::string& what) {
        Select(scene, address, bytes, white);
        clear();
        draw(false);
        const auto classic = ReadPixels(state.color);
        CheckPixels(state.color, expected, what + " through descriptors");
        clear();
        draw(true);
        const auto blocks = ReadPixels(state.color);
        CheckPixels(state.color, expected, what + " through the block");
        Require(classic == blocks, what + ": the pixels of the block draw differ from the descriptor draw");
    };
    const auto before = DrawBlockHost::Counted();
    compare(first, sizeof(red), {255, 0, 0, 255}, "the first buffer");
    compare(second, sizeof(green), {0, 255, 0, 255}, "the second buffer");
    Select(scene, first, 8, white);
    clear();
    draw(true);
    CheckPixels(state.color, {0, 0, 0, 0}, "a buffer shorter than the read");
    Select(scene, 0, 0, white);
    clear();
    draw(true);
    CheckPixels(state.color, {0, 0, 0, 0}, "an absent buffer");
    const auto served = DrawBlockHost::Counted();
    Require(served.draws >= before.draws + 4, "the block draws were not served by the block");
    Require(served.absent > before.absent, "the absent buffer was not counted");

    Select(scene, first, sizeof(red), white);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    clear();
    draw(true, &recipe);
    Require(recipe != nullptr, "the block draw did not produce a recipe");
    CheckPixels(state.color, {255, 0, 0, 255}, "the recipe's first draw");
    const auto setsBefore = context.descriptorCache->Counters();
    const auto pooledBefore = ShaderResources::DrawBindingsPooled();
    const auto repeat = [&](std::uint64_t address, const std::array<float, 4>& tint, const Pixel& expected, const std::string& what) {
        Select(scene, address, sizeof(red), tint);
        const auto shaders = Shaders(scene, true);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *recipe);
            Require(hit.recorded, what + ": the recipe refused the draw (" + DrawRecipeMissName(hit.miss) + ")");
        }
        CheckPixels(state.color, expected, what);
    };
    repeat(second, white, {0, 255, 0, 255}, "a recipe draw over a moved buffer");
    repeat(first, white, {255, 0, 0, 255}, "a recipe draw back on the first buffer");
    repeat(second, half, {0, 255, 0, 255}, "a recipe draw with changed words");
    AgcDriver::GuestMemory::Write(first, std::as_bytes(std::span(white)));
    repeat(first, half, {255, 255, 0, 255}, "a recipe draw whose words tint a rewritten buffer");
    AgcDriver::GuestMemory::Write(first, std::as_bytes(std::span(red)));
    const auto setsAfter = context.descriptorCache->Counters();
    Require(setsAfter.sets == setsBefore.sets && setsAfter.reusedSets == setsBefore.reusedSets, "a recipe draw served by the block allocated a descriptor set");
    Require(ShaderResources::DrawBindingsPooled().served == pooledBefore.served, "a recipe draw served by the block took rewritten bindings");

    constexpr std::uint32_t LargeBytes = 16384;
    constexpr std::uint32_t PlacedBytes = 32768;
    std::vector<float> filled(PlacedBytes / sizeof(float));
    for (std::size_t index = 0; index < filled.size(); ++index) filled[index] = index % 4 == 1 || index % 4 == 3 ? 1.0f : 0.0f;
    AgcDriver::GuestMemory::Write(large, std::as_bytes(std::span(filled)));
    const auto placedBefore = DrawBlockHost::Counted();
    for (int index = 0; index < 4; ++index) {
        Select(scene, large, PlacedBytes, white);
        const auto shaders = Shaders(scene, true);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *recipe);
            Require(hit.recorded, "a draw over a range above the ring size refused its recipe");
        }
        CheckPixels(state.color, {0, 255, 0, 255}, "a draw over a range above the ring size");
    }
    const auto placedAfter = DrawBlockHost::Counted();
    if (context.hostImportAlignment != 0) {
        Require(placedAfter.imported + placedAfter.mirrored >= placedBefore.imported + placedBefore.mirrored + 4, "a range above the ring size was copied instead of read in place");
        Require(placedAfter.copyBytes == placedBefore.copyBytes && placedAfter.apart == placedBefore.apart, "a range read in place was copied as well");
    }
    const auto ringBefore = DrawBlockHost::Counted();
    constexpr int Laps = 3;
    const int perChunk = static_cast<int>(AgcDriver::Core::BlockRing::ChunkBytes / LargeBytes);
    for (int index = 0; index < Laps * perChunk; ++index) {
        Select(scene, large, LargeBytes, white);
        const auto shaders = Shaders(scene, true);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *recipe);
            Require(hit.recorded, "a draw of the ring laps refused its recipe");
        }
        if (index % 64 == 63) CheckPixels(state.color, {0, 255, 0, 255}, "a draw of the ring laps");
    }
    CheckPixels(state.color, {0, 255, 0, 255}, "the last draw of the ring laps");
    const auto ringAfter = DrawBlockHost::Counted();
    Require(ringAfter.copyBytes - ringBefore.copyBytes >= static_cast<std::uint64_t>(Laps) * AgcDriver::Core::BlockRing::ChunkBytes - LargeBytes, "the ring did not take the copies of its laps");
    Require(ringAfter.ring.laps > ringBefore.ring.laps, "no chunk of the ring was written again after its batches ran");
    Require(ringAfter.ring.chunks <= 8, "the ring kept growing instead of reusing chunks whose batches ran (" + std::to_string(ringAfter.ring.chunks) + " chunks)");
    Require(ringAfter.apart == ringBefore.apart, "a copy that fits the ring went to a buffer apart");
    if (verify) {
        Require(ringAfter.verified > before.verified, "the verified mode checked no section");
        Require(ringAfter.mismatches == before.mismatches, "the verified mode found a difference");
    } else {
        Require(ringAfter.verified == before.verified, "sections were verified without the verified mode");
    }
    device.WaitIdle();
}

}

int main() {
    try {
        Block block;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (!AgcDriver::VulkanDeviceTestAccess::ContextOf(*device).bufferDeviceAddress) {
            std::cout << "skipped, the device has no buffer device address\n";
            return VulkanTestSkipped;
        }
        CheckDraws(*device, block, false);
        CheckDraws(*device, block, true);
        std::cout << "draws read their words and buffers through a block section: same pixels as through descriptors, no descriptor set per recipe draw, bounded and absent buffers read zero, ring chunks reused, verified mode clean\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
