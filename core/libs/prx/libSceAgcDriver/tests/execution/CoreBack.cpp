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
#include "DrawRecipeData_spv.h"
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

enum Counted : std::size_t { Draws, Continued, Proofs, Verified, Mismatches, Misses, Inputs, InputsHeld, InputsInPlace, Bindings, Notes, Marks };

std::vector<std::byte> Pixels(const ColorTarget& target) {
    std::vector<std::byte> pixels(64 * 64 * 4);
    ReadColorTarget(target, pixels);
    return pixels;
}

void CheckRuns(AgcDriver::VulkanDevice& device, const Block& block) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    auto vertex = Program(COVER_VERT_SPV, 21);
    auto fragment = Program(DRAW_RECIPE_DATA_SPV, 23);
    fragment.shaderDataDwords = 4;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = ShaderRecompiler::DescriptorRole::ShaderData;
    binding.descriptorSet = 0;
    binding.binding = 0;
    binding.count = 1;
    binding.bufferWritten = {false};
    fragment.bindings = {binding};
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    const auto state = TargetState(block);
    const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
    const std::array<float, 4> red{1, 0, 0, 1};
    const std::array<float, 4> green{0, 1, 0, 1};
    const auto select = [&](bool second) {
        auto& words = fragment.bindings.front().guestDescriptor;
        words.clear();
        for (const auto value : second ? green : red) words.push_back(std::bit_cast<std::uint32_t>(value));
    };
    select(false);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, shaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, "the draw of the light back end test did not produce a recipe");
    CheckPixels(state.color, {255, 0, 0, 255});
    constexpr int Repeats = 6;
    enum class Between { Nothing, RunEnd, ProofBreak };
    const auto repeat = [&](bool verify, Between between) {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        for (int index = 0; index < Repeats; ++index) {
            select(index % 2 == 0);
            if (between == Between::RunEnd) BufferCache::EndDrawRun();
            if (between == Between::ProofBreak) AgcDriver::GuestMemory::BreakProofs(AgcDriver::GuestMemory::ProofBreak::Store);
            const auto hit = DrawCoreWithRecipe(context, state, triangle, shaders, *recipe, verify);
            Require(hit.recorded, "a draw of the light back end refused its recipe");
        }
    };
    const std::array<std::uint8_t, 4> last{255, 0, 0, 255};
    const auto start = CoreBackCounted();
    repeat(false, Between::Nothing);
    CheckPixels(state.color, last);
    const auto run = CoreBackCounted();
    Require(run[Draws] == start[Draws] + Repeats, "the light back end did not count its draws");
    Require(run[Continued] != start[Continued], "no draw of the light back end continued the run of the draw before it");
    Require(run[Proofs] != start[Proofs], "no draw of the light back end inherited the proof of its template");
    Require(run[Misses] == start[Misses], "a draw of the light back end was left to the full path");
    repeat(false, Between::RunEnd);
    CheckPixels(state.color, last);
    const auto ended = CoreBackCounted();
    Require(ended[Continued] == run[Continued], "a draw continued a run that an ordered item ended");
    repeat(false, Between::ProofBreak);
    CheckPixels(state.color, last);
    const auto broken = CoreBackCounted();
    Require(broken[Proofs] == ended[Proofs], "a draw inherited the proof of its template across a break of the proofs");
    Require(broken[Continued] == ended[Continued], "a draw continued a run across a break of the proofs");
    repeat(true, Between::Nothing);
    CheckPixels(state.color, last);
    const auto verified = CoreBackCounted();
    Require(verified[Mismatches] == broken[Mismatches], "the verified light back end found a difference");
    Require(verified[Verified] != broken[Verified], "the verified light back end checked no answer");
    device.WaitIdle();
}

void CheckInputs(AgcDriver::VulkanDevice& device, const Block& block) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const auto offsets = block.Address(Unit + 4096);
    std::array<float, 20> words{};
    words[4] = -8.0f;
    AgcDriver::GuestMemory::Write(offsets, std::as_bytes(std::span(words)));
    auto vertex = Program(VERTEXRANGE_VERT_SPV, 24);
    vertex.vertexAttributes = {ShaderRecompiler::VertexAttribute{0, 4, {{static_cast<std::uint32_t>(offsets), (static_cast<std::uint32_t>(offsets >> 32u) & 0xffffu) | (32u << 16u), 3u, 77u << 12u}}, 0}};
    auto fragment = Program(COVER_FRAG_SPV, 25);
    const std::array shaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    const auto state = TargetState(block);
    const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
    std::vector<std::byte> cleared(64 * 64 * 4);
    AgcDriver::GuestMemory::Write(block.Address(), cleared);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, shaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, "the draw with a vertex input did not produce a recipe");
    const auto full = Pixels(state.color);
    constexpr int Repeats = 4;
    for (const bool verify : {false, true}) {
        AgcDriver::GuestMemory::Write(block.Address(), cleared);
        const auto before = CoreBackCounted();
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            for (int index = 0; index < Repeats; ++index) {
                const auto hit = DrawCoreWithRecipe(context, state, triangle, shaders, *recipe, verify);
                Require(hit.recorded, "a draw with a vertex input refused its recipe in the light back end");
            }
        }
        const auto after = CoreBackCounted();
        Require(after[Inputs] == before[Inputs] + Repeats, "the light back end did not count one input range for each draw");
        Require(after[Mismatches] == before[Mismatches], "the light back end found a difference in a draw input");
        Require(Pixels(state.color) == full, "the light back end and the full path drew different pixels from the same vertex input");
    }
    device.WaitIdle();
}

}

int main() {
    try {
        Block block;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        CheckRuns(*device, block);
        CheckInputs(*device, block);
        std::cout << "the light back end continues runs of draws, ends them at ordered items and proof breaks, and draws the pixels of the full path\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
