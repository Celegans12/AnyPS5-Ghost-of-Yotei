#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostAllocator.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferCache.hpp"
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
#include "DrawRecipeBuffers_spv.h"
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

void CheckLayout(AgcDriver::VulkanDevice& device, const Block& block) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    const auto offsets = block.Address(Unit + 4096);
    std::array<float, 20> words{};
    words[4] = -8.0f;
    AgcDriver::GuestMemory::Write(offsets, std::as_bytes(std::span(words)));
    const auto attribute = [&](std::uint32_t stride) {
        return ShaderRecompiler::VertexAttribute{0, 4, {{static_cast<std::uint32_t>(offsets), (static_cast<std::uint32_t>(offsets >> 32u) & 0xffffu) | (stride << 16u), 3u, 77u << 12u}}, 0};
    };
    auto narrow = Program(VERTEXRANGE_VERT_SPV, 4);
    narrow.vertexAttributes = {attribute(16)};
    auto wide = narrow;
    wide.vertexAttributes = {attribute(32)};
    auto fragment = Program(COVER_FRAG_SPV, 5);
    const std::array narrowShaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &narrow, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    const std::array wideShaders{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &wide, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &fragment, 0}};
    const auto state = TargetState(block);
    const AgcDriver::Pm4::DrawParameters triangle{0, 3, 0, 1, 0, false};
    std::vector<std::byte> cleared(64 * 64 * 4);
    AgcDriver::GuestMemory::Write(block.Address(), cleared);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, narrowShaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, "the draw with a vertex stride of 16 did not produce a recipe");
    CheckPixels(state.color, {0, 0, 0, 0});
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        const auto hit = DrawWithRecipe(context, state, triangle, narrowShaders, {}, *recipe);
        Require(hit.recorded, "a draw with the vertex layout of its recipe refused it");
    }
    CheckPixels(state.color, {0, 0, 0, 0});
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        const auto hit = DrawWithRecipe(context, state, triangle, wideShaders, {}, *recipe);
        Require(!hit.recorded && hit.miss == DrawRecipeMiss::Layout, std::string("a draw whose vertex stride differs from its recipe's was not refused for its layout (miss: ") + DrawRecipeMissName(hit.miss) + ")");
    }
    CheckPixels(state.color, {0, 0, 0, 0});
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, wideShaders);
    }
    WaitForOptimizedPipelines(context.device);
    CheckPixels(state.color, {255, 255, 255, 255});
    device.WaitIdle();
}

void CheckLeanCommitAndPassState(AgcDriver::VulkanDevice& device, const Block& block) {
    struct Restore {
        ~Restore() {
            Recorder::SetLeanCommitForTest(-1);
            PassState::SetForTest(PassState::Mode::Environment);
        }
    } restore;
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    auto vertex = Program(COVER_VERT_SPV, 11);
    auto fragment = Program(DRAW_RECIPE_DATA_SPV, 13);
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
    PassState::SetForTest(PassState::Mode::Off);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, shaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, "the draw of the lean commit test did not produce a recipe");
    CheckPixels(state.color, {255, 0, 0, 255});
    constexpr int Repeats = 6;
    const auto repeat = [&](bool endRuns) {
        std::uint64_t allocations = 0;
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        for (int index = 0; index < Repeats; ++index) {
            select(index % 2 == 0);
            if (endRuns) BufferCache::EndDrawRun();
            const auto before = AgcDriver::HostAllocationsOnThread();
            const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *recipe);
            allocations += AgcDriver::HostAllocationsOnThread() - before;
            Require(hit.recorded, "a repeated draw refused its recipe");
        }
        return allocations;
    };
    const std::array<std::uint8_t, 4> last{255, 0, 0, 255};
    Recorder::SetLeanCommitForTest(0);
    const auto pooledBefore = ShaderResources::DrawBindingsPooled();
    repeat(false);
    CheckPixels(state.color, last);
    const auto former = repeat(false);
    CheckPixels(state.color, last);
    Require(ShaderResources::DrawBindingsPooled().served == pooledBefore.served, "the former commit took bindings from the pool of the lean commit");
    Recorder::SetLeanCommitForTest(1);
    repeat(false);
    CheckPixels(state.color, last);
    const auto lean = repeat(false);
    CheckPixels(state.color, last);
    Require(ShaderResources::DrawBindingsPooled().served > pooledBefore.served, "the lean commit did not take the bindings of its draws from the pool of its batch");
    if (AgcDriver::HostAllocationsCounted()) Require(lean < former, "the lean commit allocates as much as the former one (" + std::to_string(lean) + " against " + std::to_string(former) + " over " + std::to_string(Repeats) + " draws)");
    const auto off = PassState::Counted();
    repeat(false);
    Require(PassState::Counted().draws == off.draws, "draws were counted by the pass state while it is off");
    PassState::SetForTest(PassState::Mode::On);
    const auto before = PassState::Counted();
    repeat(false);
    CheckPixels(state.color, last);
    const auto served = PassState::Counted();
    const auto servedBy = [&](PassState::Served kind) { return served.served[static_cast<std::size_t>(kind)] - before.served[static_cast<std::size_t>(kind)]; };
    Require(served.draws == before.draws + Repeats, "the pass state did not count its draws");
    Require(servedBy(PassState::Served::Access) != 0, "no draw whose template the pass knew continued it without the full comparison");
    Require(servedBy(PassState::Served::Targets) != 0, "no draw of a run reused the targets of the run");
    Require(servedBy(PassState::Served::Kept) != 0, "no draw of a run left its objects to the batch that already keeps them");
    repeat(true);
    CheckPixels(state.color, last);
    const auto ended = PassState::Counted();
    Require(ended.served[static_cast<std::size_t>(PassState::Served::Targets)] == served.served[static_cast<std::size_t>(PassState::Served::Targets)], "a draw reused the targets of a run that an ordered item ended");
    Require(ended.runs >= served.runs + Repeats, "an ordered item between two draws did not begin a new run");
    PassState::SetForTest(PassState::Mode::Verify);
    repeat(false);
    CheckPixels(state.color, last);
    const auto verified = PassState::Counted();
    Require(verified.mismatches == ended.mismatches, "the verified pass state found a difference");
    Require(verified.verified != ended.verified, "the verified pass state checked no answer");
    device.WaitIdle();
}

bool CheckBindings(AgcDriver::VulkanDevice& device, const Block& block, bool dataWords) {
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    if (!dataWords && context.hostImportAlignment == 0) {
        std::cout << "skipped, host imports are unavailable\n";
        return false;
    }
    Require(context.recorder != nullptr && context.descriptorCache != nullptr, "recipe tests need a recorder and descriptor cache");
    const auto program = [](const auto& spirv, std::uint64_t variant) { return Program(spirv, variant); };
    auto vertex = program(COVER_VERT_SPV, 1);
    auto fragment = dataWords ? program(DRAW_RECIPE_DATA_SPV, 3) : program(DRAW_RECIPE_BUFFERS_SPV, 2);
    if (dataWords) fragment.shaderDataDwords = 4;
    ShaderRecompiler::DescriptorBinding binding;
    binding.kind = ShaderRecompiler::DescriptorKind::StorageBuffer;
    binding.role = dataWords ? ShaderRecompiler::DescriptorRole::ShaderData : ShaderRecompiler::DescriptorRole::GuestBuffers;
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
    AgcDriver::GuestMemory::Write(block.Address(Unit), std::as_bytes(std::span(red)));
    AgcDriver::GuestMemory::Write(block.Address(2 * Unit), std::as_bytes(std::span(green)));
    const auto select = [&](bool second) {
        auto& words = fragment.bindings.front().guestDescriptor;
        if (dataWords) {
            const auto& color = second ? green : red;
            words.clear();
            for (const auto value : color) words.push_back(std::bit_cast<std::uint32_t>(value));
        } else {
            const auto address = block.Address(second ? 2 * Unit : Unit);
            words = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, sizeof(red), 0x31000000u};
        }
    };
    select(false);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, shaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, "the initial draw did not produce a recipe");
    CheckPixels(state.color, {255, 0, 0, 255});
    if (!dataWords) {
        select(true);
        std::shared_ptr<const AgcDriver::DrawRecipe> relocated;
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Draw(context, state, triangle, shaders, {}, &relocated);
        }
        WaitForOptimizedPipelines(context.device);
        Require(relocated != nullptr, "a draw over relocated buffers did not produce a recipe");
        CheckPixels(state.color, {0, 255, 0, 255});
        select(false);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *relocated);
            Require(hit.recorded, "a recipe made over relocated buffers refused a safe buffer change");
        }
        CheckPixels(state.color, {255, 0, 0, 255});
    }
    for (const bool second : {true, true, false}) {
        select(second);
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *recipe);
            Require(hit.recorded, "a safe buffer change refused its recipe");
        }
        CheckPixels(state.color, second ? std::array<std::uint8_t, 4>{0, 255, 0, 255} : std::array<std::uint8_t, 4>{255, 0, 0, 255});
    }
    for (const bool second : {true, false, true}) {
        select(second);
        auto patched = fragment;
        select(!second);
        const std::array copies{CompiledShader{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, CompiledShader{ShaderRecompiler::ShaderStage::Fragment, &patched, 0}};
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const auto hit = DrawWithRecipe(context, state, triangle, copies, {}, *recipe);
            Require(hit.recorded, "a copy of the fragment program whose constants moved, as a reuse of a remembered stage patches it, refused the recipe");
        }
        CheckPixels(state.color, second ? std::array<std::uint8_t, 4>{0, 255, 0, 255} : std::array<std::uint8_t, 4>{255, 0, 0, 255});
    }
    select(false);
    const auto refused = [&] {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *recipe);
        Require(!hit.recorded && hit.miss == DrawRecipeMiss::Buffers, "an unsafe binding change did not fall back to the ordinary draw path");
    };
    if (dataWords) {
        fragment.bindings.front().guestDescriptor.pop_back();
        --fragment.shaderDataDwords;
        refused();
    } else {
        select(true);
        fragment.bindings.front().bufferWritten = {true};
        refused();
        fragment.bindings.front().bufferWritten = {false};
        fragment.bindings.front().guestDescriptor = {0x1000u, 0, sizeof(red), 0x31000000u};
        refused();
        fragment.bindings.front().guestDescriptor = {0, 0, 0, 0};
        refused();
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            Draw(context, state, triangle, shaders);
        }
        CheckPixels(state.color, {0, 0, 0, 0});
    }
    device.WaitIdle();
    return true;
}

}

int main(int argc, char** argv) {
    try {
        Block block;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        const bool dataWords = argc == 2 && std::string_view(argv[1]) == "--data";
        if (!CheckBindings(*device, block, dataWords)) return VulkanTestSkipped;
        CheckLayout(*device, block);
        CheckLeanCommitAndPassState(*device, block);
        std::cout << "draw recipes consume relocated buffers and changed shader data, also through patched copies of their programs, and refuse another vertex layout\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
