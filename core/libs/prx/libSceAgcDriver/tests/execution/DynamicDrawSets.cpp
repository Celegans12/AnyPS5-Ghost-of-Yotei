#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ColorTargetTransfer.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
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
constexpr std::size_t Unit = 65536;

constexpr std::uint32_t COVER_VERT_SPV[] = {
0x07230203u,0x00010300u,0x0008000bu,0x00000028u,0x00000000u,0x00020011u,0x00000001u,0x0006000bu,
0x00000001u,0x4c534c47u,0x6474732eu,0x3035342eu,0x00000000u,0x0003000eu,0x00000000u,0x00000001u,
0x0007000fu,0x00000000u,0x00000004u,0x6e69616du,0x00000000u,0x0000000cu,0x0000001au,0x00030003u,
0x00000002u,0x000001c2u,0x00040005u,0x00000004u,0x6e69616du,0x00000000u,0x00050005u,0x00000009u,
0x69736f70u,0x6e6f6974u,0x00000000u,0x00060005u,0x0000000cu,0x565f6c67u,0x65747265u,0x646e4978u,
0x00007865u,0x00060005u,0x00000018u,0x505f6c67u,0x65567265u,0x78657472u,0x00000000u,0x00060006u,
0x00000018u,0x00000000u,0x505f6c67u,0x7469736fu,0x006e6f69u,0x00030005u,0x0000001au,0x00000000u,
0x00040047u,0x0000000cu,0x0000000bu,0x0000002au,0x00030047u,0x00000018u,0x00000002u,0x00050048u,
0x00000018u,0x00000000u,0x0000000bu,0x00000000u,0x00020013u,0x00000002u,0x00030021u,0x00000003u,
0x00000002u,0x00030016u,0x00000006u,0x00000020u,0x00040017u,0x00000007u,0x00000006u,0x00000002u,
0x00040020u,0x00000008u,0x00000007u,0x00000007u,0x00040015u,0x0000000au,0x00000020u,0x00000001u,
0x00040020u,0x0000000bu,0x00000001u,0x0000000au,0x0004003bu,0x0000000bu,0x0000000cu,0x00000001u,
0x0004002bu,0x0000000au,0x0000000eu,0x00000001u,0x0004002bu,0x0000000au,0x00000010u,0x00000002u,
0x00040017u,0x00000017u,0x00000006u,0x00000004u,0x0003001eu,0x00000018u,0x00000017u,0x00040020u,
0x00000019u,0x00000003u,0x00000018u,0x0004003bu,0x00000019u,0x0000001au,0x00000003u,0x0004002bu,
0x0000000au,0x0000001bu,0x00000000u,0x0004002bu,0x00000006u,0x0000001du,0x40000000u,0x0004002bu,
0x00000006u,0x0000001fu,0x3f800000u,0x0004002bu,0x00000006u,0x00000022u,0x3f000000u,0x00040020u,
0x00000026u,0x00000003u,0x00000017u,0x00050036u,0x00000002u,0x00000004u,0x00000000u,0x00000003u,
0x000200f8u,0x00000005u,0x0004003bu,0x00000008u,0x00000009u,0x00000007u,0x0004003du,0x0000000au,
0x0000000du,0x0000000cu,0x000500c4u,0x0000000au,0x0000000fu,0x0000000du,0x0000000eu,0x000500c7u,
0x0000000au,0x00000011u,0x0000000fu,0x00000010u,0x0004006fu,0x00000006u,0x00000012u,0x00000011u,
0x0004003du,0x0000000au,0x00000013u,0x0000000cu,0x000500c7u,0x0000000au,0x00000014u,0x00000013u,
0x00000010u,0x0004006fu,0x00000006u,0x00000015u,0x00000014u,0x00050050u,0x00000007u,0x00000016u,
0x00000012u,0x00000015u,0x0003003eu,0x00000009u,0x00000016u,0x0004003du,0x00000007u,0x0000001cu,
0x00000009u,0x0005008eu,0x00000007u,0x0000001eu,0x0000001cu,0x0000001du,0x00050050u,0x00000007u,
0x00000020u,0x0000001fu,0x0000001fu,0x00050083u,0x00000007u,0x00000021u,0x0000001eu,0x00000020u,
0x00050051u,0x00000006u,0x00000023u,0x00000021u,0x00000000u,0x00050051u,0x00000006u,0x00000024u,
0x00000021u,0x00000001u,0x00070050u,0x00000017u,0x00000025u,0x00000023u,0x00000024u,0x00000022u,
0x0000001fu,0x00050041u,0x00000026u,0x00000027u,0x0000001au,0x0000001bu,0x0003003eu,0x00000027u,
0x00000025u,0x000100fdu,0x00010038u,
};

constexpr std::uint32_t DRAW_RECIPE_BUFFERS_SPV[] = {
0x07230203u,0x00010300u,0x0008000bu,0x00000022u,0x00000000u,0x00020011u,0x00000001u,0x0006000bu,
0x00000001u,0x4c534c47u,0x6474732eu,0x3035342eu,0x00000000u,0x0003000eu,0x00000000u,0x00000001u,
0x0006000fu,0x00000004u,0x00000004u,0x6e69616du,0x00000000u,0x00000009u,0x00030010u,0x00000004u,
0x00000007u,0x00030003u,0x00000002u,0x000001c2u,0x00040005u,0x00000004u,0x6e69616du,0x00000000u,
0x00050005u,0x00000009u,0x7074756fu,0x6f437475u,0x00726f6cu,0x00040005u,0x0000000cu,0x72756f53u,
0x00006563u,0x00050006u,0x0000000cu,0x00000000u,0x64726f77u,0x00000073u,0x00040005u,0x00000010u,
0x72756f73u,0x00006563u,0x00040047u,0x00000009u,0x0000001eu,0x00000000u,0x00040047u,0x0000000bu,
0x00000006u,0x00000004u,0x00030047u,0x0000000cu,0x00000002u,0x00040048u,0x0000000cu,0x00000000u,
0x00000018u,0x00050048u,0x0000000cu,0x00000000u,0x00000023u,0x00000000u,0x00030047u,0x00000010u,
0x00000018u,0x00040047u,0x00000010u,0x00000021u,0x00000000u,0x00040047u,0x00000010u,0x00000022u,
0x00000000u,0x00020013u,0x00000002u,0x00030021u,0x00000003u,0x00000002u,0x00030016u,0x00000006u,
0x00000020u,0x00040017u,0x00000007u,0x00000006u,0x00000004u,0x00040020u,0x00000008u,0x00000003u,
0x00000007u,0x0004003bu,0x00000008u,0x00000009u,0x00000003u,0x00040015u,0x0000000au,0x00000020u,
0x00000000u,0x0003001du,0x0000000bu,0x0000000au,0x0003001eu,0x0000000cu,0x0000000bu,0x0004002bu,
0x0000000au,0x0000000du,0x00000001u,0x0004001cu,0x0000000eu,0x0000000cu,0x0000000du,0x00040020u,
0x0000000fu,0x0000000cu,0x0000000eu,0x0004003bu,0x0000000fu,0x00000010u,0x0000000cu,0x00040015u,
0x00000011u,0x00000020u,0x00000001u,0x0004002bu,0x00000011u,0x00000012u,0x00000000u,0x00040020u,
0x00000013u,0x0000000cu,0x0000000au,0x0004002bu,0x00000011u,0x00000016u,0x00000001u,0x0004002bu,
0x00000011u,0x00000019u,0x00000002u,0x0004002bu,0x00000011u,0x0000001cu,0x00000003u,0x00040017u,
0x0000001fu,0x0000000au,0x00000004u,0x00050036u,0x00000002u,0x00000004u,0x00000000u,0x00000003u,
0x000200f8u,0x00000005u,0x00070041u,0x00000013u,0x00000014u,0x00000010u,0x00000012u,0x00000012u,
0x00000012u,0x0004003du,0x0000000au,0x00000015u,0x00000014u,0x00070041u,0x00000013u,0x00000017u,
0x00000010u,0x00000012u,0x00000012u,0x00000016u,0x0004003du,0x0000000au,0x00000018u,0x00000017u,
0x00070041u,0x00000013u,0x0000001au,0x00000010u,0x00000012u,0x00000012u,0x00000019u,0x0004003du,
0x0000000au,0x0000001bu,0x0000001au,0x00070041u,0x00000013u,0x0000001du,0x00000010u,0x00000012u,
0x00000012u,0x0000001cu,0x0004003du,0x0000000au,0x0000001eu,0x0000001du,0x00070050u,0x0000001fu,
0x00000020u,0x00000015u,0x00000018u,0x0000001bu,0x0000001eu,0x0004007cu,0x00000007u,0x00000021u,
0x00000020u,0x0003003eu,0x00000009u,0x00000021u,0x000100fdu,0x00010038u,
};

constexpr std::uint32_t DRAW_RECIPE_DATA_SPV[] = {
0x07230203u,0x00010300u,0x0008000bu,0x00000020u,0x00000000u,0x00020011u,0x00000001u,0x0006000bu,
0x00000001u,0x4c534c47u,0x6474732eu,0x3035342eu,0x00000000u,0x0003000eu,0x00000000u,0x00000001u,
0x0006000fu,0x00000004u,0x00000004u,0x6e69616du,0x00000000u,0x00000009u,0x00030010u,0x00000004u,
0x00000007u,0x00030003u,0x00000002u,0x000001c2u,0x00040005u,0x00000004u,0x6e69616du,0x00000000u,
0x00050005u,0x00000009u,0x7074756fu,0x6f437475u,0x00726f6cu,0x00040005u,0x0000000cu,0x72756f53u,
0x00006563u,0x00050006u,0x0000000cu,0x00000000u,0x64726f77u,0x00000073u,0x00040005u,0x0000000eu,
0x72756f73u,0x00006563u,0x00040047u,0x00000009u,0x0000001eu,0x00000000u,0x00040047u,0x0000000bu,
0x00000006u,0x00000004u,0x00030047u,0x0000000cu,0x00000002u,0x00040048u,0x0000000cu,0x00000000u,
0x00000018u,0x00050048u,0x0000000cu,0x00000000u,0x00000023u,0x00000000u,0x00030047u,0x0000000eu,
0x00000018u,0x00040047u,0x0000000eu,0x00000021u,0x00000000u,0x00040047u,0x0000000eu,0x00000022u,
0x00000000u,0x00020013u,0x00000002u,0x00030021u,0x00000003u,0x00000002u,0x00030016u,0x00000006u,
0x00000020u,0x00040017u,0x00000007u,0x00000006u,0x00000004u,0x00040020u,0x00000008u,0x00000003u,
0x00000007u,0x0004003bu,0x00000008u,0x00000009u,0x00000003u,0x00040015u,0x0000000au,0x00000020u,
0x00000000u,0x0003001du,0x0000000bu,0x0000000au,0x0003001eu,0x0000000cu,0x0000000bu,0x00040020u,
0x0000000du,0x0000000cu,0x0000000cu,0x0004003bu,0x0000000du,0x0000000eu,0x0000000cu,0x00040015u,
0x0000000fu,0x00000020u,0x00000001u,0x0004002bu,0x0000000fu,0x00000010u,0x00000000u,0x00040020u,
0x00000011u,0x0000000cu,0x0000000au,0x0004002bu,0x0000000fu,0x00000014u,0x00000001u,0x0004002bu,
0x0000000fu,0x00000017u,0x00000002u,0x0004002bu,0x0000000fu,0x0000001au,0x00000003u,0x00040017u,
0x0000001du,0x0000000au,0x00000004u,0x00050036u,0x00000002u,0x00000004u,0x00000000u,0x00000003u,
0x000200f8u,0x00000005u,0x00060041u,0x00000011u,0x00000012u,0x0000000eu,0x00000010u,0x00000010u,
0x0004003du,0x0000000au,0x00000013u,0x00000012u,0x00060041u,0x00000011u,0x00000015u,0x0000000eu,
0x00000010u,0x00000014u,0x0004003du,0x0000000au,0x00000016u,0x00000015u,0x00060041u,0x00000011u,
0x00000018u,0x0000000eu,0x00000010u,0x00000017u,0x0004003du,0x0000000au,0x00000019u,0x00000018u,
0x00060041u,0x00000011u,0x0000001bu,0x0000000eu,0x00000010u,0x0000001au,0x0004003du,0x0000000au,
0x0000001cu,0x0000001bu,0x00070050u,0x0000001du,0x0000001eu,0x00000013u,0x00000016u,0x00000019u,
0x0000001cu,0x0004007cu,0x00000007u,0x0000001fu,0x0000001eu,0x0003003eu,0x00000009u,0x0000001fu,
0x000100fdu,0x00010038u,
};

class Block {
public:
    Block() {
#ifdef _WIN32
        data = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(Bytes, Unit));
        Require(data != nullptr, "cannot reserve dynamic draw set test memory");
        GuestArena::GuestArenaCommit_nid_postfix(data, Bytes, PAGE_READWRITE, Bytes);
#else
        void* raw = mmap(nullptr, Bytes + Unit, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(raw != MAP_FAILED, "cannot map dynamic draw set test memory");
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
    static constexpr std::size_t Bytes = 4 * Unit;
    std::byte* data = nullptr;
};

using Pixel = std::array<std::uint8_t, 4>;
constexpr std::array<Pixel, 3> Colors{{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}}};
constexpr std::array<std::array<float, 4>, 3> Values{{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}}};

void CheckPixels(const ColorTarget& target, const Pixel& expected, const std::string& what) {
    std::vector<std::byte> pixels(64 * 64 * 4);
    ReadColorTarget(target, pixels);
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

struct Counts {
    std::uint64_t reused = DrawSuite::Counted(DrawSuite::Count::SetsReused);
    std::uint64_t written = DrawSuite::Counted(DrawSuite::Count::SetsWritten);
    std::uint64_t marksServed = DrawSuite::Counted(DrawSuite::Count::MarksServed);
    std::uint64_t marksFull = DrawSuite::Counted(DrawSuite::Count::MarksFull);
    std::uint64_t movedServed = DrawSuite::Counted(DrawSuite::Count::MovedServed);
};

void CheckDraws(AgcDriver::VulkanDevice& device, const Block& block, bool dataWords, bool dynamic, int lean, std::uint64_t variant) {
    struct Restore {
        ~Restore() {
            Recorder::SetLeanCommitForTest(-1);
            DrawSuite::SetForTest(0);
        }
    } restore;
    DrawSuite::SetForTest(dynamic ? 1 : -1);
    Recorder::SetLeanCommitForTest(lean);
    const std::string mode = std::string(dataWords ? "shader data" : "moved buffers") + (dynamic ? ", dynamic sets" : ", rewritten sets") + (lean != 0 ? ", lean commit" : ", former commit");
    auto context = AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
    Require(context.recorder != nullptr && context.descriptorCache != nullptr, "dynamic draw set tests need a recorder and descriptor cache");
    auto vertex = Program(COVER_VERT_SPV, variant);
    auto fragment = dataWords ? Program(DRAW_RECIPE_DATA_SPV, variant + 1) : Program(DRAW_RECIPE_BUFFERS_SPV, variant + 1);
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
    for (std::size_t color = 0; color < Values.size(); ++color) AgcDriver::GuestMemory::Write(block.Address(Unit + color * 4096), std::as_bytes(std::span(Values[color])));
    const auto select = [&](std::size_t color) {
        auto& words = fragment.bindings.front().guestDescriptor;
        if (dataWords) {
            words.clear();
            for (const auto value : Values[color]) words.push_back(std::bit_cast<std::uint32_t>(value));
        } else {
            const auto address = block.Address(Unit + color * 4096);
            words = {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u) & 0xffffu, static_cast<std::uint32_t>(sizeof(Values[color])), 0x31000000u};
        }
    };
    select(0);
    std::shared_ptr<const AgcDriver::DrawRecipe> recipe;
    {
        std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
        Draw(context, state, triangle, shaders, {}, &recipe);
    }
    WaitForOptimizedPipelines(context.device);
    Require(recipe != nullptr, mode + ": the first draw did not produce a recipe");
    CheckPixels(state.color, Colors[0], mode + ", first draw");
    const Counts before;
    const auto draw = [&](std::size_t color) {
        select(color);
        const auto hit = DrawWithRecipe(context, state, triangle, shaders, {}, *recipe);
        Require(hit.recorded, mode + ": a draw refused its recipe (" + DrawRecipeMissName(hit.miss) + ")");
    };
    for (const std::size_t color : {1u, 2u, 2u, 0u, 1u, 0u, 0u, 2u}) {
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            draw(color);
        }
        CheckPixels(state.color, Colors[color], mode + ", one draw per batch");
    }
    for (const std::size_t last : {1u, 0u, 2u}) {
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            for (std::size_t index = 0; index < 9; ++index) draw((last + 8 - index) % 3);
        }
        CheckPixels(state.color, Colors[last], mode + ", nine draws in a batch");
    }
    device.WaitIdle();
    const Counts after;
    const auto reused = after.reused - before.reused;
    const auto written = after.written - before.written;
    std::cout << mode << ": sets reused " << reused << ", written " << written << ", layer marks served " << after.marksServed - before.marksServed << ", full " << after.marksFull - before.marksFull << ", moved bindings served " << after.movedServed - before.movedServed << '\n';
    if (!dynamic) {
        Require(reused == 0 && written == 0 && after.marksServed == before.marksServed && after.movedServed == before.movedServed, mode + ": the switch is off and a draw took the dynamic path");
        return;
    }
    if (context.limits.maxDescriptorSetStorageBuffersDynamic == 0) return;
    Require(reused + written != 0, mode + ": no draw bound a set with dynamic offsets");
    Require(reused > written, mode + ": draws whose buffers only moved inside the same buffer wrote a set more often than they reused one");
    Require(after.marksServed != before.marksServed, mode + ": no draw of a pass left the layer marks of its target to the first one");
    if (!dataWords) Require(after.movedServed != before.movedServed, mode + ": no draw served the comparison of its unmoved buffers from the last one");
}

}

int main() {
    try {
        Block block;
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        std::uint64_t variant = 0x64796e00u;
        for (const bool dataWords : {true, false}) {
            if (!dataWords && AgcDriver::VulkanDeviceTestAccess::ContextOf(*device).hostImportAlignment == 0) continue;
            for (const int lean : {1, 0}) {
                for (const bool dynamic : {true, false}) {
                    CheckDraws(*device, block, dataWords, dynamic, lean, variant);
                    variant += 16;
                }
            }
        }
        std::cout << "draws bind one set per template with dynamic offsets and show the pixels of the rewritten sets\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
