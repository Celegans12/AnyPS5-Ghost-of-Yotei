#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::Graphics::Recorder;
using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::RuntimeSwitch;
namespace RuntimeSwitches = AgcDriver::Graphics::RuntimeSwitches;

constexpr std::byte Kept{0x40};
constexpr std::uint32_t BaseVertexLocation = 0x8c;
constexpr std::uint32_t Clipped = 0x40000000u;

alignas(256) constexpr std::array<std::uint32_t, 20> VertexCode{
    0x7e000d05u, 0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e060208u, 0x7e0802f2u, 0x7e0a02ffu, 0x44800000u, 0xf80000d1u, 0x00000005u, 0xf80008cfu, 0x04030201u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 36> SlotVertexCode{
    0x7e000d05u, 0x100c00ffu, 0x3eaaaaabu, 0x7e0c4906u, 0x100e0cffu, 0x40400000u, 0x08000f00u,
    0x080200f4u, 0x10020300u, 0x100202f4u, 0x060202f3u, 0x060400f3u, 0x10040500u, 0x060404f3u,
    0x7e120d08u, 0x101212f4u, 0x060c0d09u, 0x100c0cf0u, 0x100202ffu, 0x3e000000u, 0x06020306u, 0x060202ffu, 0xbf600000u,
    0x7e060208u, 0x7e0802f2u, 0x7e0a02ffu, 0x44800000u, 0xf80000d1u, 0x00000005u, 0xf80008cfu, 0x04030201u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u,
};
alignas(256) constexpr std::array<std::uint32_t, 8> PixelCode{0x7e000200u, 0x7e020280u, 0xf800180fu, 0x00010100u, 0xbf810000u, 0xbf810000u, 0xbf810000u, 0xbf810000u};
constexpr std::size_t TargetBytes = 65536;
constexpr std::uint32_t RenderTargetTileMode = 0x1b;
using Target = std::array<std::byte, TargetBytes>;

Target& NewTarget() {
#ifdef _WIN32
    auto* block = static_cast<std::byte*>(GuestArena::GuestArenaAllocate_nid_postfix(TargetBytes, 65536));
    if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, TargetBytes, PAGE_READWRITE, TargetBytes);
#else
    constexpr std::uintptr_t alignment = 65536;
    std::byte* block = nullptr;
    void* mapped = mmap(nullptr, TargetBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped != MAP_FAILED) {
        const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
        const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
        if (aligned != begin) munmap(mapped, aligned - begin);
        if (aligned + TargetBytes != begin + TargetBytes + alignment) munmap(reinterpret_cast<void*>(aligned + TargetBytes), begin + alignment - aligned);
        block = reinterpret_cast<std::byte*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, TargetBytes);
    }
#endif
    Require(block != nullptr, "indirect draw hoist: cannot allocate a target block");
    std::fill(block, block + TargetBytes, std::byte{0});
    GuestAllocations::Mutation().Add(block, TargetBytes, true, true, true);
    return *reinterpret_cast<Target*>(block);
}

constexpr std::array<std::uint32_t, 16> RecordWords{3, 1, 0, 0, 6, 1, 6, 0, 3, 2, 9, 0, 3, 1, 0, 0};
constexpr std::size_t RecordBytes = 65536;
std::uint32_t* Records = nullptr;

struct Header {
    Shader shader{};
    ShaderUserData userData{};
};

alignas(8) Header VertexHeader{};
alignas(8) Header SlotVertexHeader{};
alignas(8) Header PixelHeader{};

void Registered(Header& header, const std::uint32_t* code, std::size_t bytes, std::uint8_t type) {
    header.shader.file_header = 0x34333231u;
    header.shader.version = 0x18u;
    header.shader.header_size = sizeof(Header);
    header.shader.shader_size = static_cast<std::uint32_t>(bytes);
    header.shader.code = code;
    header.shader.type = type;
    header.shader.user_data = &header.userData;
    AgcDriverRegisterShader_nid_postfix(&header.shader);
}

void Append(std::vector<std::uint32_t>& commands, std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) {
    commands.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
    commands.insert(commands.end(), payload);
}

void AppendState(std::vector<std::uint32_t>& commands, std::uint32_t primitive, const std::uint32_t* vertexCode) {
    const auto vertex = reinterpret_cast<std::uintptr_t>(vertexCode);
    const auto pixel = reinterpret_cast<std::uintptr_t>(PixelCode.data());
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> context{
        {0x2d5, 0x2000},
        {0x1b6, 0}, {0x207, 0x01200000}, {0x200, 0}, {0x203, 0x800},
        {0x2dc, 0xaa00}, {0x2f8, 0}, {0x292, 2}, {0x293, 0},
        {0x80, 0}, {0x8d, 0}, {0x83, 0xffff}, {0x8c, 0xa},
        {0x2f9, 0x2d}, {0x313, 0x6000}, {0x30e, 0xffffffff}, {0x30f, 0xffffffff},
        {0x206, 0x43f}, {0x204, 0x80000}, {0x205, 0x240},
        {0x8e, 0xf}, {0x8f, 0xf}, {0x202, 0xcc0010},
        {0x1c4, 0}, {0x1c5, 9}, {0x1c3, 4}, {0x31c, 0x28028},
        {0x31b, 0}, {0x31d, 0}, {0x3b0, (63u << 14u) | 3u},
        {0x3b8, 0x9000000u | (RenderTargetTileMode << 14u)}, {0x1e0, 0},
        {0xc, 0}, {0xd, 0x40040},
        {0x81, 0x80000000}, {0x82, 0x40040},
        {0x90, 0x80000000}, {0x91, 0x40040},
        {0x94, 0x80000000}, {0x95, 0x40040},
        {0x10f, std::bit_cast<std::uint32_t>(32.0f)}, {0x110, std::bit_cast<std::uint32_t>(32.0f)},
        {0x111, std::bit_cast<std::uint32_t>(-2.0f)}, {0x112, std::bit_cast<std::uint32_t>(2.0f)},
        {0x113, std::bit_cast<std::uint32_t>(1.0f)}, {0x114, 0},
        {0xb4, 0}, {0xb5, std::bit_cast<std::uint32_t>(1.0f)},
        {0x2fe, 0}, {0x302, 0}, {0x306, 0}, {0x30a, 0},
    };
    const std::vector<std::pair<std::uint32_t, std::uint32_t>> shader{
        {0xc8, static_cast<std::uint32_t>(vertex >> 8u)}, {0xc9, static_cast<std::uint32_t>(vertex >> 40u)}, {0x8b, 1u << 1u},
        {0x008, static_cast<std::uint32_t>(pixel >> 8u)}, {0x009, static_cast<std::uint32_t>(pixel >> 40u)}, {0x00b, 1u << 1u}, {0x00c, std::bit_cast<std::uint32_t>(1.0f)},
    };
    Append(commands, 0x79, {0x242, primitive});
    Append(commands, 0x79, {0x24a, 0});
    for (const auto& [offset, value] : context) Append(commands, 0x69, {offset, value});
    for (const auto& [offset, value] : shader) Append(commands, 0x76, {offset, value});
    const auto records = reinterpret_cast<std::uintptr_t>(Records);
    Append(commands, 0x11, {1, static_cast<std::uint32_t>(records), static_cast<std::uint32_t>(records >> 32u)});
}

void AppendTarget(std::vector<std::uint32_t>& commands, Target& memory) {
    const auto color = reinterpret_cast<std::uintptr_t>(memory.data());
    Append(commands, 0x69, {0x318, static_cast<std::uint32_t>(color >> 8u)});
    Append(commands, 0x69, {0x390, static_cast<std::uint32_t>(color >> 40u)});
}

void Submit(std::vector<std::uint32_t>& commands, const Target& target) {
    Packet packet{commands.data(), static_cast<std::uint32_t>(commands.size()), 0, {}};
    Require(sceAgcDriverSubmitDcb(&packet) == 0, "the draw submission failed");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::Graphics::StorageTexture::FlushPending(reinterpret_cast<std::uintptr_t>(target.data()), target.size(), nullptr, "test read-back");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(target.data()), target.size());
}

std::size_t Covered(const Target& memory) {
    return static_cast<std::size_t>(std::count_if(memory.begin(), memory.end(), [](std::byte value) { return value != Kept; }));
}

void Same(const Target& actual, const Target& expected, const std::string& what) {
    for (std::size_t index = 0; index < expected.size(); ++index) {
        Require(actual[index] == expected[index], what + ": byte " + std::to_string(index) + " is " + std::to_string(static_cast<unsigned>(actual[index])) + ", expected " + std::to_string(static_cast<unsigned>(expected[index])));
    }
}

struct Counts {
    Recorder::IndirectHoistStats hoist;
    AgcDriver::Graphics::DrawPassCounts passes;
};

Counts Now() {
    return {Recorder::IndirectHoistCounts(), AgcDriver::Graphics::DrawPassTotals()[0]};
}

std::string Describe(const Counts& before, const Counts& after) {
    return "joined " + std::to_string(after.hoist.joinedInPlace - before.hoist.joinedInPlace) + " in place " + std::to_string(after.hoist.joinedRewrite - before.hoist.joinedRewrite) + " rewritten, forced " + std::to_string(after.hoist.forcedNoChain - before.hoist.forcedNoChain) + " no chain " + std::to_string(after.hoist.forcedNoPrelude - before.hoist.forcedNoPrelude) + " no prelude, splits " + std::to_string(after.hoist.splits - before.hoist.splits) + " (" + std::to_string(after.hoist.splitsRefused - before.hoist.splitsRefused) + " refused), preludes " + std::to_string(after.hoist.preludes - before.hoist.preludes) + ", passes begun " + std::to_string(after.passes.passesBegun - before.passes.passesBegun) + " continued " + std::to_string(after.passes.passesContinued - before.passes.passesContinued);
}

Counts Delta(const Counts& before, const Counts& after) {
    const auto& b = before.hoist;
    const auto& a = after.hoist;
    return {{a.joinedInPlace - b.joinedInPlace, a.joinedRewrite - b.joinedRewrite, a.forcedNoChain - b.forcedNoChain, a.forcedNoPrelude - b.forcedNoPrelude, a.splits - b.splits, a.splitsRefused - b.splitsRefused, a.preludes - b.preludes, a.rewritesHoisted - b.rewritesHoisted}, {after.passes.draws - before.passes.draws, after.passes.passesBegun - before.passes.passesBegun, after.passes.passesContinued - before.passes.passesContinued, 0, 0}};
}

const char* Arm() {
    return RuntimeSwitches::Set(RuntimeSwitch::NoIndirectHoist) ? "APS5_NO_INDIRECT_HOIST" : "hoisted";
}

Counts DrawRewritten(Target& target, bool indirect) {
    target.fill(Kept);
    std::vector<std::uint32_t> commands;
    AppendState(commands, 17, SlotVertexCode.data());
    AppendTarget(commands, target);
    constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 4> shapes{{{3, 1}, {6, 1}, {3, 2}, {3, 1}}};
    for (std::size_t draw = 0; draw < shapes.size(); ++draw) {
        if (indirect) {
            Append(commands, 0x24, {static_cast<std::uint32_t>(draw * 16u), 0, 0, 2});
        } else {
            Append(commands, 0x2f, {shapes[draw].second});
            Append(commands, 0x2d, {shapes[draw].first, 2});
        }
    }
    const auto before = Now();
    Submit(commands, target);
    const auto delta = Delta(before, Now());
    std::printf("rewritten records (%s): covered %zu; %s\n", indirect ? Arm() : "direct reference", Covered(target), Describe({}, delta).c_str());
    return delta;
}

Counts DrawInPlace(Target& target, bool indirect) {
    target.fill(Kept);
    std::vector<std::uint32_t> commands;
    AppendState(commands, 4, VertexCode.data());
    AppendTarget(commands, target);
    Append(commands, 0x76, {BaseVertexLocation, 0});
    Append(commands, 0x2d, {3, 2});
    if (indirect) {
        Append(commands, 0x76, {BaseVertexLocation, Clipped});
        Append(commands, 0x24, {48, BaseVertexLocation, 0, 2});
        Append(commands, 0x24, {48, BaseVertexLocation, 0, 2});
    }
    const auto before = Now();
    Submit(commands, target);
    const auto delta = Delta(before, Now());
    std::printf("in-place records (%s): covered %zu; %s\n", indirect ? Arm() : "direct reference", Covered(target), Describe({}, delta).c_str());
    return delta;
}

void RewrittenRecordsJoinTheOpenPass(Target& direct, Target& old, Target& hoisted) {
    const auto reference = DrawRewritten(direct, false);
    Require(reference.passes.draws == 4, "the direct reference draws were not recorded into resident targets (" + std::to_string(reference.passes.draws) + " recorded of 4)");
    Require(Covered(direct) != 0, "the direct rect-list draws covered nothing");
    RuntimeSwitches::Assign(RuntimeSwitch::NoIndirectHoist, true);
    const auto before = DrawRewritten(old, true);
    RuntimeSwitches::Assign(RuntimeSwitch::NoIndirectHoist, false);
    const auto first = DrawRewritten(hoisted, true);
    Same(old, direct, "the rewritten indirect draws with APS5_NO_INDIRECT_HOIST");
    Same(hoisted, direct, "the hoisted rewritten indirect draws");
    Require(before.hoist.joinedRewrite == 0 && before.passes.passesBegun == 4, "with APS5_NO_INDIRECT_HOIST every rewritten indirect draw must begin its own pass (" + std::to_string(before.passes.passesBegun) + " passes, " + std::to_string(before.hoist.joinedRewrite) + " joined)");
    Require(first.hoist.joinedRewrite == 2 && first.hoist.preludes == 1 && first.hoist.rewritesHoisted == 2 && first.passes.passesBegun == 2, "the hoisted rewrites: " + Describe({}, first) + "; expected 2 joined, 1 prelude with 2 rewrites, 2 passes");
    const auto again = DrawRewritten(hoisted, true);
    Same(hoisted, direct, "the hoisted rewritten indirect draws into a predicted pass");
    Require(again.hoist.joinedRewrite == 3 && again.passes.passesBegun == 1, "a pass predicted to take rewritten indirect draws: " + Describe({}, again) + "; expected 3 joined, 1 pass");
}

void InPlaceRecordsJoinTheOpenPass(Target& direct, Target& old, Target& hoisted) {
    DrawInPlace(direct, false);
    Require(Covered(direct) != 0, "the direct draw covered nothing");
    RuntimeSwitches::Assign(RuntimeSwitch::NoIndirectHoist, true);
    const auto gpuBefore = AgcDriver::Graphics::IndirectDrawCounts()[0];
    const auto before = DrawInPlace(old, true);
    RuntimeSwitches::Assign(RuntimeSwitch::NoIndirectHoist, false);
    const auto first = DrawInPlace(hoisted, true);
    const auto gpuSide = AgcDriver::Graphics::IndirectDrawCounts()[0] - gpuBefore;
    Same(old, direct, "the in-place indirect draws with APS5_NO_INDIRECT_HOIST");
    Same(hoisted, direct, "the hoisted in-place indirect draws");
    if (gpuSide == 0) {
        std::puts("in-place records: this device read the records on the CPU (no GPU-side draw), so the pass counts are not checked");
        return;
    }
    Require(gpuSide == 4, "in-place records: " + std::to_string(gpuSide) + " of 4 indirect draws were GPU-side");
    Require(before.hoist.joinedInPlace == 0 && before.passes.passesBegun == 3, "with APS5_NO_INDIRECT_HOIST every in-place indirect draw must begin its own pass: " + Describe({}, before));
    Require(first.hoist.joinedInPlace == 2 && first.passes.passesBegun == 1, "the hoisted in-place draws: " + Describe({}, first) + "; expected 2 joined, 1 pass");
}

}

int main() {
    try {
#ifdef _WIN32
        _putenv_s("APS5_PROFILE_DRAW", "1");
#else
        setenv("APS5_PROFILE_DRAW", "1", 1);
#endif
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        Registered(VertexHeader, VertexCode.data(), sizeof(VertexCode), 2);
        Registered(SlotVertexHeader, SlotVertexCode.data(), sizeof(SlotVertexCode), 2);
        Registered(PixelHeader, PixelCode.data(), sizeof(PixelCode), 1);
        Records = static_cast<std::uint32_t*>(::operator new(RecordBytes, std::align_val_t{RecordBytes}));
        std::fill(Records, Records + RecordBytes / sizeof(std::uint32_t), 0u);
        std::copy(RecordWords.begin(), RecordWords.end(), Records);
        {
            GuestAllocations::Mutation mutation;
            mutation.Add(Records, RecordBytes, true, true, true);
        }
        RewrittenRecordsJoinTheOpenPass(NewTarget(), NewTarget(), NewTarget());
        InPlaceRecordsJoinTheOpenPass(NewTarget(), NewTarget(), NewTarget());
        AgcDriverShutdown_nid_postfix();
        std::puts("indirect draws joined the open pass and drew what the old path did");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        AgcDriverShutdown_nid_postfix();
        return 1;
    }
}
