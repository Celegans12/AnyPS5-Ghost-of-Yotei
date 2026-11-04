#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t Stride = 16;
alignas(256) std::array<std::uint32_t, Lanes * Stride> Output{};

constexpr std::array<std::uint32_t, 1> LinearLane{0x7e060300};
constexpr std::array<std::uint32_t, 2> PlaneLane{0x34060283, 0x4a060700};
constexpr std::array<std::uint32_t, 4> CubeLane{0x34080282, 0x4a060900, 0x34080484, 0x4a060903};

constexpr std::array<std::uint32_t, 71> BallotBody{
    0x360a0681, 0x7d8a0a80, 0x7e14026a, 0x7e16026b, 0xe0702000, 0x80010a03, 0xe0702004, 0x80010b03,
    0xbefe0400, 0x360a0682, 0x7d8a0a80, 0xbefe04c1, 0x7e14026a, 0x7e16026b, 0xe0702008, 0x80010a03,
    0xe070200c, 0x80010b03,
    0xbefe0402, 0x7d8806a8, 0xbefe04c1, 0x7e14026a, 0x7e16026b, 0xe0702010, 0x80010a03, 0xe0702014,
    0x80010b03,
    0xbe8a0480, 0xbe8c0480, 0xbefe0400, 0x360a0684, 0x7d8a0a80, 0xbe88246a, 0xbf880004, 0x7d8806b0,
    0xbe8a046a, 0x7d82068c, 0xbe8c046a, 0xbefe04c1, 0x7e14020a, 0x7e16020b, 0xe0702018, 0x80010a03,
    0xe070201c, 0x80010b03, 0x7e14020c, 0x7e16020d, 0xe0702020, 0x80010a03, 0xe0702024, 0x80010b03,
    0x360a0681, 0x7d880694, 0xbe8a046a, 0xbefe0402, 0x7d8206ac, 0xbe8c046a, 0xbefe0400, 0x7d8a0a80,
    0xbe8e046a, 0xbefe04c1,
    0x7e14020a, 0x7e16020b, 0xe0702028, 0x80010a03, 0xe070202c, 0x80010b03, 0x7e14020c, 0x7e16020d,
    0xe0702030, 0x80010a03,
};

constexpr std::array<std::uint32_t, 9> BallotTail{
    0xe0702034, 0x80010b03, 0x7e14020e, 0x7e16020f, 0xe0702038, 0x80010a03, 0xe070203c, 0x80010b03,
    0xbf810000,
};

struct Shape {
    const char* name;
    std::array<std::uint32_t, 3> threads;
    std::uint32_t components;
    std::span<const std::uint32_t> prefix;
};

struct Masks {
    std::uint64_t first;
    std::uint64_t second;
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t records) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | ((Stride * 4u) << 16u), records, 0x11016facu};
}

bool NamesWaveBallot(std::span<const std::uint32_t> words) {
    constexpr std::string_view name = "wave_ballot";
    for (std::size_t index = 5; index < words.size();) {
        const auto count = words[index] >> 16u;
        if (count == 0u) break;
        if ((words[index] & 0xffffu) == 5u && count > 2u) {
            const auto* text = reinterpret_cast<const char*>(words.data() + index + 2u);
            const auto bytes = (count - 2u) * 4u;
            if (std::string_view(text, strnlen(text, bytes)) == name) return true;
        }
        index += count;
    }
    return false;
}

void Run(AgcDriver::VulkanDevice& device, const Shape& shape, std::span<const std::uint32_t> code, Masks masks, bool spanned, bool oneBarrier) {
    Output.fill(0xdeadbeefu);
    std::vector<std::uint32_t> userData(8, 0u);
    userData[0] = static_cast<std::uint32_t>(masks.first);
    userData[1] = static_cast<std::uint32_t>(masks.first >> 32u);
    userData[2] = static_cast<std::uint32_t>(masks.second);
    userData[3] = static_cast<std::uint32_t>(masks.second >> 32u);
    const auto output = BufferDescriptor(Output.data(), Lanes);
    std::copy(output.begin(), output.end(), userData.begin() + 4);
    const auto address = reinterpret_cast<std::uintptr_t>(code.data());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{address, std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{shape.threads, 0, {false, false, false}, false, shape.components};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, address, code, 0, {}},
        {Lanes, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.target.spannedWaves = spanned;
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    Require(result.fullSubgroups == spanned, std::string(shape.name) + (spanned ? ": the program did not span two host subgroups" : ": the program spans host subgroups without APS5_SPANNED_WAVE"));
    Require(NamesWaveBallot(result.spirv.Words()) == (spanned && oneBarrier), std::string(shape.name) + ": the one-barrier spanned ballot was " + (spanned && oneBarrier ? "not emitted" : "emitted without APS5_SPANNED_BALLOT_ONE_BARRIER"));
    device.Dispatch(result, 1, 1, 1, {}, address);
    device.WaitIdle();
}

std::uint64_t Lanewise(bool (*predicate)(std::uint32_t)) {
    std::uint64_t mask = 0;
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        if (predicate(lane)) mask |= std::uint64_t{1} << lane;
    }
    return mask;
}

std::array<std::uint64_t, 8> Expected(Masks masks) {
    const auto odd = Lanewise([](std::uint32_t lane) { return (lane & 1u) != 0; });
    const auto bit1 = Lanewise([](std::uint32_t lane) { return (lane & 2u) != 0; });
    const auto bit2 = Lanewise([](std::uint32_t lane) { return (lane & 4u) != 0; });
    const auto below40 = Lanewise([](std::uint32_t lane) { return lane < 40u; });
    const auto below48 = Lanewise([](std::uint32_t lane) { return lane < 48u; });
    const auto above12 = Lanewise([](std::uint32_t lane) { return lane > 12u; });
    const auto below20 = Lanewise([](std::uint32_t lane) { return lane < 20u; });
    const auto above44 = Lanewise([](std::uint32_t lane) { return lane > 44u; });
    const auto branch = masks.first & bit2;
    return {odd, masks.first & bit1, masks.second & below40, branch & below48, branch & above12, below20, masks.second & above44, masks.first & odd};
}

void Check(const Shape& shape, Masks masks, bool spanned) {
    const auto expected = Expected(masks);
    const std::array<const char*, 8> names{"full exec", "first mask", "second mask", "branch below 48", "branch above 12", "back-to-back below 20", "back-to-back above 44", "back-to-back odd"};
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        for (std::uint32_t ballot = 0; ballot < expected.size(); ++ballot) {
            for (std::uint32_t half = 0; half < 2u; ++half) {
                const auto actual = Output[lane * Stride + ballot * 2u + half];
                const auto wanted = static_cast<std::uint32_t>(expected[ballot] >> (32u * half));
                char masksText[64];
                std::snprintf(masksText, sizeof(masksText), "%016llx/%016llx", static_cast<unsigned long long>(masks.first), static_cast<unsigned long long>(masks.second));
                Require(actual == wanted, std::string(spanned ? "spanned " : "two-lane ") + shape.name + " masks " + masksText + ": lane " + std::to_string(lane) + " " + names[ballot] + (half == 0u ? " low" : " high") + " word is " + std::to_string(actual) + ", expected " + std::to_string(wanted));
            }
        }
    }
}

std::vector<std::uint32_t> Program(std::span<const std::uint32_t> prefix) {
    std::vector<std::uint32_t> code(prefix.begin(), prefix.end());
    code.insert(code.end(), BallotBody.begin(), BallotBody.end());
    code.insert(code.end(), BallotTail.begin(), BallotTail.end());
    return code;
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        if (device->Target().subgroupSize < 32) {
            std::printf("skipped, subgroup size %u cannot hold a wave32\n", device->Target().subgroupSize);
            return VulkanTestSkipped;
        }
        const auto* switchValue = std::getenv("APS5_SPANNED_BALLOT_ONE_BARRIER");
        const bool oneBarrier = switchValue != nullptr && std::string_view(switchValue) != "0";
        const bool spannable = device->Target().subgroupSize == 32u && device->ComputeFullSubgroups();
        if (!spannable) std::puts("spanned mode skipped: the device has no full 32-lane compute subgroups");
        const std::array<Shape, 3> shapes{{
            {"64x1x1", {64, 1, 1}, 1, LinearLane},
            {"8x8x1", {8, 8, 1}, 2, PlaneLane},
            {"4x4x4", {4, 4, 4}, 3, CubeLane},
        }};
        const std::array<Masks, 6> cases{{
            {~0ull, ~0ull},
            {0x00000000ffffffffull, 0xffffffff00000000ull},
            {0x5555aaaa0f0f00ffull, 0x8000000100010000ull},
            {0ull, 0x00000001ffffffffull},
            {0xfffffffe7fffffffull, 0x0000000080000000ull},
            {0xf0f0f0f000000000ull, 0ull},
        }};
        for (const auto& shape : shapes) {
            alignas(256) static std::array<std::uint32_t, 128> storage{};
            const auto program = Program(shape.prefix);
            Require(program.size() <= storage.size(), "the test program does not fit its storage");
            storage.fill(0xbf810000u);
            std::copy(program.begin(), program.end(), storage.begin());
            const std::span<const std::uint32_t> code{storage.data(), program.size()};
            for (const bool spanned : {false, true}) {
                if (spanned && !spannable) continue;
                for (const auto masks : cases) {
                    for (std::uint32_t repeat = 0; repeat < 2; ++repeat) {
                        Run(*device, shape, code, masks, spanned, oneBarrier);
                        Check(shape, masks, spanned);
                    }
                }
            }
        }
        std::printf("spanned ballot exec mask tests passed (two-lane%s, %s ballot)\n", spannable ? " and spanned" : "", oneBarrier ? "one-barrier" : "three-barrier");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
