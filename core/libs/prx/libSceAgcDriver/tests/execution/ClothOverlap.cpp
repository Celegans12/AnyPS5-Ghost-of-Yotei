#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t Programs = 4;
constexpr std::uint32_t ProgramDwords = 64;
constexpr std::uint32_t Lanes = 64;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t HalfBytes = BlockBytes / 2;
constexpr std::uint32_t OutputStride = 0x400;
constexpr std::uint32_t CodeDwords = 13;
constexpr std::uint32_t Rounds = 8;

alignas(256) std::array<std::uint32_t, Programs * ProgramDwords> Code{};
std::array<std::uint64_t, Programs> ProgramAddresses{};

enum class Mode { Off, Clean, Force, Conflict };

Mode mode = Mode::Clean;
std::uint32_t* Block = nullptr;

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("cloth overlap: " + what);
}

// Program p: out[lane] = guest[lane * 1024 + first] + guest[lane * 1024 + first + 4] (two global loads through the
// address table), stored through the shared V# over the first half of the block at outputBase + lane * 4. In the
// conflict mode program 3 reads at +0x18 and stores at 0x318: its last two lanes write 0x410 and 0x414, which lane 1
// of programs 0-2 reads, so the run's order matters and only the redo gives the serial result.
std::uint32_t OutputBase(std::uint32_t program) {
    return mode == Mode::Conflict && program == 3u ? 0x318u : 0x20u + program * OutputStride;
}

std::uint32_t ReadOffset(std::uint32_t program) {
    return mode == Mode::Conflict && program == 3u ? 0x18u : 0x10u;
}

void MakePrograms() {
    for (std::uint32_t program = 0; program < ProgramAddresses.size(); ++program) {
        auto* code = Code.data() + program * ProgramDwords;
        const std::array<std::uint32_t, CodeDwords> words{
            0x3404008au, 0x34060082u, 0x4a0606ffu, OutputBase(program),
            0xdc308000u | ReadOffset(program), 0x04000002u, 0xdc308000u | (ReadOffset(program) + 4u), 0x05000002u,
            0xbf8c3f70u, 0x4a080b04u, 0xe0701000u, 0x80010403u, 0xbf810000u};
        std::memcpy(code, words.data(), sizeof(words));
        Shader shader{};
        shader.file_header = 0x34333231;
        shader.version = 0x18;
        shader.header_size = sizeof(Shader);
        shader.shader_size = CodeDwords * 4u;
        shader.code = code;
        AgcDriverRegisterShader_nid_postfix(&shader);
    }
}

std::vector<std::uint32_t> Submission() {
    const auto base = reinterpret_cast<std::uintptr_t>(Block);
    std::vector<std::uint32_t> words{
        0xc0037600u, 0x207u, Lanes, 1u, 1u,
        0xc0017600u, 0x213u, 8u << 1u,
        0xc0087600u, 0x240u, static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(base >> 32u), 0u, 0u,
        static_cast<std::uint32_t>(base), static_cast<std::uint32_t>((base >> 32u) & 0xffffu), static_cast<std::uint32_t>(HalfBytes), 0x31016facu,
    };
    for (const auto address : ProgramAddresses) {
        words.insert(words.end(), {0xc0027600u, 0x20cu, static_cast<std::uint32_t>(address >> 8u), static_cast<std::uint32_t>(address >> 40u), 0xc0031500u, 1u, 1u, 1u, 0x8041u});
    }
    return words;
}

void Round(std::uint32_t queue, std::uint32_t round) {
    std::vector<std::uint32_t> expected(BlockBytes / 4u);
    for (std::uint32_t dword = 0; dword < BlockBytes / 4u; ++dword) expected[dword] = 0x1000u + dword * 3u + round * 7u;
    for (std::uint32_t dword = 0; dword < BlockBytes / 4u; ++dword) Block[dword] = expected[dword];
    for (std::uint32_t program = 0; program < ProgramAddresses.size(); ++program) {
        std::array<std::uint32_t, Lanes> values{};
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            const auto read = (lane * 1024u + ReadOffset(program)) / 4u;
            values[lane] = expected[read] + expected[read + 1u];
        }
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) expected[OutputBase(program) / 4u + lane] = values[lane];
    }
    auto words = Submission();
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitAcb(queue, &packet) == 0, "submission failed");
    AgcDriverWaitIdle_nid_postfix();
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(Block), HalfBytes);
    for (std::uint32_t dword = 0; dword < HalfBytes / 4u; ++dword) {
        const auto value = std::atomic_ref<std::uint32_t>(Block[dword]).load(std::memory_order_acquire);
        check(value == expected[dword], "round " + std::to_string(round) + " byte 0x" + std::to_string(dword * 4u) + " is " + std::to_string(value) + ", serial order gives " + std::to_string(expected[dword]));
    }
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

}

int main(int argc, char** argv) {
    try {
        const std::string name = argc > 1 ? argv[1] : "clean";
        mode = name == "off" ? Mode::Off : name == "force" ? Mode::Force : name == "conflict" ? Mode::Conflict : Mode::Clean;
        if (mode != Mode::Off) SetEnvironment("APS5_CLOTH_OVERLAP", "1");
        SetEnvironment("APS5_CLOTH_OVERLAP_DIAG", mode == Mode::Force ? "force,inline" : "inline");
        {
            const auto probe = OpenVulkanTestDevice();
            if (!probe) return VulkanTestSkipped;
        }
#ifdef _WIN32
        Block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        Block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
        for (std::uint32_t program = 0; program < Programs; ++program) ProgramAddresses[program] = reinterpret_cast<std::uintptr_t>(Code.data() + program * ProgramDwords);
        AgcDriver::VulkanDevice::SetClothFamily(ProgramAddresses);
        check(Block != nullptr, "cannot allocate the guest block");
        GuestAllocations::Mutation().Add(Block, HalfBytes, true, true, true);
        GuestAllocations::Mutation().Add(reinterpret_cast<std::uint8_t*>(Block) + HalfBytes, HalfBytes, true, true, true);
        MakePrograms();
        const std::uint32_t queue = 0x51u;
        for (std::uint32_t round = 0; round < Rounds; ++round) Round(queue, round);
        const auto counts = AgcDriver::VulkanDevice::ClothOverlapCounts();
        std::printf("cloth overlap (%s): %u rounds byte-identical to serial order; checked runs %llu, members %llu, conflicts %llu\n", name.c_str(), Rounds, static_cast<unsigned long long>(counts[0]), static_cast<unsigned long long>(counts[1]), static_cast<unsigned long long>(counts[2]));
        if (mode == Mode::Off) check(counts[0] == 0, "checked runs ran with APS5_CLOTH_OVERLAP unset");
        else check(counts[0] != 0 && counts[1] >= 2u * counts[0], "no checked run of two or more members was formed");
        if (mode == Mode::Clean) check(counts[2] == 0, "a disjoint run was found conflicting");
        if (mode == Mode::Force || mode == Mode::Conflict) check(counts[2] == counts[0], "a run that must be redone was not");
        LibcRunShutdown_nid_postfix();
        std::puts("cloth overlap tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
