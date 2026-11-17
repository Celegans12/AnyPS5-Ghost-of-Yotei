#include "prx/libkernel/include/Api.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t BufferBytes = Lanes * 4u;
constexpr std::size_t BlockBytes = 65536;
constexpr std::uint32_t Rounds = 24;

alignas(256) constexpr std::array<std::uint32_t, 8> Code{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a060481u, 0xe0701000u, 0x80010301u, 0xbf810000u};

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("early end of pipe: " + what);
}

void setVariable(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

std::uint32_t* NewBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    check(block != nullptr, "cannot allocate a guest block");
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true, false);
    return block;
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

}

int main() {
    setVariable("APS5_EARLY_EOP_INTERRUPT", "1");
    setVariable("APS5_NO_OPPORTUNISTIC_REAP", "1");
    try {
        {
            const auto device = OpenVulkanTestDevice();
            if (!device) return VulkanTestSkipped;
        }
        check(AgcDriver::Graphics::Recorder::EarlyEndOfPipe(), "APS5_EARLY_EOP_INTERRUPT was not seen by the recorder");
        struct ShutdownGuard {
            ~ShutdownGuard() { try { AgcDriverShutdown_nid_postfix(); } catch (...) {} }
        } shutdown;

        Shader shader{};
        shader.file_header = 0x34333231;
        shader.version = 0x18;
        shader.header_size = sizeof(Shader);
        shader.shader_size = sizeof(Code);
        shader.code = Code.data();
        AgcDriverRegisterShader_nid_postfix(&shader);

        auto* input = NewBlock();
        auto* output = NewBlock();
        auto* label = NewBlock();
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) input[lane] = 0x5b000000u | lane;

        KernelEqueue eq = 0;
        check(sceKernelCreateEqueue(&eq, "AGC early end of pipe") == 0, "event queue creation failed");
        auto owner = EqueuePin_nid_postfix(eq);
        int tag = 0;
        check(sceAgcDriverAddEqEvent(eq, 0, &tag) == 0, "graphics event registration failed");

        const auto codeAddress = reinterpret_cast<std::uintptr_t>(Code.data());
        const auto labelAddress = reinterpret_cast<std::uintptr_t>(label);
        const auto in = BufferDescriptor(input, BufferBytes);
        const auto out = BufferDescriptor(output, BufferBytes);
        std::array<std::uint32_t, Lanes> seen{};
        for (std::uint32_t round = 1; round <= Rounds; ++round) {
            std::memset(output, 0, BufferBytes);
            std::vector<std::uint32_t> words{
                0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
                0xc0037600u, 0x207u, Lanes, 1u, 1u,
                0xc0017600u, 0x213u, 8u << 1u,
                0xc0087600u, 0x240u, in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3],
                0xc0031500u, 1u, 1u, 1u, 0x8041u,
                0xc0064900u, 0x06000528u, (2u << 29u) | (2u << 24u) | (1u << 16u), static_cast<std::uint32_t>(labelAddress), static_cast<std::uint32_t>(labelAddress >> 32u), round, 0u, 0u,
            };
            Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
            check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");

            std::array<KernelEvent, 1> events{};
            int delivered = 0;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while ((delivered = owner->GetTriggeredEvents(events.data(), 1)) == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            const auto stored = static_cast<volatile const std::uint32_t*>(output);
            for (std::uint32_t lane = 0; lane < Lanes; ++lane) seen[lane] = stored[lane];
            const std::uint32_t visible = static_cast<volatile const std::uint32_t*>(label)[0];
            check(delivered == 1 && events[0].udata == &tag, "round " + std::to_string(round) + ": the end-of-pipe interrupt was not delivered");
            check(visible == round, "round " + std::to_string(round) + ": the end-of-pipe label was not written when its interrupt was delivered");
            for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
                check(seen[lane] == input[lane] + 1u, "round " + std::to_string(round) + ": the end-of-pipe interrupt was delivered before the dispatch's store to guest memory landed (lane " + std::to_string(lane) + " read " + std::to_string(seen[lane]) + ")");
            }
            AgcDriverWaitIdle_nid_postfix();
        }

        check(sceAgcDriverDeleteEqEvent(eq, 0) == 0, "graphics event deletion failed");
        owner.reset();
        check(sceKernelDeleteEqueue(eq) == 0, "event queue deletion failed");
        AgcDriverShutdown_nid_postfix();
        std::puts("early end of pipe tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
