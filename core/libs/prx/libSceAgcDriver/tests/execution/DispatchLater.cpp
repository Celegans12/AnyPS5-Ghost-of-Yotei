#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchLater.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "TargetAddressWindow.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DispatchLaterCounted;
using AgcDriver::DriverDetail::DispatchLaterCounts;
using AgcDriver::DriverDetail::DrawPipeline;

constexpr std::uint32_t Lanes = 32;
constexpr std::uint32_t BufferBytes = Lanes * 4u;
constexpr std::size_t BlockBytes = 65536;

std::uint32_t* NewBlock() {
#ifdef _WIN32
    auto* block = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* block = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, BlockBytes));
#endif
    if (block == nullptr) throw std::runtime_error("dispatch later: cannot allocate a guest block");
    std::memset(block, 0, BlockBytes);
    GuestAllocations::Mutation().Add(block, BlockBytes, true, true, false);
    return block;
}

std::uint32_t* Input = nullptr;
std::uint32_t* Middle = nullptr;
std::uint32_t* Output = nullptr;
std::uint32_t* Copied = nullptr;
std::uint32_t* Table = nullptr;
std::uint32_t* TargetA = nullptr;
std::uint32_t* TargetB = nullptr;

void NewTargetPair() {
    constexpr std::size_t regionBytes = 3u * BlockBytes;
#ifdef _WIN32
    auto* region = static_cast<std::uint32_t*>(VirtualAlloc(nullptr, regionBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
    auto* region = static_cast<std::uint32_t*>(std::aligned_alloc(BlockBytes, regionBytes));
#endif
    if (region == nullptr) throw std::runtime_error("dispatch later: cannot allocate contiguous target blocks");
    std::memset(region, 0, regionBytes);
    GuestAllocations::Mutation().Add(region, regionBytes, true, true, false);
    const auto indices = AgcDriver::Tests::TargetAddressWindowPair(reinterpret_cast<std::uintptr_t>(region), BlockBytes);
    constexpr auto wordsPerBlock = BlockBytes / sizeof(std::uint32_t);
    TargetA = region + indices[0] * wordsPerBlock;
    TargetB = region + indices[1] * wordsPerBlock;
}

alignas(256) constexpr std::array<std::uint32_t, 8> Code{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a060481u, 0xe0701000u, 0x80010301u, 0xbf810000u};

alignas(256) constexpr std::array<std::uint32_t, 11> TableCode{0xf4080200u, 0xfa000000u, 0xbf8cc07fu, 0x34020082u, 0xe0301000u, 0x80020201u, 0xbf8c3f70u, 0x4a060481u, 0xe0701000u, 0x80010301u, 0xbf810000u};

void check(bool condition, const std::string& what) {
    if (!condition) throw std::runtime_error("dispatch later: " + what);
}

template <typename Ready>
bool WaitFor(Ready ready, std::chrono::milliseconds limit) {
    const auto deadline = std::chrono::steady_clock::now() + limit;
    while (!ready()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

std::array<std::uint32_t, 7> FillPacket(std::uint32_t value, const void* destination, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(destination);
    return {0xc0055000u, 2u << 29u, value, 0u, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), bytes};
}

std::array<std::uint32_t, 7> CopyPacket(const void* source, const void* destination, std::uint32_t bytes) {
    const auto from = reinterpret_cast<std::uintptr_t>(source);
    const auto to = reinterpret_cast<std::uintptr_t>(destination);
    return {0xc0055000u, 0u, static_cast<std::uint32_t>(from), static_cast<std::uint32_t>(from >> 32u), static_cast<std::uint32_t>(to), static_cast<std::uint32_t>(to >> 32u), bytes};
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

void Append(std::vector<std::uint32_t>& words, std::span<const std::uint32_t> more) {
    words.insert(words.end(), more.begin(), more.end());
}

std::vector<std::uint32_t> BufferDispatch(const void* from, const void* to) {
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(Code.data());
    const auto input = BufferDescriptor(from, BufferBytes);
    const auto output = BufferDescriptor(to, BufferBytes);
    return {
        0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
        0xc0037600u, 0x207u, Lanes, 1u, 1u,
        0xc0017600u, 0x213u, 8u << 1u,
        0xc0087600u, 0x240u, input[0], input[1], input[2], input[3], output[0], output[1], output[2], output[3],
        0xc0031500u, 1u, 1u, 1u, 0x8041u,
    };
}

std::vector<std::uint32_t> TableDispatch(std::uint32_t unusedWord) {
    const auto codeAddress = reinterpret_cast<std::uintptr_t>(TableCode.data());
    const auto tableAddress = reinterpret_cast<std::uintptr_t>(Table);
    const auto output = BufferDescriptor(Output, BufferBytes);
    return {
        0xc0027600u, 0x20cu, static_cast<std::uint32_t>(codeAddress >> 8u), static_cast<std::uint32_t>(codeAddress >> 40u),
        0xc0037600u, 0x207u, Lanes, 1u, 1u,
        0xc0017600u, 0x213u, 8u << 1u,
        0xc0087600u, 0x240u, static_cast<std::uint32_t>(tableAddress), static_cast<std::uint32_t>(tableAddress >> 32u), unusedWord, 0u, output[0], output[1], output[2], output[3],
        0xc0031500u, 1u, 1u, 1u, 0x8041u,
    };
}

void Submit(std::vector<std::uint32_t>& words) {
    Packet packet{words.data(), static_cast<std::uint32_t>(words.size()), 0, {}};
    check(sceAgcDriverSubmitDcb(&packet) == 0, "submission failed");
}

void Flush(const void* data) {
    AgcDriver::GuestMemory::FlushGpuWrites(reinterpret_cast<std::uintptr_t>(data), BufferBytes);
}

Shader ComputeShader(const void* code, std::uint32_t bytes) {
    Shader shader{};
    shader.file_header = 0x34333231;
    shader.version = 0x18;
    shader.header_size = sizeof(Shader);
    shader.shader_size = bytes;
    shader.code = code;
    return shader;
}

void ChainedDispatchesSeeEachOther() {
    const std::array<std::uint32_t, 4> values{0x0a0b0c0du, 0x11112222u, 0x33334444u, 0x55556666u};
    for (std::size_t round = 0; round < values.size(); ++round) {
        const auto value = values[round];
        const auto before = DispatchLaterCounted();
        std::vector<std::uint32_t> words;
        Append(words, FillPacket(value, Input, BufferBytes));
        Append(words, BufferDispatch(Input, Middle));
        Append(words, BufferDispatch(Middle, Output));
        Append(words, CopyPacket(Output, Copied, BufferBytes));
        Submit(words);
        AgcDriverWaitIdle_nid_postfix();
        Flush(Middle);
        Flush(Output);
        Flush(Copied);
        const auto after = DispatchLaterCounted();
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            check(Middle[lane] == value + 1u, "chain round " + std::to_string(round) + ": lane " + std::to_string(lane) + " of the first dispatch's output is " + std::to_string(Middle[lane]) + ", expected " + std::to_string(value + 1u) + " (it ran before the fill queued ahead of it)");
            check(Output[lane] == value + 2u, "chain round " + std::to_string(round) + ": lane " + std::to_string(lane) + " of the second dispatch's output is " + std::to_string(Output[lane]) + ", expected " + std::to_string(value + 2u) + " (it ran before the dispatch queued ahead of it)");
            check(Copied[lane] == value + 2u, "chain round " + std::to_string(round) + ": lane " + std::to_string(lane) + " of the copy after the dispatches is " + std::to_string(Copied[lane]) + ", expected " + std::to_string(value + 2u) + " (the copy read the buffer before the dispatch that writes it)");
        }
        check(after.handed - before.handed == 2u, "chain round " + std::to_string(round) + ": " + std::to_string(after.handed - before.handed) + " of the 2 direct dispatches were handed to the draw finisher");
        std::printf("dispatch later: chain round %zu, %llu handed, %llu queued as commit items, %llu dispatched on the finisher\n", round, static_cast<unsigned long long>(after.handed - before.handed), static_cast<unsigned long long>(after.queued - before.queued), static_cast<unsigned long long>(after.inlined - before.inlined));
    }
}

void TableUnderAQueuedWrite(bool verifying) {
    const std::array<std::uint32_t, 4> tableValues{0x6a6a0001u, 0x6b6b0002u, 0x6c6c0003u, 0x6d6d0004u};
    const auto initial = BufferDescriptor(TargetA, BufferBytes);
    std::copy(initial.begin(), initial.end(), Table);
    for (std::size_t round = 0; round < tableValues.size(); ++round) {
        auto* target = round % 2 == 0 ? TargetB : TargetA;
        const auto value = tableValues[round];
        check(WaitFor([] { return DrawPipeline::QueuedItems() == 0; }, std::chrono::seconds(10)), "the pipeline did not empty before table round " + std::to_string(round));
        const auto released = std::make_shared<std::atomic<bool>>(false);
        struct Release {
            std::atomic<bool>& flag;
            ~Release() { flag.store(true, std::memory_order_release); }
        } release{*released};
        DrawPipeline::Queue0().Enqueue([released] {
            while (!released->load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }, {});
        const auto before = DispatchLaterCounted();
        std::vector<std::uint32_t> words;
        Append(words, FillPacket(value, target, BufferBytes));
        const auto descriptor = BufferDescriptor(target, BufferBytes);
        Append(words, FillPacket(descriptor[0], Table, 4u));
        Append(words, TableDispatch(round == 0 ? 0u : 1u));
        Submit(words);
        const bool handedWhileHeld = WaitFor([&] { return DispatchLaterCounted().handed > before.handed; }, std::chrono::seconds(verifying ? 1 : 10));
        released->store(true, std::memory_order_release);
        AgcDriverWaitIdle_nid_postfix();
        Flush(Output);
        const auto after = DispatchLaterCounted();
        for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
            check(Output[lane] == value + 1u, "table round " + std::to_string(round) + ": lane " + std::to_string(lane) + " of the output is " + std::to_string(Output[lane]) + ", expected " + std::to_string(value + 1u) + " (the draw finisher read the descriptor table before the queued write of the table)");
        }
        check(after.handed - before.handed == 1u, "table round " + std::to_string(round) + ": the dispatch whose table has a queued write was not handed to the draw finisher");
        if (!verifying) check(handedWhileHeld, "table round " + std::to_string(round) + ": the worker waited for the held committer before handing the dispatch to the draw finisher");
        std::printf("dispatch later: table round %zu, handed while the committer was held: %s, %llu queued as commit items, %llu dispatched on the finisher\n", round, handedWhileHeld ? "yes" : "no", static_cast<unsigned long long>(after.queued - before.queued), static_cast<unsigned long long>(after.inlined - before.inlined));
    }
}

int Run(bool verifying) {
    {
        const auto probe = OpenVulkanTestDevice();
        if (!probe) return VulkanTestSkipped;
    }
    check(AgcDriver::DriverDetail::DispatchLater(), "APS5_DISPATCH_LATER is not seen by the driver");
    check(AgcDriver::DriverDetail::VerifyDispatchLater() == verifying, "APS5_VERIFY_DISPATCH_LATER is not what the test set");
    check(DrawPipeline::Depth() != 0, "the draw pipeline has no depth");
    Shader shader = ComputeShader(Code.data(), sizeof(Code));
    AgcDriverRegisterShader_nid_postfix(&shader);
    Shader tableShader = ComputeShader(TableCode.data(), sizeof(TableCode));
    AgcDriverRegisterShader_nid_postfix(&tableShader);
    Input = NewBlock();
    Middle = NewBlock();
    Output = NewBlock();
    Copied = NewBlock();
    Table = NewBlock();
    NewTargetPair();
    check((reinterpret_cast<std::uintptr_t>(TargetA) >> 32u) == (reinterpret_cast<std::uintptr_t>(TargetB) >> 32u), "the two targets do not share their high address dword");
    ChainedDispatchesSeeEachOther();
    TableUnderAQueuedWrite(verifying);
    const auto counted = DispatchLaterCounted();
    check(counted.queued != 0, "no dispatch resolved on the draw finisher was queued as a commit item");
    if (!verifying) check(counted.inlined != 0, "no dispatch resolved on the draw finisher was dispatched there after a drain of the committer");
    check(counted.queued + counted.inlined >= counted.handed, "the draw finisher finished " + std::to_string(counted.queued + counted.inlined) + " of the " + std::to_string(counted.handed) + " dispatches handed to it");
    if (verifying) {
        check(counted.verified == counted.handed, "the worker repeated " + std::to_string(counted.verified) + " captures for " + std::to_string(counted.handed) + " dispatches handed to the draw finisher");
        check(counted.unverified == 0, std::to_string(counted.unverified) + " captures could not be repeated on the worker");
        check(counted.differing == 0, std::to_string(counted.differing) + " captures differ between the worker and the draw finisher");
    }
    std::printf("dispatch later%s: %llu handed, %llu kept, %llu queued, %llu dispatched on the finisher, %llu verified, %llu differing\n", verifying ? " (verified)" : "", static_cast<unsigned long long>(counted.handed), static_cast<unsigned long long>(counted.kept), static_cast<unsigned long long>(counted.queued), static_cast<unsigned long long>(counted.inlined), static_cast<unsigned long long>(counted.verified), static_cast<unsigned long long>(counted.differing));
    return 0;
}

void SetVariable(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (value[0] == '\0') unsetenv(name);
    else setenv(name, value, 1);
#endif
}

int RunVerifyingChild(const char* self) {
    SetVariable("APS5_VERIFY_DISPATCH_LATER", "1");
#ifdef _WIN32
    const std::string quoted = std::string("\"") + self + "\"";
    const char* arguments[] = {quoted.c_str(), "verify", nullptr};
    const auto code = _spawnv(_P_WAIT, self, arguments);
    SetVariable("APS5_VERIFY_DISPATCH_LATER", "");
    return code < 0 ? 1 : static_cast<int>(code);
#else
    const auto child = fork();
    if (child == 0) {
        execl(self, self, "verify", static_cast<char*>(nullptr));
        _exit(127);
    }
    SetVariable("APS5_VERIFY_DISPATCH_LATER", "");
    int status = 0;
    if (child < 0 || waitpid(child, &status, 0) != child) return 1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}

}

int main(int argc, char** argv) {
    SetVariable("APS5_PIPELINED_DRAWS", "1");
    SetVariable("APS5_DRAW_PIPELINE", "1");
    SetVariable("APS5_PIPELINE_DISPATCH", "1");
    SetVariable("APS5_PIPELINE_DISPATCH_MISSES", "1");
    SetVariable("APS5_DISPATCH_LATER", "1");
    SetVariable("APS5_VERIFY_PIPELINE_ORDER", "1");
    const bool verifying = argc > 1 && std::string(argv[1]) == "verify";
    if (!verifying) {
        const auto child = RunVerifyingChild(argv[0]);
        if (child != 0) {
            std::cerr << "dispatch later: the run under APS5_VERIFY_DISPATCH_LATER exited with " << child << '\n';
            return child;
        }
    }
    try {
        const auto result = Run(verifying);
        LibcRunShutdown_nid_postfix();
        if (result == 0) std::puts(verifying ? "dispatch later tests passed under APS5_VERIFY_DISPATCH_LATER" : "dispatch later tests passed");
        return result;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        try { LibcRunShutdown_nid_postfix(); } catch (...) {}
        return 1;
    }
}
