#include "prx/libSceAgcDriver/Backend/include/GpuBackend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct FakeHost {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> interrupts;
    std::vector<Aps5GpuFlip> reserved;
    std::vector<std::uint64_t> ready;
    std::vector<std::pair<std::uint32_t, std::int32_t>> renderingWaits;
    std::vector<std::string> lines;
    std::uint64_t nextToken = 100;
};

FakeHost& Fake(void* context) {
    return *static_cast<FakeHost*>(context);
}

Aps5GpuHost MakeHost(FakeHost& fake) {
    Aps5GpuHost host{};
    host.abi_version = APS5_GPU_BACKEND_ABI_VERSION;
    host.size = sizeof(Aps5GpuHost);
    host.context = &fake;
    host.log = [](void* context, const char* line) { Fake(context).lines.emplace_back(line); };
    host.fatal = [](void*, const char* message) {
        std::fprintf(stderr, "fatal: %s\n", message);
        std::abort();
    };
    host.raise_interrupt = [](void* context, std::uint32_t event, std::uint32_t id) { Fake(context).interrupts.emplace_back(event, id); };
    host.flip_reserve = [](void* context, const Aps5GpuFlip* flip, std::uint64_t* token) {
        Fake(context).reserved.push_back(*flip);
        *token = ++Fake(context).nextToken;
        return static_cast<int>(APS5_GPU_OK);
    };
    host.flip_wait_room = [](void*, std::uint32_t) {};
    host.flip_ready = [](void* context, std::uint64_t token) { Fake(context).ready.push_back(token); };
    host.flip_cancel = [](void*, std::uint64_t) {};
    host.display_buffer = [](void*, std::uint32_t, std::int32_t, Aps5GpuDisplayBuffer*) { return static_cast<int>(APS5_GPU_ERROR); };
    host.rendering_wait = [](void* context, std::uint32_t handle, std::int32_t index) { Fake(context).renderingWaits.emplace_back(handle, index); };
    host.frame_ready = [](void*, std::int32_t, const void*, std::uint32_t, std::uint32_t, std::uint32_t) {};
    host.backing_view = [](void*, std::uint8_t**, std::uint64_t*) { return static_cast<int>(APS5_GPU_UNSUPPORTED); };
    host.direct_mapping = [](void*, std::uint64_t, std::uint64_t*, std::uint64_t*, std::uint64_t*, std::int32_t*) { return static_cast<int>(APS5_GPU_ERROR); };
    host.mapping_generation = [](void*) { return std::uint64_t{1}; };
    host.protect_guest = [](void*, std::uint64_t, std::uint64_t, std::uint32_t) { return static_cast<int>(APS5_GPU_OK); };
    host.read_tsc = [](void*) { return std::uint64_t{3000000000}; };
    host.tsc_frequency = [](void*) { return std::uint64_t{1000000000}; };
    host.process_time_us = [](void*) { return std::uint64_t{0}; };
    host.process_time_counter = [](void*) { return std::uint64_t{0}; };
    host.is_guest_thread = [](void*) { return -1; };
    host.app_title_id = [](void*) { return "PPSA00000"; };
    host.app_content_version = [](void*) { return "01.000"; };
    return host;
}

std::uint32_t Low(const void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer));
}

std::uint32_t High(const void* pointer) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(pointer) >> 32u);
}

void IndirectBuffer(std::vector<std::uint32_t>& words, const std::vector<std::uint32_t>& target, bool chain) {
    words.insert(words.end(), {0xc0023f00u, Low(target.data()), High(target.data()), 0x0f200000u | (chain ? 1u << 20u : 0u) | static_cast<std::uint32_t>(target.size())});
}

void WriteData(std::vector<std::uint32_t>& words, const std::uint32_t* label, std::uint32_t value) {
    words.insert(words.end(), {0xc0033700u, 0x00000500u, Low(label), High(label), value});
}

void ReleaseMem(std::vector<std::uint32_t>& words, const void* label, std::uint32_t dataSelect, std::uint32_t interrupt, std::uint64_t value, std::uint32_t context) {
    words.insert(words.end(), {0xc0064900u, 0x00000014u, (interrupt << 24u) | (dataSelect << 29u), Low(label), High(label), static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32u), context});
}

void ChainsCallsLabelsAndFlips() {
    FakeHost fake;
    const auto host = MakeHost(fake);
    const Aps5GpuBackendApi* api = nullptr;
    Aps5GpuBackend* backend = nullptr;
    Require(Aps5NullGpuBackendCreate(&host, &api, &backend) == APS5_GPU_OK && api != nullptr && backend != nullptr, "create");
    Require(api->abi_version == APS5_GPU_BACKEND_ABI_VERSION && api->size == sizeof(Aps5GpuBackendApi), "interface version");
    Require(api->start(backend) == APS5_GPU_OK, "start");

    alignas(8) std::uint32_t labelA = 0;
    alignas(8) std::uint64_t labelB = 0;
    alignas(8) std::uint32_t labelC = 0;
    alignas(8) std::uint32_t labelD = 0;

    std::vector<std::uint32_t> called;
    ReleaseMem(called, &labelB, 2, 2, 0x1122334455667788ull, 0x416);
    std::vector<std::uint32_t> chained;
    chained.insert(chained.end(), {0xc0021018u, 1u, 0u, 0u});
    chained.insert(chained.end(), {0xc004105cu, 1u, 0u, 1u, 0x89abcdefu, 0x01234567u});
    ReleaseMem(chained, &labelC, 1, 0, 7, 0);
    std::vector<std::uint32_t> main;
    WriteData(main, &labelA, 0x5a5a5a5au);
    IndirectBuffer(main, called, false);
    IndirectBuffer(main, chained, true);
    WriteData(main, &labelD, 1);

    Require(api->submit(backend, 0, main.data(), static_cast<std::uint32_t>(main.size())) == APS5_GPU_OK, "submit");
    Require(labelA == 0x5a5a5a5au, "WRITE_DATA before the call");
    Require(labelB == 0x1122334455667788ull, "64-bit RELEASE_MEM in the called buffer");
    Require(labelC == 7, "32-bit RELEASE_MEM in the chained buffer");
    Require(labelD == 0, "a packet after a chain ran");
    Require(fake.renderingWaits.size() == 1 && fake.renderingWaits[0] == std::make_pair(1u, 0), "rendering wait");
    Require(fake.reserved.size() == 1 && fake.reserved[0].handle == 1 && fake.reserved[0].index == 0 && fake.reserved[0].mode == 1 && fake.reserved[0].argument == 0x0123456789abcdefll, "flip reservation");
    Require(fake.ready.size() == 1 && fake.ready[0] == fake.nextToken, "flip readiness");
    Require(fake.interrupts.size() == 2 && fake.interrupts[0] == std::make_pair(0u, 0x416u) && fake.interrupts[1] == std::make_pair(0u, 0u), "interrupts: the release's context id, then the flip's");

    const std::uint32_t truncated[] = {0xc0064900u, 0};
    Require(api->submit(backend, 0x52, truncated, 2) == APS5_GPU_OK, "submit of a truncated packet");
    std::size_t malformed = 0;
    for (const auto& line : fake.lines) malformed += line.find("malformed packet") != std::string::npos;
    Require(malformed == 1, "a truncated packet is reported once");

    api->shutdown(backend);
    Require(!fake.lines.empty() && fake.lines.back().find("1 chained and 1 called buffers") != std::string::npos, "the shutdown counts name the chain and the call");
    api->destroy(backend);
}

}

int main() {
    try {
        ChainsCallsLabelsAndFlips();
    } catch (const std::exception& error) {
        std::cerr << "null GPU backend test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "null GPU backend tests passed\n";
    return 0;
}
