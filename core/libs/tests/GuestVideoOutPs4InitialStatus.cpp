#include "SceTypes.hpp"
#include "prx/libSceVideoOut/include/Output.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "tests/VideoOutTestEnvironment.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static constexpr int SYSTEM_USER = 255;
static constexpr int MAIN_BUS = 0;

static void Require(bool value, const char* message) {
    if (!value) throw std::logic_error(message);
}

static void CheckFlipStatus(int handle) {
    alignas(VideoOutFlipStatus) std::array<std::uint8_t, sizeof(VideoOutFlipStatus)> raw{};
    raw.fill(0xaa);
    Require(sceVideoOutGetFlipStatus(handle, reinterpret_cast<VideoOutFlipStatus*>(raw.data())) == 0, "the PS4 flip status failed");
    Require(std::all_of(raw.begin() + sizeof(VideoOutFlipStatusPs4), raw.end(), [](std::uint8_t value) { return value == 0xaa; }), "the PS4 flip status overflowed its 64 bytes");
    VideoOutFlipStatusPs4 flip;
    std::memcpy(&flip, raw.data(), sizeof(flip));
    Require(flip.count == 0 && flip.flipArg == -1 && flip.currentBuffer == -1 && flip.flipPendingNum == 0, "the PS4 flip status fields are wrong");
}

static void CheckVblankStatus(int handle) {
    alignas(VideoOutVblankStatus) std::array<std::uint8_t, sizeof(VideoOutVblankStatus)> raw{};
    raw.fill(0xaa);
    Require(sceVideoOutGetVblankStatus(handle, reinterpret_cast<VideoOutVblankStatus*>(raw.data())) == 0, "the PS4 vblank status failed");
    Require(std::all_of(raw.begin() + sizeof(VideoOutVblankStatusPs4), raw.end(), [](std::uint8_t value) { return value == 0xaa; }), "the PS4 vblank status overflowed its 32 bytes");
}

static int Run() {
    VideoOutTestEnvironment environment(true);
    int handle = 0;
    try {
        handle = sceVideoOutOpen(SYSTEM_USER, MAIN_BUS, 0, nullptr);
    } catch (const std::runtime_error& error) {
        if (std::getenv("ANYPS5_REQUIRE_DISPLAY") != nullptr) throw;
        std::printf("skipped, no display or Vulkan device: %s\n", error.what());
        return 77;
    }
    Require(handle > 0, "sceVideoOutOpen failed");
    CheckFlipStatus(handle);
    CheckVblankStatus(handle);
    Require(sceVideoOutClose(handle) == 0, "closing the PS4 video out handle failed");
    LibcRunShutdown_nid_postfix();
    std::puts("PS4 initial VideoOut status tests passed");
    return 0;
}

int main() {
    try {
        return Run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
