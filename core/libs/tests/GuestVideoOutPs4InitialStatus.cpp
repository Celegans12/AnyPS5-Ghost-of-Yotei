#include "SceTypes.hpp"
#include "prx/libSceVideoOut/include/Buffer.hpp"
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
#include <string_view>

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

template<typename Operation>
static void RequireInvalidOption(Operation operation) {
    try {
        operation();
    } catch (const std::runtime_error& error) {
        Require(std::string_view(error.what()).find("VIDEO_OUT_ERROR_INVALID_OPTION") != std::string_view::npos, error.what());
        return;
    }
    throw std::logic_error("invalid PS4 VR option accepted");
}

static void CheckVrOptions(int handle) {
    alignas(4096) static std::array<std::uint8_t, 3 * 32768> storage{};
    VideoOutBufferAttribute attribute{};
    sceVideoOutSetBufferAttribute(&attribute, 0x80000000u, 1, 0, 64, 64, 64);
    constexpr std::array<std::uint32_t, 3> options{0, 7, 15};
    for (int index = 0; index < 3; ++index) {
        void* address = storage.data() + index * 32768;
        attribute.option = options[index];
        Require(sceVideoOutRegisterBuffers(handle, index, &address, 1, &attribute) == index, "PS4 linear VR buffer registration failed");
    }
    attribute.pitch_in_pixel = 128;
    Require(sceVideoOutSubmitChangeBufferAttribute(handle, 0, &attribute) == 0, "PS4 VR option update failed");
    void* address = storage.data();
    for (const auto option : {3u, 6u, 0x17u}) {
        attribute.option = option;
        RequireInvalidOption([&] { sceVideoOutRegisterBuffers(handle, 3, &address, 1, &attribute); });
        RequireInvalidOption([&] { sceVideoOutSubmitChangeBufferAttribute(handle, 0, &attribute); });
    }
}

static int Run(bool vrOptions) {
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
    if (vrOptions) CheckVrOptions(handle);
    else {
        CheckFlipStatus(handle);
        CheckVblankStatus(handle);
    }
    Require(sceVideoOutClose(handle) == 0, "closing the PS4 video out handle failed");
    LibcRunShutdown_nid_postfix();
    return 0;
}

int main(int argc, char** argv) {
    try {
        const bool vrOptions = argc == 2 && std::string_view(argv[1]) == "vr-options";
        Require(argc == 1 || vrOptions, "unknown PS4 VideoOut test mode");
        return Run(vrOptions);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
