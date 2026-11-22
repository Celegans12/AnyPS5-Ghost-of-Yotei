#include "prx/libSceAgc/DcbFlow/include/Memory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

constexpr std::uint32_t OutputHandle = 7;
constexpr std::uint32_t Extent = 128;
constexpr std::uint32_t BlockBytes = Extent * Extent * 4;
constexpr std::uint64_t Bgra8 = 0x8000000000000000ull;
constexpr std::array<std::uint32_t, 2> Colors{0xff9c632au, 0xffc275b6u};

void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

std::uint64_t address(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}

class Output final : public AgcDriver::IVideoOutput {
    class Request final : public AgcDriver::IFlipRequest {
    public:
        Request(Output& output, std::optional<AgcDriver::DisplayBuffer> source)
            : output(output), source(source) {}

        std::optional<AgcDriver::DisplayBuffer> Source() override { return source; }
        void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override { ++output.ready; }
        void Fail(std::exception_ptr error) noexcept override { output.Fail(error); }

    private:
        Output& output;
        std::optional<AgcDriver::DisplayBuffer> source;
    };

public:
    Output(const std::byte* first, const std::byte* second)
        : buffers{{{address(first), Bgra8, Extent, Extent}, {address(second), Bgra8, Extent, Extent}}} {}

    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo& flip) override {
        if (flip.index == -1) return std::make_shared<Request>(*this, std::nullopt);
        return std::make_shared<Request>(*this, buffers.at(static_cast<std::size_t>(flip.index)));
    }

    void Fail(std::exception_ptr) noexcept override { failed = true; }

    std::atomic<unsigned> ready{0};
    std::atomic<bool> failed{false};

private:
    std::array<AgcDriver::DisplayBuffer, 2> buffers;
};

void captureFrames() {
    constexpr std::size_t GuestBytes = 4 * BlockBytes;
    constexpr std::uintptr_t FixtureAddress = 0x200800000ull;
    auto* const memory = static_cast<std::byte*>(
        GuestArena::GuestArenaAllocateAtOrAbove_nid_postfix(FixtureAddress, GuestBytes, BlockBytes));
    require(memory != nullptr, "cannot allocate capture fixture memory");
    GuestArena::GuestArenaCommit_nid_postfix(memory, GuestBytes, 0x04, BlockBytes);
    {
        GuestAllocations::Mutation registration;
        registration.Add(memory, GuestBytes, true, true, true);
    }

    auto* const source = memory + BlockBytes;
    const auto output = std::make_shared<Output>(memory + 2 * BlockBytes, memory + 3 * BlockBytes);
    std::array<std::uint32_t*, Colors.size()> commands{};
    std::array<std::uint32_t, Colors.size()> lengths{};
    for (std::size_t index = 0; index < commands.size(); ++index) {
        constexpr std::size_t CommandWords = 64;
        auto*& words = commands[index];
        words = reinterpret_cast<std::uint32_t*>(memory) + index * CommandWords;
        auto* const end = words + CommandWords;
        CommandBuffer buffer{words, end, words, end, nullptr, nullptr, 0};
        constexpr std::uint8_t Memory = 0;
        const auto target = address(memory + (2 + index) * BlockBytes);
        require(sceAgcDcbDmaData(&buffer, 0, Memory, 0, target, Memory, 0, address(source),
                                BlockBytes, 0, 1, 0) != nullptr, "cannot encode a guest DMA submission");
        lengths[index] = static_cast<std::uint32_t>(buffer.cursor_up - words);
    }

    AgcDriverRegisterVideoOutput_nid_postfix(OutputHandle, output);
    AgcDriverSubmitFlip_nid_postfix({OutputHandle, -1, 1, 0});
    AgcDriverWaitIdle_nid_postfix();
    const auto draw = [&](std::size_t index) {
        std::fill_n(reinterpret_cast<std::uint32_t*>(source), BlockBytes / 4, Colors[index]);
        AgcDriverSubmitGraphics_nid_postfix(commands[index], lengths[index]);
        AgcDriverSubmitFlip_nid_postfix({OutputHandle, static_cast<std::int32_t>(index), 1, 0});
        AgcDriverWaitIdle_nid_postfix();
    };
    draw(0);
    std::async(std::launch::async, draw, 1).get();
    require(!output->failed && output->ready == Colors.size() + 1, "the live GPU did not complete all fixture flips");
    AgcDriverUnregisterVideoOutput_nid_postfix(OutputHandle, output);
    AgcDriverShutdown_nid_postfix();
}

void checkCapture(const std::filesystem::path& path) {
    using AgcDriver::FrameCallKind;
    using AgcDriver::FrameMemoryKind;
    const AgcDriver::FrameCaptureReader capture(path);
    require(!capture.Truncated() && capture.Version() == 2, "the live recorder did not finish a version 2 capture");
    const auto callsOf = [&](FrameCallKind kind) {
        return std::ranges::count(capture.Calls(), kind, &AgcDriver::FrameCall::kind);
    };
    require(callsOf(FrameCallKind::FrameStart) == 1 && callsOf(FrameCallKind::End) == 1 &&
                callsOf(FrameCallKind::Submit) == 2 && callsOf(FrameCallKind::Flip) == 2 &&
                callsOf(FrameCallKind::Memory) >= 2,
            "the live capture does not contain both guest submissions, flips and CPU updates");
    const auto thread = capture.Calls().front().thread;
    require(std::ranges::all_of(capture.Calls(), [thread](const auto& call) { return call.thread == thread; }),
            "the replay would allow another thread to overtake a recorded call");
    require(std::ranges::count(capture.Chunks(), FrameMemoryKind::Reference, &AgcDriver::FrameMemoryChunk::kind) ==
                Colors.size(), "the live capture does not contain both display references");
    for (std::size_t index = 0; index < Colors.size(); ++index) {
        const auto reference = std::ranges::find_if(capture.Chunks(), [index](const auto& chunk) {
            return chunk.kind == FrameMemoryKind::Reference && chunk.group == index;
        });
        require(reference != capture.Chunks().end(), "missing a captured display buffer index");
        const auto pixels = capture.Read(*reference);
        const auto color = Colors[index];
        require(pixels.size() == BlockBytes, "display reference has the wrong size");
        for (std::size_t byte = 0; byte < pixels.size(); ++byte) {
            require(pixels[byte] == static_cast<std::byte>((color >> ((byte % 4) * 8)) & 0xffu),
                    "the live display reference does not show the corresponding submitted color");
        }
    }
}

std::string quote(const std::filesystem::path& path) {
    return '"' + path.string() + '"';
}

std::string readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

bool run(std::string command, const std::filesystem::path& log) {
    command += " > " + quote(log) + " 2>&1";
#ifdef _WIN32
    command = '"' + command + '"';
#endif
    const bool ok = std::system(command.c_str()) == 0;
    std::cout << readFile(log);
    return ok;
}

int roundTrip(const std::filesystem::path& self, const std::filesystem::path& replay,
              const std::filesystem::path& root) {
    const auto runId = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto capture = root / std::to_string(runId);
    require(std::filesystem::create_directories(capture), "cannot create a fresh test capture directory");
#ifdef _WIN32
    require(_putenv_s("APS5_FRAME_CAPTURE", capture.string().c_str()) == 0, "cannot set the capture path");
#else
    require(setenv("APS5_FRAME_CAPTURE", capture.string().c_str(), 1) == 0, "cannot set the capture path");
#endif
    const auto recordLog = capture / "record.log";
    if (!run(quote(self) + " --record", recordLog)) {
        const auto log = readFile(recordLog);
        if (std::getenv("ANYPS5_REQUIRE_VULKAN") == nullptr &&
            (log.find("cannot load vulkan-1.dll") != std::string::npos ||
             log.find("cannot load libvulkan.so.1") != std::string::npos ||
             log.find("Unable to find a compatible Vulkan Driver") != std::string::npos ||
             log.find("Could not find suitable device:") != std::string::npos ||
             log.find("kyty presentation: the Vulkan loader offers no window surface extension") != std::string::npos)) {
            return 77;
        }
        throw std::runtime_error("live Kyty capture failed");
    }
    checkCapture(capture);
    const auto replayLog = capture / "replay.log";
    require(run(quote(replay) + " " + quote(capture) + " --strict --no-synthesize --repeat 1 --cold-pipelines", replayLog),
            "offline Kyty replay failed");
    const auto result = readFile(replayLog);
    require(result.find("\"skipped_bytes\":0,") != std::string::npos &&
                result.find("\"unfinished\":0,") != std::string::npos &&
                result.find("\"frames_compared\":2,") != std::string::npos &&
                result.find("\"pixels_identical\":1.000000,") != std::string::npos &&
                result.find("\"max_channel_diff\":0,") != std::string::npos,
            "the replay did not reproduce both captured displays byte for byte");
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected the replay executable or --record");
        if (std::string_view(argv[1]) == "--record") {
            captureFrames();
            return 0;
        }
        const char* path = std::getenv("APS5_FRAME_CAPTURE");
        require(path != nullptr && *path != '\0', "APS5_FRAME_CAPTURE is not set");
        return roundTrip(argv[0], argv[1], path);
    } catch (const std::exception& error) {
        std::cerr << "kyty_shim_frame_capture_round_trip: " << error.what() << '\n';
        return 1;
    }
}
