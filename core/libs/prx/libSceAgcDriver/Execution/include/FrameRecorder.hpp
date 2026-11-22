#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FRAMERECORDER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_FRAMERECORDER_HPP

#include "prx/libSceAgcDriver/Execution/include/FrameCapture.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "SceTypes.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <optional>

namespace AgcDriver {

bool ReadFrameCaptureMemory(std::uint64_t address, std::span<std::byte> destination);
bool IsFrameCaptureDriverMemory(std::uint64_t address, std::uint64_t bytes);
std::vector<std::uint32_t> FrameCapturePendingEopTokens();
void StartFrameCaptureSerials();
void StopFrameCaptureSerials() noexcept;

class FrameRecorder {
public:
    static FrameRecorder* Active();

    void RegisterShader(const Shader* shader, const std::function<void()>& accept = {});
    void ResolveShaderAbi(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive);
    void ResolveGraphicsStagesAbi(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive);
    void ResolveGraphicsAbi(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType);
    void RegisterVideoOutput(std::uint32_t handle, const std::function<void()>& accept = {});
    void UnregisterVideoOutput(std::uint32_t handle, const std::function<void()>& accept = {});
    void SetTessellationFactorRing(std::uint64_t address, std::uint32_t size, const std::function<void()>& accept);
    void Submit(const Packet& packet, std::uint32_t queue, const std::function<bool()>& accept);
    void Flip(const FlipInfo& info, const std::function<void()>& accept);
    std::uint32_t ReserveEopFlip(const FlipInfo& info, const std::function<std::uint32_t()>& reserve);
    void Suspend(const std::function<void()>& accept);
    void PresentBuffer(const DisplayBuffer& buffer);

private:
    enum class State { Waiting, Recording, Done };

    struct Range {
        std::uint64_t address;
        std::uint64_t bytes;
        FrameRangeKind kind;
        std::uint32_t protection;
        std::uint32_t access;
        std::vector<std::uint64_t> hashes;
    };

    FrameRecorder(std::filesystem::path directory, std::filesystem::path trigger, std::uint64_t atBoundary, std::uint64_t frames);
    FrameCall make(FrameCallKind kind) const;
    void write(FrameCall call);
    void record(FrameCall call, const std::function<void()>& accept = {});
    void reserveEopFlipLocked(const FlipInfo& info, std::uint32_t token);
    void handOff(FrameCall call, const std::function<bool()>& accept);
    void frameBoundaryLocked();
    void start();
    void finish();
    void emitMemory(bool snapshot);
    void collectRanges();
    bool covers(std::uint64_t address, std::uint64_t bytes) const;
    void checkDisk() const;
    void stop(const char* reason) noexcept;

    const std::filesystem::path directory;
    const std::filesystem::path trigger;
    const std::uint64_t atBoundary;
    const std::chrono::steady_clock::time_point origin;
    std::uint64_t frames;
    std::mutex mutex;
    State state = State::Waiting;
    std::atomic<bool> done{false};
    std::unique_ptr<FrameCaptureWriter> writer;
    std::vector<FrameCall> prologue;
    std::optional<FrameCall> initialTessellationRing;
    std::map<std::uint32_t, FrameCall> reservations;
    std::vector<Range> ranges;
    std::mutex displayMutex;
    std::map<std::uint64_t, DisplayBuffer> displayBuffers;
    bool collectDisplay = false;
    std::uint64_t boundaries = 0;
    std::uint64_t recordedFrames = 0;
    std::uint64_t startBoundary = 0;
    std::uint64_t attempts = 0;
    std::uint64_t ordinal = 0;
    std::uint64_t memoryGroups = 0;
    std::uint64_t allocationGeneration = 0;
    std::uint64_t unrecordedAccepts = 0;
};

}

#endif
