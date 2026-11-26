#include "prx/libSceAgcDriver/KytyShim/src/Replay.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/EntryApi.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#if !defined(ANYPS5_KYTY_DLL)
#include "prx/libSceAgcDriver/Execution/include/FrameRecorder.hpp"
#include "kernel/memory.h"
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/uio.h>
#include <unistd.h>
#endif
#endif

#include "common/debugCounters.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>


namespace KytyShim::Replay {
namespace {

constexpr std::uint32_t ComputeQueueBase = 0x20;
constexpr std::uint32_t ReleaseMemHeader = 0xc0064900u;
constexpr std::uint32_t BottomOfPipeEvent = 0x28u | (5u << 8u);
constexpr std::uint32_t LabelData64 = 2u << 29u;
constexpr std::uint64_t ParsedMargin = 64;
constexpr std::size_t WaitCapacity = 64;
constexpr std::int64_t HoldGraceMs = 1000;
constexpr unsigned TrackReplaySerials = 1;
constexpr unsigned TrackCaptureSerials = 2;

struct LabelPacket {
    std::uint64_t serial = 0;
    std::array<std::uint32_t, 8> words{};
};

struct Queue {
    std::mutex mutex;
    std::atomic<std::uint64_t> accepted{0};
    std::atomic<std::uint64_t> completed{0};
    std::deque<LabelPacket> packets;
};

struct State {
    std::atomic<unsigned> tracking{0};
    std::array<Queue, SerialQueues> queues;
    std::mutex holdMutex;
    std::atomic<bool> held{false};
};

State& state() {
    static auto* instance = new State;
    return *instance;
}

std::size_t QueueSlot(std::uint32_t queue) {
    if (queue == 0) return 0;
    if (queue < ComputeQueueBase || queue - ComputeQueueBase >= SerialQueues - 1) throw std::runtime_error("kyty replay serials: queue " + std::to_string(queue) + " is not a Kyty queue");
    return 1 + (queue - ComputeQueueBase);
}

std::uint64_t Completed(std::size_t slot) {
    auto& queue = state().queues[slot];
    const auto label = std::atomic_ref<std::uint64_t>(Entry::SerialLabels()[slot]).load(std::memory_order_acquire);
    const auto accepted = queue.accepted.load(std::memory_order_acquire);
    auto known = queue.completed.load(std::memory_order_acquire);
    while (label > known && label <= accepted && !queue.completed.compare_exchange_weak(known, label, std::memory_order_acq_rel)) {
    }
    return std::max(known, label <= accepted ? label : known);
}

void RequireTracking(const char* what) {
    if (!state().tracking.load(std::memory_order_acquire)) throw std::runtime_error(std::string("kyty GPU backend: ") + what + " counts submissions only after AgcDriverTrackSerials (agc_frame_replay calls it before its first submission)");
}

std::pair<std::uint64_t, std::uint64_t> Progress() {
    auto& s = state();
    std::uint64_t completed = 0;
    for (std::size_t slot = 0; slot < SerialQueues; ++slot) completed += Completed(slot);
    std::uint64_t accepted = 0;
    for (const auto& queue : s.queues) accepted += queue.accepted.load(std::memory_order_acquire);
    return {accepted, completed};
}

std::size_t Blocked(std::array<Libs::Graphics::GuestGpu::BlockedWait, WaitCapacity>& waits) {
    if (!Entry::GpuStarted()) return 0;
    return Entry::Gpu().BlockedWaits(waits.data(), waits.size());
}

bool Settled() {
    std::array<Libs::Graphics::GuestGpu::BlockedWait, WaitCapacity> waits{};
    const auto count = Blocked(waits);
    auto& s = state();
    for (std::size_t slot = 0; slot < SerialQueues; ++slot) {
        const auto done = Completed(slot);
        if (s.queues[slot].accepted.load(std::memory_order_acquire) <= done) continue;
        const bool parked = std::any_of(waits.begin(), waits.begin() + static_cast<std::ptrdiff_t>(count), [slot](const auto& wait) { return wait.queue == slot; });
        if (!parked) return false;
    }
    return true;
}

bool Readable(vk::Format format) {
    switch (format) {
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eA2R10G10B10UnormPack32:
        return true;
    default:
        return false;
    }
}

std::runtime_error Unreadable(vk::Format format) {
    return std::runtime_error("kyty replay: reading a display image of Vulkan format " + vk::to_string(format) + " is not implemented (8-bit and 10-bit RGBA/BGRA only)");
}

std::array<std::byte, 4> Bgra(vk::Format format, const std::uint8_t* texel) {
    switch (format) {
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
        return {std::byte{texel[0]}, std::byte{texel[1]}, std::byte{texel[2]}, std::byte{texel[3]}};
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return {std::byte{texel[2]}, std::byte{texel[1]}, std::byte{texel[0]}, std::byte{texel[3]}};
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eA2R10G10B10UnormPack32: {
        std::uint32_t word = 0;
        std::memcpy(&word, texel, sizeof(word));
        const auto low = static_cast<std::byte>((word >> 2u) & 0xffu);
        const auto green = static_cast<std::byte>((word >> 12u) & 0xffu);
        const auto high = static_cast<std::byte>((word >> 22u) & 0xffu);
        if (format == vk::Format::eA2B10G10R10UnormPack32) return {high, green, low, std::byte{0xff}};
        return {low, green, high, std::byte{0xff}};
    }
    default:
        throw Unreadable(format);
    }
}

}

bool Tracking() {
    return state().tracking.load(std::memory_order_acquire);
}

bool Replaying() {
    return (state().tracking.load(std::memory_order_acquire) & TrackReplaySerials) != 0;
}

void RequireNotHeld(const char* what) {
    if (state().held.load(std::memory_order_acquire)) throw std::runtime_error(std::string("kyty replay: ") + what + " waits for the GPU while the replay holds it settled (AgcDriverReleaseSettled first)");
}

void NoteSubmission(Libs::Graphics::GuestGpu& gpu, std::uint32_t queue) {
    auto& s = state();
    if (!s.tracking.load(std::memory_order_acquire)) return;
    const auto slot = QueueSlot(queue);
    auto& entry = s.queues[slot];
    std::lock_guard lock(entry.mutex);
    const auto done = Completed(slot);
    while (!entry.packets.empty() && entry.packets.front().serial + ParsedMargin <= done) entry.packets.pop_front();
    const auto serial = entry.accepted.load(std::memory_order_relaxed) + 1;
    const auto address = reinterpret_cast<std::uint64_t>(Entry::SerialLabels() + slot);
    auto& packet = entry.packets.emplace_back();
    packet.serial = serial;
    packet.words = {ReleaseMemHeader, BottomOfPipeEvent, LabelData64, static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32u), static_cast<std::uint32_t>(serial), static_cast<std::uint32_t>(serial >> 32u), 0};
    entry.accepted.store(serial, std::memory_order_release);
    const std::span<const std::uint32_t> words(packet.words);
    if (queue == 0) gpu.Submit(words);
    else gpu.SubmitCompute(queue, words);
}

}

namespace AgcDriver {

#if !defined(ANYPS5_KYTY_DLL)
void StartFrameCaptureSerials() {
    static_cast<void>(KytyShim::Entry::SerialLabels());
    KytyShim::Replay::state().tracking.fetch_or(KytyShim::Replay::TrackCaptureSerials, std::memory_order_release);
}

void StopFrameCaptureSerials() noexcept {
    KytyShim::Replay::state().tracking.fetch_and(~KytyShim::Replay::TrackCaptureSerials, std::memory_order_release);
}

bool ReadFrameCaptureMemory(std::uint64_t address, std::span<std::byte> destination) {
    namespace Memory = Libs::LibKernel::Memory;
    if (destination.empty()) return true;
    if (address > std::numeric_limits<std::uintptr_t>::max() - destination.size()) return false;
    if (KytyShim::Entry::GpuStarted() && Libs::Graphics::GuestRange{address, destination.size()}.Valid()) {
        auto& renderer = KytyShim::Entry::Renderer();
        auto& buffers = renderer.GetBufferCache();
        bool read = false;
        bool needsGpu = true;
        try {
            renderer.GetGpu().SendCommandSync([&] {
                try {
                    const auto size = destination.size();
                    if (renderer.GetTextureCache().IsRegionGpuModified(address, size, true) ||
                        buffers.DeferredLabelPending(address, size)) return;
                    if (!buffers.HasGpuDirtyBytes(address, size)) {
                        needsGpu = false;
                        return;
                    }
                    if (buffers.TryReadMirroredGpuWrites(address, destination.data(), size)) {
                        read = true;
                        return;
                    }
                    buffers.ReadMemory(address, size);
                    if (!buffers.HasGpuDirtyBytes(address, size)) {
                        read = Memory::TryReadBacking(address, destination.data(), size);
                    }
                } catch (const std::exception&) {
                    read = false;
                }
            });
        } catch (const std::exception&) {
            return false;
        }
        if (needsGpu) return read;
    }
    if (Memory::TryReadBacking(address, destination.data(), destination.size())) return true;
    const auto* source = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address));
#if defined(_WIN32)
    SIZE_T received = 0;
    return ReadProcessMemory(GetCurrentProcess(), source, destination.data(), destination.size(), &received) != 0 && received == destination.size();
#else
    const iovec local{destination.data(), destination.size()};
    const iovec remote{const_cast<void*>(source), destination.size()};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(destination.size());
#endif
}
#endif

std::vector<std::byte> ReadDisplayBuffer(const DisplayBuffer& buffer) {
    KytyShim::Replay::RequireNotHeld("ReadDisplayBuffer");
    AgcDriverWaitIdle_nid_postfix();
    const auto info = KytyShim::Present::DisplayBufferImage(buffer);
    auto& renderer = KytyShim::Entry::Renderer();
    std::vector<std::byte> pixels(static_cast<std::size_t>(buffer.width) * buffer.height * 4);
    std::exception_ptr failure;
    renderer.GetGpu().SendCommandSync([&] {
        try {
            namespace Graphics = Libs::Graphics;
            auto& scheduler = renderer.GetCommandScheduler();
            if (!scheduler.Active()) throw std::runtime_error("kyty replay: the Kyty GPU has recorded no command buffer, so it holds no display image to read");
            Common::LockGuard lock(renderer.GetMutex());
            auto& cache = renderer.GetTextureCache();
            Graphics::TextureCache::ImageDesc desc{};
            desc.info = info;
            desc.view_info.format = info.pixel_format;
            desc.view_info.type = vk::ImageViewType::e2D;
            desc.view_info.aspect = vk::ImageAspectFlagBits::eColor;
            desc.view_info.base_level = 0;
            desc.view_info.level_count = 1;
            desc.view_info.base_layer = 0;
            desc.view_info.layer_count = 1;
            desc.view_info.usage = vk::ImageUsageFlagBits::eTransferSrc;
            desc.type = Graphics::TextureCache::BindingType::VideoOut;
            const auto id = cache.FindImage(desc);
            auto& image = cache.GetImage(id);
            cache.UpdateImage(id);
            const auto format = image.backing.format;
            if (!KytyShim::Replay::Readable(format)) throw KytyShim::Replay::Unreadable(format);
            if (image.backing.extent.width < buffer.width || image.backing.extent.height < buffer.height) throw std::runtime_error("kyty replay: the display image is smaller than the display buffer");
            const auto bytes = static_cast<std::uint64_t>(buffer.width) * buffer.height * 4;
            auto download = renderer.GetBufferCache().GetUtilityBuffer(Graphics::MemoryUsage::Download).MapDownload(bytes, 4);
            vk::BufferImageCopy copy{};
            copy.bufferOffset = download.offset;
            copy.imageSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
            copy.imageExtent = vk::Extent3D{buffer.width, buffer.height, 1};
            image.Download(std::span(&copy, 1), download.buffer->Handle(), download.offset, bytes);
            scheduler.Finish();
            download.buffer->Invalidate(download.offset, bytes);
            const auto* source = download.mapped;
            for (std::size_t pixel = 0; pixel < pixels.size() / 4; ++pixel) {
                const auto converted = KytyShim::Replay::Bgra(format, source + pixel * 4);
                std::memcpy(pixels.data() + pixel * 4, converted.data(), converted.size());
            }
        } catch (...) {
            failure = std::current_exception();
        }
    });
    if (failure) std::rethrow_exception(failure);
    return pixels;
}

}

extern "C" void AgcDriverTrackSerials_nid_postfix() {
    static_cast<void>(KytyShim::Entry::SerialLabels());
    KytyShim::Replay::state().tracking.fetch_or(KytyShim::Replay::TrackReplaySerials, std::memory_order_release);
}

extern "C" void AgcDriverSerialProgress_nid_postfix(std::uint64_t* accepted, std::uint64_t* completed) {
    KytyShim::Replay::RequireTracking("AgcDriverSerialProgress");
    const auto [submitted, done] = KytyShim::Replay::Progress();
    *accepted = submitted;
    *completed = done;
}

extern "C" bool AgcDriverSettled_nid_postfix() {
    KytyShim::Replay::RequireTracking("AgcDriverSettled");
    return KytyShim::Replay::Settled();
}

extern "C" bool AgcDriverHoldSettled_nid_postfix(std::int64_t milliseconds) {
    KytyShim::Replay::RequireTracking("AgcDriverHoldSettled");
    auto& s = KytyShim::Replay::state();
    std::lock_guard lock(s.holdMutex);
    if (s.held) return KytyShim::Replay::Settled();
    if (!KytyShim::Entry::GpuStarted()) {
        s.held = true;
        return true;
    }
    auto& gpu = KytyShim::Entry::Gpu();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max<std::int64_t>(0, milliseconds));
    for (;;) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (!gpu.HoldSubmissions(static_cast<std::uint32_t>(std::clamp<std::int64_t>(left, KytyShim::Replay::HoldGraceMs, std::numeric_limits<std::uint32_t>::max())))) return false;
        if (KytyShim::Replay::Settled()) {
            s.held = true;
            return true;
        }
        gpu.ReleaseSubmissions();
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

extern "C" void AgcDriverReleaseSettled_nid_postfix() {
    auto& s = KytyShim::Replay::state();
    std::lock_guard lock(s.holdMutex);
    if (!s.held) return;
    s.held = false;
    if (KytyShim::Entry::GpuStarted()) KytyShim::Entry::Gpu().ReleaseSubmissions();
}

extern "C" std::size_t AgcDriverAwaitedWaits_nid_postfix(AgcDriver::AwaitedWait* waits, std::size_t capacity) {
    std::array<Libs::Graphics::GuestGpu::BlockedWait, KytyShim::Replay::WaitCapacity> blocked{};
    const auto count = std::min(capacity, KytyShim::Replay::Blocked(blocked));
    for (std::size_t index = 0; index < count; ++index) {
        const auto& wait = blocked[index];
        waits[index] = AgcDriver::AwaitedWait{wait.address, wait.reference, wait.mask, wait.function, wait.bytes};
    }
    return count;
}

extern "C" void AgcDriverWaitForPipelineJournal_nid_postfix() {
    auto& renderer = KytyShim::Entry::Renderer();
    if (std::getenv("APS5_REPLAY_WAIT_MANIFEST") != nullptr &&
        !renderer.GetPipelineCache().WaitForManifestCompletion(std::chrono::seconds(30))) {
        throw std::runtime_error("frame replay: the pipeline manifest did not finish preparing its shaders and pipelines");
    }
    AgcDriverWaitIdle_nid_postfix();
}

extern "C" std::size_t AgcDriverReplayCounters_nid_postfix(std::uint64_t* values, std::size_t capacity) {
    using Common::DebugCounters::Counter;
    static constexpr std::array Reported{Counter::Draws, Counter::Dispatches, Counter::QueueSubmits, Counter::TextureUploadBytes, Counter::BufferUploadBytes, Counter::TextureDownloadBytes, Counter::BufferDownloadBytes, Counter::GpuFaults};
    Common::DebugCounters::SetEnabled(true);
    const auto count = std::min(capacity, Reported.size());
    for (std::size_t index = 0; index < count; ++index) values[index] = Common::DebugCounters::Get(Reported[index]);
    return count;
}

extern "C" void AgcDriverTracePresent_nid_postfix(std::uint32_t width, std::uint32_t height) {
    static_cast<void>(width);
    static_cast<void>(height);
}
