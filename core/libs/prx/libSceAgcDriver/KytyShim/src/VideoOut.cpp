#include "common/assert.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/presentation/precompileNotice.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/presentation/window/windowInternal.h"
#include "libs/errno.h"

#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/KeyboardOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/Replay.hpp"
#include "prx/libc/include/RunReport.hpp"
#include "prx/libc/include/Shutdown.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace KytyShim::Entry {

Libs::Graphics::RenderContext& Renderer();
void JoinPrestart(bool rethrow);
std::shared_ptr<AgcDriver::IVideoOutput> VideoOutput(std::uint32_t handle);

}

namespace KytyShim::Present {

void WaitRendering(std::uint32_t handle, std::uint32_t index);

namespace {

constexpr std::array<std::array<float, 4>, AgcDriver::OverlayColorCount> OverlayBasePalette{{{0.0f, 0.0f, 0.0f, 0.6f}, {1.0f, 1.0f, 1.0f, 1.0f}, {0.35f, 0.8f, 0.35f, 1.0f}, {0.06f, 0.07f, 0.09f, 0.94f}, {0.22f, 0.24f, 0.29f, 1.0f}, {1.0f, 0.78f, 0.2f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}}};

struct PendingFlip {
    std::shared_ptr<AgcDriver::IFlipRequest> request;
    std::uint32_t handle = 0;
    int index = 0;
    std::optional<AgcDriver::DisplayBuffer> source;
    Libs::Graphics::Presenter::Frame* frame = nullptr;
    bool dropped = false;
};

struct ReadyFlip {
    std::weak_ptr<AgcDriver::IFlipRequest> request;
    const void* owner = nullptr;
    std::uint32_t handle = 0;
    int index = 0;
    std::optional<AgcDriver::DisplayBuffer> source;
    Libs::Graphics::Presenter::Frame* frame = nullptr;
};

struct PresentationState {
    std::mutex mutex;
    std::unique_ptr<Libs::Graphics::WindowContext> window;
    std::unique_ptr<Libs::VideoOut::VideoOutDriver> driver;
    std::map<std::uint64_t, PendingFlip> flips;
    std::vector<ReadyFlip> ready;
    std::uint64_t nextFlip = 1;
    void* windowHandle = nullptr;
    bool attaching = false;
    VkSurfaceKHR (*createSurface)(void* context, VkInstance instance) = nullptr;
};

PresentationState& State() {
    static auto* state = new PresentationState;
    return *state;
}

[[noreturn]] void Stop(const char* where, std::exception_ptr error) noexcept {
    try {
        std::rethrow_exception(error);
    } catch (const ProcessShutdown&) {
        try {
            LibcAwaitExit_nid_postfix();
        } catch (...) {
        }
        std::fprintf(stderr, "[kyty-videoout] %s: process shutdown outside an exit\n", where);
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "[kyty-videoout] %s: %s\n", where, failure.what());
    } catch (...) {
        std::fprintf(stderr, "[kyty-videoout] %s: unknown failure\n", where);
    }
    std::fflush(stderr);
    LibcTerminate_nid_no_patch(1);
}

class BlockingWaiter {
public:
    BlockingWaiter() {
        std::thread([this] { run(); }).detach();
    }

    std::future<void> Start(std::function<void()> blocking) {
        std::packaged_task<void()> task(std::move(blocking));
        auto done = task.get_future();
        {
            std::lock_guard lock(mutex);
            tasks.push_back(std::move(task));
        }
        ready.notify_one();
        return done;
    }

private:
    void run() {
        for (;;) {
            std::packaged_task<void()> task;
            {
                std::unique_lock lock(mutex);
                ready.wait(lock, [this] { return !tasks.empty(); });
                task = std::move(tasks.front());
                tasks.pop_front();
            }
            task();
        }
    }

    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::packaged_task<void()>> tasks;
};

void WaitRunningUrgentCommands(std::function<void()> blocking) {
    static auto* const waiter = new BlockingWaiter;
    auto waiting = waiter->Start(std::move(blocking));
    while (waiting.wait_for(std::chrono::microseconds(200)) != std::future_status::ready) {
        Libs::Graphics::GuestGpu::RunUrgentCommandsWhileWaiting();
    }
    waiting.get();
}

struct LoaderLock {
    LoaderLock() { AgcDriverLockVulkanLoader_nid_postfix(); }
    ~LoaderLock() { AgcDriverUnlockVulkanLoader_nid_postfix(); }
    LoaderLock(const LoaderLock&) = delete;
    LoaderLock& operator=(const LoaderLock&) = delete;
};

PFN_vkGetInstanceProcAddr LoaderGetInstanceProcAddr() {
    static const PFN_vkGetInstanceProcAddr proc = [] {
        const LoaderLock lock;
#ifdef _WIN32
        const HMODULE library = LoadLibraryW(L"vulkan-1.dll");
        if (library == nullptr) EXIT("kyty presentation: cannot load vulkan-1.dll: error %lu\n", static_cast<unsigned long>(GetLastError()));
        const auto found = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
        if (found == nullptr) EXIT("kyty presentation: vulkan-1.dll has no vkGetInstanceProcAddr\n");
#else
        void* const library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) EXIT("kyty presentation: cannot load libvulkan.so.1: %s\n", dlerror());
        const auto found = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
        if (found == nullptr) EXIT("kyty presentation: libvulkan.so.1 has no vkGetInstanceProcAddr\n");
#endif
        return found;
    }();
    return proc;
}

template <typename Function>
Function LoaderGlobal(const char* name) {
    const auto function = reinterpret_cast<Function>(LoaderGetInstanceProcAddr()(VK_NULL_HANDLE, name));
    if (function == nullptr) EXIT("kyty presentation: the Vulkan loader has no %s\n", name);
    return function;
}

VKAPI_ATTR VkResult VKAPI_CALL LockedCreateInstance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* allocator, VkInstance* instance) {
    const LoaderLock lock;
    return LoaderGlobal<PFN_vkCreateInstance>("vkCreateInstance")(info, allocator, instance);
}

VKAPI_ATTR VkResult VKAPI_CALL LockedEnumerateInstanceExtensionProperties(const char* layer, std::uint32_t* count, VkExtensionProperties* properties) {
    const LoaderLock lock;
    return LoaderGlobal<PFN_vkEnumerateInstanceExtensionProperties>("vkEnumerateInstanceExtensionProperties")(layer, count, properties);
}

VKAPI_ATTR VkResult VKAPI_CALL LockedEnumerateInstanceLayerProperties(std::uint32_t* count, VkLayerProperties* properties) {
    const LoaderLock lock;
    return LoaderGlobal<PFN_vkEnumerateInstanceLayerProperties>("vkEnumerateInstanceLayerProperties")(count, properties);
}

VKAPI_ATTR VkResult VKAPI_CALL LockedEnumerateInstanceVersion(std::uint32_t* version) {
    const LoaderLock lock;
    return LoaderGlobal<PFN_vkEnumerateInstanceVersion>("vkEnumerateInstanceVersion")(version);
}

VKAPI_ATTR VkResult VKAPI_CALL LockedEnumeratePhysicalDevices(VkInstance instance, std::uint32_t* count, VkPhysicalDevice* devices) {
    const LoaderLock lock;
    const auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(LoaderGetInstanceProcAddr()(instance, "vkEnumeratePhysicalDevices"));
    if (enumerate == nullptr) EXIT("kyty presentation: the Vulkan loader has no vkEnumeratePhysicalDevices\n");
    return enumerate(instance, count, devices);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL LockedGetInstanceProcAddr(VkInstance instance, const char* name) {
    const auto loader = LoaderGetInstanceProcAddr();
    if (name != nullptr && std::strcmp(name, "vkGetInstanceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&LockedGetInstanceProcAddr);
    if (name != nullptr && instance != VK_NULL_HANDLE && std::strcmp(name, "vkEnumeratePhysicalDevices") == 0) {
        return loader(instance, name) == nullptr ? nullptr : reinterpret_cast<PFN_vkVoidFunction>(&LockedEnumeratePhysicalDevices);
    }
    if (instance == VK_NULL_HANDLE && name != nullptr) {
        if (std::strcmp(name, "vkCreateInstance") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&LockedCreateInstance);
        if (std::strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&LockedEnumerateInstanceExtensionProperties);
        if (std::strcmp(name, "vkEnumerateInstanceLayerProperties") == 0) return reinterpret_cast<PFN_vkVoidFunction>(&LockedEnumerateInstanceLayerProperties);
        if (std::strcmp(name, "vkEnumerateInstanceVersion") == 0) {
            return loader(VK_NULL_HANDLE, name) == nullptr ? nullptr : reinterpret_cast<PFN_vkVoidFunction>(&LockedEnumerateInstanceVersion);
        }
    }
    return loader(instance, name);
}

PFN_vkGetInstanceProcAddr LoadVulkanLoader() {
    LoaderGetInstanceProcAddr();
    return &LockedGetInstanceProcAddr;
}

std::vector<const char*> SurfaceInstanceExtensions(PFN_vkGetInstanceProcAddr proc) {
    const auto enumerate = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(proc(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
    if (enumerate == nullptr) EXIT("kyty presentation: the Vulkan loader has no vkEnumerateInstanceExtensionProperties\n");
    std::uint32_t count = 0;
    if (enumerate(nullptr, &count, nullptr) != VK_SUCCESS) EXIT("kyty presentation: cannot list the Vulkan instance extensions\n");
    std::vector<VkExtensionProperties> available(count);
    if (enumerate(nullptr, &count, available.data()) != VK_SUCCESS) EXIT("kyty presentation: cannot list the Vulkan instance extensions\n");
    available.resize(count);
#ifdef _WIN32
    static constexpr const char* Candidates[] = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_win32_surface"};
#else
    static constexpr const char* Candidates[] = {VK_KHR_SURFACE_EXTENSION_NAME, "VK_KHR_xlib_surface", "VK_KHR_xcb_surface", "VK_KHR_wayland_surface"};
#endif
    std::vector<const char*> enabled;
    for (const char* name : Candidates) {
        for (const auto& extension : available) {
            if (std::strcmp(extension.extensionName, name) == 0) {
                enabled.push_back(name);
                break;
            }
        }
    }
    if (enabled.empty() || enabled.front() != Candidates[0] || enabled.size() == 1) EXIT("kyty presentation: the Vulkan loader offers no window surface extension (win32 on Windows; xlib, xcb or wayland on Linux)\n");
    return enabled;
}

VkSurfaceKHR CreateHostSurface(void* context, VkInstance instance) {
    try {
        auto& state = State();
        if (state.createSurface == nullptr) throw std::runtime_error("no window surface callback");
        return state.createSurface(context, instance);
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "[kyty-videoout] window surface creation failed: %s\n", failure.what());
    } catch (...) {
        std::fprintf(stderr, "[kyty-videoout] window surface creation failed\n");
    }
    return VK_NULL_HANDLE;
}

Libs::Graphics::WindowContext& AttachedWindow(const AgcDriver::PresentationWindow& window) {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    if (state.window == nullptr || state.windowHandle == nullptr || state.windowHandle != window.context) throw std::runtime_error("kyty presentation: the presentation window is not attached");
    return *state.window;
}

void UpdateDrawableSize(Libs::Graphics::WindowContext& host, const AgcDriver::PresentationWindow& window) {
    if (window.getDrawableSize == nullptr) throw std::runtime_error("kyty presentation: missing window drawable size query");
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    window.getDrawableSize(window.context, &width, &height);
    if (width == 0 || height == 0) return;
    Common::LockGuard lock(host.mutex);
    host.graphic_ctx.screen_width = width;
    host.graphic_ctx.screen_height = height;
}

Libs::Graphics::ImageInfo DisplayImageInfo(const AgcDriver::DisplayBuffer& buffer) {
    const auto compression = Libs::Graphics::ClassifyVideoOutCompression(buffer.dccAddress != 0, buffer.dccAddress, buffer.dccControl, buffer.dccClearColor);
    if (compression == Libs::Graphics::VideoOutCompression::Unsupported) throw std::runtime_error("kyty presentation: unsupported DCC display buffer, metadata " + std::to_string(buffer.dccAddress) + ", control " + std::to_string(buffer.dccControl) + ", clear color " + std::to_string(buffer.dccClearColor));
    if (buffer.tilingMode != 0 || buffer.pitchInPixel != 0) throw std::runtime_error("kyty presentation: only render-target tiled display buffers are supported, tiling " + std::to_string(buffer.tilingMode) + ", pitch " + std::to_string(buffer.pitchInPixel));
    if (buffer.width == 0 || buffer.height == 0 || buffer.width > 16384 || buffer.height > 16384) throw std::runtime_error("kyty presentation: invalid display buffer extent");
    if (buffer.address == 0 || (buffer.address & 65535u) != 0) throw std::runtime_error("kyty presentation: display buffer address is not 64 KiB aligned");
    Libs::Graphics::VideoOutPixelFormatInfo format{};
    if (!Libs::Graphics::DecodeVideoOutPixelFormat(buffer.pixelFormat, format)) throw std::runtime_error("kyty presentation: unsupported display pixel format " + std::to_string(buffer.pixelFormat));
    Libs::Graphics::ImageInfo info{};
    info.data = {buffer.address, 0};
    info.pixel_format = format.format;
    info.guest_format = format.guest_format;
    info.type = Libs::Graphics::Prospero::ImageType::kColor2D;
    info.extent = vk::Extent3D{buffer.width, buffer.height, 1};
    info.resources = {1, 1};
    info.bytes_per_block = format.bytes_per_element;
    info.samples = 1;
    info.tile_mode = Libs::Graphics::Prospero::TileMode::kRenderTarget;
    info.bgra16 = format.bgra16;
    info.UpdateSize();
    if (info.data.size == 0 || info.tiling.block.block_size != 65536) throw std::runtime_error("kyty presentation: invalid display buffer footprint");
    if (compression != Libs::Graphics::VideoOutCompression::Uncompressed) {
        Libs::Graphics::TileSizeAlign dccSize{};
        if (!Libs::Graphics::TileGetDccSize(buffer.width, buffer.height, 1, format.bytes_per_element, 1, Libs::Graphics::Prospero::TileMode::kRenderTarget, dccSize)) throw std::runtime_error("kyty presentation: invalid DCC footprint for the display buffer");
        info.metadata.range = {buffer.dccAddress, dccSize.size};
        info.metadata.kind = Libs::Graphics::ImageMetadataKind::Dcc;
        info.metadata.compression = compression;
    }
    Libs::Graphics::ImageOps::Validate(info);
    if (!Libs::Graphics::IsSupportedVideoOutFormat(info)) throw std::runtime_error("kyty presentation: unsupported normalized display format");
    return info;
}

void PrepareSource(PendingFlip& flip, Libs::Graphics::Presenter& presenter, Libs::Graphics::CommandBuffer& buffer) {
    flip.source = flip.request->Source();
    if (!flip.source.has_value()) return;
    const auto info = DisplayImageInfo(*flip.source);
    flip.frame = &presenter.PrepareFrame(buffer, info, flip.source->pixelFormat, false);
}

std::uint64_t DropFlip(std::uint32_t handle, int index) {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    const auto id = state.nextFlip++;
    state.flips.emplace(id, PendingFlip{nullptr, handle, index, std::nullopt, nullptr, true});
    return id;
}

std::optional<ReadyFlip> TakeReadyFlip(const void* context, Libs::Graphics::Presenter& presenter) {
    std::optional<ReadyFlip> taken;
    std::vector<Libs::Graphics::Presenter::Frame*> abandoned;
    {
        auto& state = State();
        std::lock_guard lock(state.mutex);
        for (auto flip = state.ready.begin(); flip != state.ready.end();) {
            if (flip->request.expired()) {
                if (flip->frame != nullptr) abandoned.push_back(flip->frame);
                flip = state.ready.erase(flip);
            } else if (!taken.has_value() && flip->owner == context) {
                taken = std::move(*flip);
                flip = state.ready.erase(flip);
            } else {
                ++flip;
            }
        }
    }
    for (auto* frame : abandoned) presenter.Discard(*frame);
    if (!taken.has_value()) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            std::fprintf(stderr, "[kyty-videoout] a presentation matches no flip request of the Kyty GPU: the frame is prepared now, out of the flip's command buffer\n");
            std::fflush(stderr);
        }
    }
    return taken;
}

}

void SubmitCpuFlip(Libs::Graphics::GuestGpu& gpu, const AgcDriver::FlipInfo& info) {
    const auto output = Entry::VideoOutput(info.handle);
    if (output == nullptr) throw std::runtime_error("kyty presentation: CPU flip on unregistered output " + std::to_string(info.handle));
    output->WaitForFlipRoom();
    PendingFlip pending{output->Reserve(info), info.handle, info.index, std::nullopt, nullptr};
    if (pending.request == nullptr) throw std::runtime_error("kyty presentation: the video output refused a CPU flip reservation");
    std::uint64_t id = 0;
    {
        auto& state = State();
        std::lock_guard lock(state.mutex);
        id = state.nextFlip++;
        state.flips.emplace(id, std::move(pending));
    }
    gpu.SubmitFlipPreparation(id);
}

void WaitRendering(std::uint32_t handle, std::uint32_t index) {
    const auto output = Entry::VideoOutput(handle);
    if (output == nullptr) throw std::runtime_error("kyty presentation: rendering wait on unregistered output " + std::to_string(handle));
    const auto wait = output->CaptureRenderingWait(index);
    WaitRunningUrgentCommands([wait] { wait->Wait(); });
}

Libs::Graphics::ImageInfo DisplayBufferImage(const AgcDriver::DisplayBuffer& buffer) {
    return DisplayImageInfo(buffer);
}

}

namespace Libs::VideoOut {

struct VideoOutDriver::Impl {
    explicit Impl(Graphics::Presenter& owner) : presenter(owner) {}
    Graphics::Presenter& presenter;
};

VideoOutDriver::VideoOutDriver(uint32_t, uint32_t, Graphics::Presenter& presenter) : m_impl(std::make_unique<Impl>(presenter)) {}

VideoOutDriver::~VideoOutDriver() = default;

VideoOutDriver::Impl& VideoOutDriver::State() noexcept {
    return *m_impl;
}

int VideoOutDriver::SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index, int flip_mode, int64_t flip_arg, uint64_t& request_id) {
    try {
        const auto output = KytyShim::Entry::VideoOutput(static_cast<std::uint32_t>(handle));
        if (output == nullptr) return VIDEO_OUT_ERROR_INVALID_HANDLE;
        auto request = output->TryReserve(AgcDriver::FlipInfo{static_cast<std::uint32_t>(handle), index, static_cast<std::uint32_t>(flip_mode), flip_arg});
        if (request == nullptr) return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
        KytyShim::Present::PendingFlip pending{std::move(request), static_cast<std::uint32_t>(handle), index, std::nullopt, nullptr};
        KytyShim::Present::PrepareSource(pending, State().presenter, buffer);
        auto& state = KytyShim::Present::State();
        std::lock_guard lock(state.mutex);
        request_id = state.nextFlip++;
        state.flips.emplace(request_id, std::move(pending));
        return OK;
    } catch (const ProcessShutdown&) {
        request_id = KytyShim::Present::DropFlip(static_cast<std::uint32_t>(handle), index);
        return OK;
    } catch (...) {
        KytyShim::Present::Stop("GPU flip submission", std::current_exception());
    }
}

void VideoOutDriver::PrepareFlip(uint64_t request_id, Graphics::CommandBuffer& buffer) {
    try {
        KytyShim::Present::PendingFlip pending;
        auto& state = KytyShim::Present::State();
        {
            std::lock_guard lock(state.mutex);
            const auto found = state.flips.find(request_id);
            if (found == state.flips.end()) throw std::runtime_error("flip preparation for an unknown request " + std::to_string(request_id));
            if (found->second.frame != nullptr || found->second.source.has_value()) throw std::runtime_error("flip request " + std::to_string(request_id) + " prepared twice");
            pending.request = found->second.request;
        }
        KytyShim::Present::PrepareSource(pending, State().presenter, buffer);
        std::lock_guard lock(state.mutex);
        auto& flip = state.flips.at(request_id);
        flip.source = std::move(pending.source);
        flip.frame = pending.frame;
    } catch (...) {
        KytyShim::Present::Stop("CPU flip preparation", std::current_exception());
    }
}

void VideoOutDriver::CompleteFlip(uint64_t request_id) {
    try {
        std::shared_ptr<AgcDriver::IFlipRequest> request;
        {
            auto& state = KytyShim::Present::State();
            std::lock_guard lock(state.mutex);
            const auto found = state.flips.find(request_id);
            if (found == state.flips.end()) throw std::runtime_error("flip completion for an unknown request " + std::to_string(request_id));
            auto& pending = found->second;
            if (pending.dropped) {
                state.flips.erase(found);
                return;
            }
            request = std::move(pending.request);
            state.ready.push_back({request, dynamic_cast<const void*>(request.get()), pending.handle, pending.index, std::move(pending.source), pending.frame});
            state.flips.erase(found);
        }
        request->GpuReady(std::make_shared<AgcDriver::FrameTiming>(request_id));
    } catch (const ProcessShutdown&) {
    } catch (...) {
        KytyShim::Present::Stop("GPU flip completion", std::current_exception());
    }
}

void VideoOutDriver::WaitForSubmitSlot(int handle) {
    try {
        const auto output = KytyShim::Entry::VideoOutput(static_cast<std::uint32_t>(handle));
        if (output == nullptr) throw std::runtime_error("flip slot wait on unregistered output " + std::to_string(handle));
        KytyShim::Present::WaitRunningUrgentCommands([output] { output->WaitForFlipRoom(); });
    } catch (const ProcessShutdown&) {
    } catch (...) {
        KytyShim::Present::Stop("flip slot wait", std::current_exception());
    }
}

void VideoOutDriver::WaitFlipDone(const uint64_t*) {
    std::fprintf(stderr, "[kyty-videoout] Kyty's wait-flip-done packet reached the kyty backend, but AnyPS5's libSceAgc emits the rendering wait packet instead\n");
    std::fflush(stderr);
    LibcTerminate_nid_no_patch(1);
}

VideoOutDriver& VideoOutInit(uint32_t width, uint32_t height, Graphics::Presenter& presenter) {
    auto& state = KytyShim::Present::State();
    std::lock_guard lock(state.mutex);
    EXIT_IF(state.driver != nullptr);
    state.driver = std::make_unique<VideoOutDriver>(width, height, presenter);
    return *state.driver;
}

void VideoOutShutdown() {
    std::unique_ptr<VideoOutDriver> driver;
    {
        auto& state = KytyShim::Present::State();
        std::lock_guard lock(state.mutex);
        driver = std::move(state.driver);
        state.flips.clear();
        state.ready.clear();
    }
}

}

namespace Libs::Graphics {

Presenter& WindowInit(uint32_t width, uint32_t height) {
    auto& state = KytyShim::Present::State();
    {
        std::lock_guard lock(state.mutex);
        EXIT_IF(state.window != nullptr);
    }
    auto host = std::make_unique<WindowContext>();
    host->graphic_ctx.screen_width = width;
    host->graphic_ctx.screen_height = height;
    host->graphic_ctx.pipeline_cache_persistent =
        !KytyShim::Replay::Tracking() || std::getenv("APS5_HEADLESS_PIPELINE_CACHE") != nullptr;
    host->host_get_instance_proc_addr = KytyShim::Present::LoadVulkanLoader();
    host->host_extensions = KytyShim::Present::SurfaceInstanceExtensions(host->host_get_instance_proc_addr);
    host->host_create_surface = &KytyShim::Present::CreateHostSurface;
    host->CreateVulkan();
    auto& presenter = *host->presenter;
    std::lock_guard lock(state.mutex);
    state.window = std::move(host);
    return presenter;
}

void WindowShutdown() {
    std::unique_ptr<WindowContext> window;
    {
        auto& state = KytyShim::Present::State();
        std::lock_guard lock(state.mutex);
        window = std::move(state.window);
        state.windowHandle = nullptr;
    }
}

struct SystemOverlay::Impl {
    vk::Extent2D extent;
    AgcDriver::OverlayLayout layout;
};

SystemOverlay::SystemOverlay(GraphicContext&): m_impl(std::make_unique<Impl>()) {}

SystemOverlay::~SystemOverlay() = default;

bool SystemOverlay::PrepareFrame(vk::Extent2D extent, vk::Format format, uint32_t) {
    try {
        auto layout = AgcDriver::SystemOverlayLayout(VkExtent2D{extent.width, extent.height});
        if (!layout.has_value()) return false;
        if (format != vk::Format::eB8G8R8A8Unorm && format != vk::Format::eR8G8B8A8Unorm) throw std::runtime_error("kyty presentation: system dialogs and the system keyboard are drawn only into an 8-bit SDR swapchain");
        m_impl->extent = extent;
        m_impl->layout = std::move(*layout);
        return true;
    } catch (...) {
        KytyShim::Present::Stop("system overlay", std::current_exception());
    }
}

void SystemOverlay::Record(vk::CommandBuffer command, vk::ImageView target) {
    try {
        vk::RenderingAttachmentInfo attachment{};
        attachment.imageView = target;
        attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        attachment.loadOp = vk::AttachmentLoadOp::eLoad;
        attachment.storeOp = vk::AttachmentStoreOp::eStore;
        vk::RenderingInfo rendering{};
        rendering.renderArea.extent = m_impl->extent;
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &attachment;
        command.beginRendering(&rendering);
        std::vector<vk::ClearRect> run;
        std::size_t runColor = AgcDriver::OverlayColorCount;
        const auto flush = [&] {
            if (run.empty()) return;
            const auto& color = KytyShim::Present::OverlayBasePalette[runColor];
            vk::ClearAttachment clear{};
            clear.aspectMask = vk::ImageAspectFlagBits::eColor;
            clear.colorAttachment = 0;
            clear.clearValue.color = vk::ClearColorValue(std::array<float, 4>{color[0] * color[3], color[1] * color[3], color[2] * color[3], 1.0f});
            command.clearAttachments(1, &clear, static_cast<uint32_t>(run.size()), run.data());
            run.clear();
        };
        const auto width = static_cast<std::int64_t>(m_impl->extent.width);
        const auto height = static_cast<std::int64_t>(m_impl->extent.height);
        for (const auto& rect : m_impl->layout.rects) {
            const auto left = std::clamp<std::int64_t>(rect.x, 0, width);
            const auto top = std::clamp<std::int64_t>(rect.y, 0, height);
            const auto right = std::clamp<std::int64_t>(static_cast<std::int64_t>(rect.x) + rect.width, 0, width);
            const auto bottom = std::clamp<std::int64_t>(static_cast<std::int64_t>(rect.y) + rect.height, 0, height);
            if (right <= left || bottom <= top) continue;
            const std::size_t color = rect.color < AgcDriver::OverlayColorCount ? rect.color : 0;
            if (color != runColor) {
                flush();
                runColor = color;
            }
            run.push_back(vk::ClearRect{vk::Rect2D{{static_cast<int32_t>(left), static_cast<int32_t>(top)}, {static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top)}}, 0, 1});
        }
        flush();
        command.endRendering();
    } catch (...) {
        KytyShim::Present::Stop("system overlay", std::current_exception());
    }
}

void SystemOverlay::ReleaseVulkan() {}

SystemOverlayVisualState GetSystemOverlayVisualState() noexcept {
    return {AgcDriver::SystemOverlayShown(), 0};
}

void ShutdownSystemOverlayInput() {}

bool PerformanceOverlayEnabled() noexcept {
    return false;
}

bool ManifestProgressVisible() noexcept {
    return false;
}

void ShowManifestProgress(const ManifestProgress&) {}

void HideManifestProgress() {}

void ShowPrecompileNotice(const PrecompileNotice& notice) {
    static std::atomic<bool> told{false};
    if (notice.visible && !told.exchange(true)) {
        std::fprintf(stderr, "[kyty-videoout] the pipeline precompile notice is not drawn with ANYPS5_GPU_BACKEND=kyty: the game waits for its pipelines without it\n");
        std::fflush(stderr);
    }
}

void HidePrecompileNotice() {}

PrecompileAnswer TakePrecompileAnswer() noexcept {
    return PrecompileAnswer::None;
}

bool PrecompileNoticeVisible() noexcept {
    return false;
}

}

namespace Libs::Vr {

void OpenXrInit() {}

void OpenXrAppendExtensions(std::vector<const char*>&, const std::vector<vk::ExtensionProperties>&, bool) {}

bool OpenXrPhysicalDevice(vk::Instance, vk::PhysicalDevice&) {
    return false;
}

void OpenXrStartSession(Graphics::GraphicContext&) {}

void OpenXrShutdown() {}

void OpenXrQueueEyeSource(Graphics::CommandBuffer&, const Graphics::VulkanImage&) {
    EXIT("OpenXR headsets are not built with ANYPS5_GPU_BACKEND=kyty\n");
}

}

extern "C" void AgcDriverAttachWindow_nid_postfix(const AgcDriver::PresentationWindow& window) {
    if (window.context == nullptr || window.createSurface == nullptr || window.getDrawableSize == nullptr) throw std::runtime_error("kyty presentation: incomplete presentation window");
    KytyShim::Entry::JoinPrestart(true);
    if (!KytyShim::Entry::Renderer().GetGpu().Pipelined()) throw std::runtime_error("kyty presentation: the Kyty GPU must run pipelined (GPU pipeline stages must include the command processor)");
    auto& state = KytyShim::Present::State();
    Libs::Graphics::WindowContext* host = nullptr;
    {
        std::lock_guard lock(state.mutex);
        if (state.window == nullptr) throw std::runtime_error("kyty presentation: the Kyty GPU started without its window context");
        if (state.attaching || state.windowHandle != nullptr || state.window->surface != nullptr) throw std::runtime_error("kyty presentation: the Kyty presenter is already bound to a window and cannot move to another one");
        for (const char* needed : window.extensions) {
            bool enabled = false;
            for (const char* name : state.window->host_extensions) enabled = enabled || std::strcmp(name, needed) == 0;
            if (!enabled) throw std::runtime_error(std::string("kyty presentation: the window needs the Vulkan instance extension ") + needed + ", which the Kyty instance did not enable");
        }
        state.createSurface = window.createSurface;
        state.attaching = true;
        host = state.window.get();
    }
    host->window = static_cast<SDL_Window*>(window.context);
    host->host_context = window.context;
    host->RecreateSurface();
    vk::Bool32 supported = VK_FALSE;
    if (host->graphic_ctx.physical_device.getSurfaceSupportKHR(host->graphic_ctx.queue_family, host->surface, &supported) != vk::Result::eSuccess || supported != VK_TRUE) throw std::runtime_error("kyty presentation: the Vulkan queue the Kyty renderer selected cannot present to this window");
    host->RefreshSurfaceCapabilities();
    if (!(host->surface_capabilities.capabilities.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)) throw std::runtime_error("kyty presentation: the window surface cannot be a blit destination");
    KytyShim::Present::UpdateDrawableSize(*host, window);
    host->presenter->AttachSurface();
    std::lock_guard lock(state.mutex);
    state.windowHandle = window.context;
    state.attaching = false;
}

namespace KytyShim::Present {
namespace {

constexpr int OverlayBus = 1;

struct GuestImage {
    Libs::Graphics::ImageInfo info;
    std::uint64_t pixelFormat = 0;
};

std::optional<GuestImage> DescribeGuestImage(const AgcDriver::DisplayBuffer* buffer) {
    if (buffer == nullptr) return std::nullopt;
    return GuestImage{DisplayImageInfo(*buffer), buffer->pixelFormat};
}

struct PreparedFrames {
    std::binary_semaphore done{0};
    std::uint64_t tick = 0;
    Libs::Graphics::Presenter::Frame* main = nullptr;
    Libs::Graphics::Presenter::Frame* overlay = nullptr;
};

std::shared_ptr<PreparedFrames> PrepareGuestFrames(Libs::Graphics::RenderContext& renderer, Libs::Graphics::Presenter& presenter, const std::optional<GuestImage>& main, const std::optional<GuestImage>& overlay) {
    auto prepared = std::make_shared<PreparedFrames>();
    renderer.GetGpu().SendUrgentCommandSync([&renderer, &presenter, main, overlay, prepared] {
        auto& scheduler = renderer.GetCommandScheduler();
        if (main.has_value()) prepared->main = &presenter.PrepareFrame(scheduler.Current(), main->info, main->pixelFormat, false);
        if (overlay.has_value()) prepared->overlay = &presenter.PrepareFrame(scheduler.Current(), overlay->info, overlay->pixelFormat, false);
        if (main.has_value()) scheduler.DeferHostOperation([prepared] { prepared->done.release(); }, true);
        prepared->tick = scheduler.CurrentTick();
        scheduler.Flush();
    });
    if (main.has_value()) prepared->done.acquire();
    else renderer.GetCommandScheduler().GetMasterSemaphore().Wait(prepared->tick);
    return prepared;
}

void PresentLayers(Libs::Graphics::Presenter& presenter, Libs::Graphics::Presenter::Frame& main, Libs::Graphics::Presenter::Frame* overlay) {
    if (overlay == nullptr) {
        presenter.ClearLayer(OverlayBus);
        presenter.Present(main);
        return;
    }
    const std::array<Libs::Graphics::Presenter::Layer, 2> layers{{{&main, 0, false}, {overlay, OverlayBus, true}}};
    presenter.Present(std::span(layers));
}

void PresentGuestLayers(Libs::Graphics::WindowContext& host, Libs::Graphics::Presenter& presenter, Libs::Graphics::Presenter::Frame& main, Libs::Graphics::Presenter::Frame* overlay) {
    const auto before = host.loop.presented_frames.load(std::memory_order_relaxed);
    PresentLayers(presenter, main, overlay);
    if (host.loop.presented_frames.load(std::memory_order_relaxed) != before) LibcRunReportFramePresented_nid_no_patch();
}

}
}

extern "C" void AgcDriverPresentBuffer_nid_postfix(const AgcDriver::PresentationWindow& window, const AgcDriver::DisplayBuffer& buffer, const AgcDriver::DisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
    if (gpuReady == nullptr || context == nullptr) throw std::runtime_error("kyty presentation: missing GPU completion callback");
    auto& host = KytyShim::Present::AttachedWindow(window);
    KytyShim::Present::UpdateDrawableSize(host, window);
    auto& presenter = *host.presenter;
    auto& renderer = *host.render_context;
    const auto overlayImage = KytyShim::Present::DescribeGuestImage(plane);
    const auto flip = KytyShim::Present::TakeReadyFlip(context, presenter);
    if (flip.has_value() && flip->frame != nullptr && flip->source == buffer) {
        const auto overlay = overlayImage.has_value() ? KytyShim::Present::PrepareGuestFrames(renderer, presenter, std::nullopt, overlayImage) : nullptr;
        KytyShim::Present::PresentGuestLayers(host, presenter, *flip->frame, overlay != nullptr ? overlay->overlay : nullptr);
        gpuReady(context);
        return;
    }
    if (flip.has_value() && flip->frame != nullptr) presenter.Discard(*flip->frame);
    const auto prepared = KytyShim::Present::PrepareGuestFrames(renderer, presenter, KytyShim::Present::DescribeGuestImage(&buffer), overlayImage);
    KytyShim::Present::PresentGuestLayers(host, presenter, *prepared->main, prepared->overlay);
    gpuReady(context);
}

extern "C" void AgcDriverPresentClear_nid_postfix(const AgcDriver::PresentationWindow& window, bool opaque, const AgcDriver::DisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
    if (gpuReady == nullptr || context == nullptr) throw std::runtime_error("kyty presentation: missing GPU completion callback");
    if (window.width == 0 || window.height == 0) throw std::runtime_error("kyty presentation: invalid output extent");
    auto& host = KytyShim::Present::AttachedWindow(window);
    KytyShim::Present::UpdateDrawableSize(host, window);
    auto& presenter = *host.presenter;
    const auto overlayImage = KytyShim::Present::DescribeGuestImage(plane);
    if (const auto flip = KytyShim::Present::TakeReadyFlip(context, presenter); flip.has_value() && flip->frame != nullptr) presenter.Discard(*flip->frame);
    auto& blank = presenter.PrepareBlankFrame(window.width, window.height, opaque);
    const auto overlay = overlayImage.has_value() ? KytyShim::Present::PrepareGuestFrames(*host.render_context, presenter, std::nullopt, overlayImage) : nullptr;
    KytyShim::Present::PresentLayers(presenter, blank, overlay != nullptr ? overlay->overlay : nullptr);
    gpuReady(context);
}

extern "C" void AgcDriverReleaseWindow_nid_postfix(void* window) {
    auto& state = KytyShim::Present::State();
    std::lock_guard lock(state.mutex);
    if (window != nullptr && state.windowHandle == window) state.windowHandle = nullptr;
}
