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
#include "graphics/presentation/window/windowInternal.h"
#include "libs/errno.h"

#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libc/include/Shutdown.hpp"

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace KytyShim {

namespace {

constexpr std::uint32_t OutputHandleLimit = 8;
constexpr int BufferLimit = 16;

struct TrackedFlip {
    std::shared_ptr<AgcDriver::IFlipRequest> request;
    std::uint32_t handle = 0;
    int index = 0;
};

struct PresentationState {
    std::mutex mutex;
    std::condition_variable attached;
    std::map<std::uint32_t, std::shared_ptr<AgcDriver::IVideoOutput>> outputs;
    std::array<std::array<std::uint64_t, BufferLimit>, OutputHandleLimit> labels{};
    std::map<std::uint64_t, TrackedFlip> flips;
    std::set<std::uint64_t> cancelled;
    std::uint64_t nextFlip = 1;
    std::unique_ptr<Libs::Graphics::WindowContext> window;
    std::unique_ptr<Libs::VideoOut::VideoOutDriver> driver;
    void* windowHandle = nullptr;
    VkSurfaceKHR (*createSurface)(void* context, VkInstance instance) = nullptr;
    bool released = false;
};

PresentationState& State() {
    static auto* state = new PresentationState;
    return *state;
}

std::shared_ptr<AgcDriver::IVideoOutput> FindOutput(std::uint32_t handle) {
    auto& state = State();
    std::lock_guard lock(state.mutex);
    const auto found = state.outputs.find(handle);
    return found != state.outputs.end() ? found->second : nullptr;
}

void FailOutputs(std::exception_ptr error) noexcept {
    std::map<std::uint32_t, std::shared_ptr<AgcDriver::IVideoOutput>> outputs;
    {
        auto& state = State();
        std::lock_guard lock(state.mutex);
        outputs = state.outputs;
    }
    for (const auto& [handle, output] : outputs) output->Fail(error);
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
        LibcTerminate_nid_no_patch(1);
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "[kyty-videoout] %s: %s\n", where, failure.what());
    } catch (...) {
        std::fprintf(stderr, "[kyty-videoout] %s: unknown failure\n", where);
    }
    std::fflush(stderr);
    FailOutputs(error);
    LibcTerminate_nid_no_patch(1);
}

template <typename Blocking>
void WaitRunningUrgentCommands(Blocking&& blocking) {
    auto waiting = std::async(std::launch::async, std::forward<Blocking>(blocking));
    while (waiting.wait_for(std::chrono::microseconds(200)) != std::future_status::ready) {
        Libs::Graphics::GuestGpu::RunUrgentCommandsWhileWaiting();
    }
    waiting.get();
}

PFN_vkGetInstanceProcAddr LoadVulkanLoader() {
    static void* const library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) throw std::runtime_error(std::string("kyty presentation: cannot load libvulkan.so.1: ") + dlerror());
    const auto proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
    if (proc == nullptr) throw std::runtime_error("kyty presentation: libvulkan.so.1 has no vkGetInstanceProcAddr");
    return proc;
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
    if (state.window == nullptr || state.windowHandle != window.context) throw std::runtime_error("kyty presentation: the presentation window is not attached");
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
    if (buffer.dccAddress != 0 || buffer.dccClearColor != 0) throw std::runtime_error("kyty presentation: DCC-compressed display buffers are not supported (AgcDriver::DisplayBuffer carries no DCC control word)");
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
    Libs::Graphics::ImageOps::Validate(info);
    if (!Libs::Graphics::IsSupportedVideoOutFormat(info)) throw std::runtime_error("kyty presentation: unsupported normalized display format");
    return info;
}

void PresentFrame(Libs::Graphics::WindowContext& host, Libs::Graphics::Presenter::Frame& frame, void (*gpuReady)(void*), void* context) {
    host.presenter->Present(frame);
    gpuReady(context);
}

}

Libs::Graphics::RenderContext& Renderer() {
    auto& state = State();
    std::unique_lock lock(state.mutex);
    state.attached.wait(lock, [&] { return state.driver != nullptr || state.released; });
    if (state.released) throw std::runtime_error("kyty presentation: the renderer was released with its window");
    return *state.window->render_context;
}

std::shared_ptr<AgcDriver::IVideoOutput> VideoOutput(std::uint32_t handle) {
    return FindOutput(handle);
}

std::uint64_t TrackFlip(std::uint32_t handle, int index, std::shared_ptr<AgcDriver::IFlipRequest> request) {
    if (request == nullptr) throw std::runtime_error("kyty presentation: null flip request");
    auto& state = State();
    std::lock_guard lock(state.mutex);
    const auto id = state.nextFlip++;
    state.flips.emplace(id, TrackedFlip{std::move(request), handle, index});
    return id;
}

void ShutdownPresentation() {
    std::unique_ptr<Libs::Graphics::WindowContext> window;
    std::unique_ptr<Libs::VideoOut::VideoOutDriver> driver;
    {
        auto& state = State();
        std::lock_guard lock(state.mutex);
        if (state.window == nullptr) return;
        state.released = true;
        state.windowHandle = nullptr;
        window = std::move(state.window);
        driver = std::move(state.driver);
        state.flips.clear();
        state.cancelled.clear();
        state.attached.notify_all();
    }
    window->render_context->ShutdownGpu();
    driver.reset();
    window.reset();
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

int VideoOutDriver::SubmitFlipFromGpu(Graphics::CommandBuffer&, int handle, int index, int flip_mode, int64_t flip_arg, uint64_t& request_id) {
    try {
        const auto output = KytyShim::FindOutput(static_cast<std::uint32_t>(handle));
        if (output == nullptr) return VIDEO_OUT_ERROR_INVALID_HANDLE;
        auto request = output->TryReserve(AgcDriver::FlipInfo{static_cast<std::uint32_t>(handle), index, static_cast<std::uint32_t>(flip_mode), flip_arg});
        if (request == nullptr) return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
        request_id = KytyShim::TrackFlip(static_cast<std::uint32_t>(handle), index, std::move(request));
        return OK;
    } catch (...) {
        KytyShim::Stop("GPU flip submission", std::current_exception());
    }
}

void VideoOutDriver::PrepareFlip(uint64_t request_id, Graphics::CommandBuffer&) {
    auto& state = KytyShim::State();
    std::lock_guard lock(state.mutex);
    if (state.flips.contains(request_id) || state.cancelled.contains(request_id) || state.released) return;
    std::fprintf(stderr, "[kyty-videoout] flip preparation for an unknown request %llu\n", static_cast<unsigned long long>(request_id));
    std::fflush(stderr);
    LibcTerminate_nid_no_patch(1);
}

void VideoOutDriver::CompleteFlip(uint64_t request_id) {
    try {
        KytyShim::TrackedFlip flip;
        {
            auto& state = KytyShim::State();
            std::lock_guard lock(state.mutex);
            const auto found = state.flips.find(request_id);
            if (found == state.flips.end()) {
                if (state.cancelled.erase(request_id) != 0 || state.released) return;
                throw std::runtime_error("flip completion for an unknown request " + std::to_string(request_id));
            }
            flip = std::move(found->second);
            state.flips.erase(found);
        }
        flip.request->GpuReady(std::make_shared<AgcDriver::FrameTiming>(request_id));
    } catch (const ProcessShutdown&) {
    } catch (...) {
        KytyShim::Stop("GPU flip completion", std::current_exception());
    }
}

void VideoOutDriver::WaitForSubmitSlot(int handle) {
    try {
        const auto output = KytyShim::FindOutput(static_cast<std::uint32_t>(handle));
        if (output == nullptr) throw std::runtime_error("flip slot wait on unregistered output " + std::to_string(handle));
        KytyShim::WaitRunningUrgentCommands([output] { output->WaitForFlipRoom(); });
    } catch (...) {
        KytyShim::Stop("flip slot wait", std::current_exception());
    }
}

void VideoOutDriver::WaitFlipDone(const uint64_t* label) {
    try {
        auto& state = KytyShim::State();
        std::shared_ptr<AgcDriver::IVideoOutput> output;
        std::uint32_t index = 0;
        {
            std::lock_guard lock(state.mutex);
            for (std::uint32_t handle = 0; handle < KytyShim::OutputHandleLimit && output == nullptr; ++handle) {
                const auto& labels = state.labels[handle];
                const auto* first = labels.data();
                if (label < first || label >= first + labels.size()) continue;
                const auto found = state.outputs.find(handle);
                if (found == state.outputs.end()) throw std::runtime_error("flip wait on unregistered output " + std::to_string(handle));
                output = found->second;
                index = static_cast<std::uint32_t>(label - first);
            }
            if (output == nullptr) throw std::runtime_error("flip wait on a label the video outputs do not own");
        }
        const auto wait = output->CaptureRenderingWait(index);
        KytyShim::WaitRunningUrgentCommands([wait] { wait->Wait(); });
    } catch (...) {
        KytyShim::Stop("flip done wait", std::current_exception());
    }
}

uint64_t* VideoOutGetBufferLabels(int handle, int* bus) {
    auto& state = KytyShim::State();
    std::lock_guard lock(state.mutex);
    if (handle <= 0 || static_cast<std::uint32_t>(handle) >= KytyShim::OutputHandleLimit || !state.outputs.contains(static_cast<std::uint32_t>(handle))) return nullptr;
    *bus = handle - 1;
    return state.labels[static_cast<std::size_t>(handle)].data();
}

}

namespace Libs::Graphics {

struct SystemOverlay::Impl {};

SystemOverlay::SystemOverlay(GraphicContext&) {
    EXIT("the system overlay is not built with ANYPS5_GPU_BACKEND=kyty\n");
}

SystemOverlay::~SystemOverlay() = default;

bool SystemOverlay::PrepareFrame(vk::Extent2D, vk::Format, uint32_t) {
    EXIT("the system overlay is not built with ANYPS5_GPU_BACKEND=kyty\n");
}

void SystemOverlay::Record(vk::CommandBuffer, vk::ImageView) {
    EXIT("the system overlay is not built with ANYPS5_GPU_BACKEND=kyty\n");
}

void SystemOverlay::ReleaseVulkan() {}

SystemOverlayVisualState GetSystemOverlayVisualState() noexcept {
    return {false, 0};
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

extern "C" void AgcDriverRegisterVideoOutput_nid_postfix(std::uint32_t handle, const std::shared_ptr<AgcDriver::IVideoOutput>& output) {
    if (output == nullptr) throw std::runtime_error("kyty presentation: null video output");
    if (handle == 0 || handle >= KytyShim::OutputHandleLimit) throw std::runtime_error("kyty presentation: video output handle " + std::to_string(handle) + " is out of range");
    auto& state = KytyShim::State();
    std::lock_guard lock(state.mutex);
    if (!state.outputs.emplace(handle, output).second) throw std::runtime_error("kyty presentation: video output already registered");
    state.labels[handle].fill(0);
}

extern "C" void AgcDriverUnregisterVideoOutput_nid_postfix(std::uint32_t handle, const std::shared_ptr<AgcDriver::IVideoOutput>& output) {
    auto& state = KytyShim::State();
    std::lock_guard lock(state.mutex);
    const auto found = state.outputs.find(handle);
    if (found == state.outputs.end() || found->second != output) throw std::runtime_error("kyty presentation: video output registration mismatch");
    for (auto it = state.flips.begin(); it != state.flips.end();) {
        if (it->second.handle != handle) {
            ++it;
            continue;
        }
        state.cancelled.insert(it->first);
        it = state.flips.erase(it);
    }
    state.outputs.erase(found);
}

extern "C" void AgcDriverAttachWindow_nid_postfix(const AgcDriver::PresentationWindow& window) {
    auto& state = KytyShim::State();
    {
        std::lock_guard lock(state.mutex);
        if (state.window != nullptr) throw std::runtime_error("kyty presentation: a window is already attached; the Kyty renderer cannot move to another window");
        if (state.released) throw std::runtime_error("kyty presentation: the Kyty renderer cannot be created again after its window was released");
        if (window.context == nullptr || window.createSurface == nullptr || window.getDrawableSize == nullptr) throw std::runtime_error("kyty presentation: incomplete presentation window");
        state.createSurface = window.createSurface;
    }
    std::uint32_t width = window.width;
    std::uint32_t height = window.height;
    window.getDrawableSize(window.context, &width, &height);
    if (width == 0 || height == 0) {
        width = window.width;
        height = window.height;
    }
    auto host = std::make_unique<Libs::Graphics::WindowContext>();
    host->graphic_ctx.screen_width = width;
    host->graphic_ctx.screen_height = height;
    host->window = static_cast<SDL_Window*>(window.context);
    host->host_context = window.context;
    host->host_extensions.assign(window.extensions.begin(), window.extensions.end());
    host->host_create_surface = &KytyShim::CreateHostSurface;
    host->host_get_instance_proc_addr = KytyShim::LoadVulkanLoader();
    Common::InitializeThreads();
    host->CreateVulkan();
    auto driver = std::make_unique<Libs::VideoOut::VideoOutDriver>(window.width, window.height, *host->presenter);
    host->render_context->InitializeGpu(driver.get());
    if (!host->render_context->GetGpu().Pipelined()) throw std::runtime_error("kyty presentation: the GPU must run pipelined (KYTY GPU pipeline stages must include the command processor)");
    std::lock_guard lock(state.mutex);
    state.window = std::move(host);
    state.driver = std::move(driver);
    state.windowHandle = window.context;
    state.attached.notify_all();
}

extern "C" void AgcDriverPresentBuffer_nid_postfix(const AgcDriver::PresentationWindow& window, const AgcDriver::DisplayBuffer& buffer, const AgcDriver::DisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
    if (gpuReady == nullptr || context == nullptr) throw std::runtime_error("kyty presentation: missing GPU completion callback");
    if (plane != nullptr) throw std::runtime_error("kyty presentation: the overlay video plane is not presented with ANYPS5_GPU_BACKEND=kyty");
    auto& host = KytyShim::AttachedWindow(window);
    KytyShim::UpdateDrawableSize(host, window);
    const auto info = KytyShim::DisplayImageInfo(buffer);
    const bool hdr = Libs::Graphics::IsHdrVideoOutPixelFormat(buffer.pixelFormat);
    struct Prepared {
        std::binary_semaphore done{0};
        Libs::Graphics::Presenter::Frame* frame = nullptr;
    };
    auto prepared = std::make_shared<Prepared>();
    auto& renderer = *host.render_context;
    auto& presenter = *host.presenter;
    renderer.GetGpu().SendUrgentCommandSync([&renderer, &presenter, info, hdr, prepared] {
        auto& scheduler = renderer.GetCommandScheduler();
        prepared->frame = &presenter.PrepareFrame(scheduler.Current(), info, hdr, false);
        scheduler.DeferHostOperation([prepared] { prepared->done.release(); }, true);
        scheduler.Flush();
    });
    prepared->done.acquire();
    KytyShim::PresentFrame(host, *prepared->frame, gpuReady, context);
}

extern "C" void AgcDriverPresentClear_nid_postfix(const AgcDriver::PresentationWindow& window, bool opaque, const AgcDriver::DisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
    if (gpuReady == nullptr || context == nullptr) throw std::runtime_error("kyty presentation: missing GPU completion callback");
    if (plane != nullptr) throw std::runtime_error("kyty presentation: the overlay video plane is not presented with ANYPS5_GPU_BACKEND=kyty");
    if (window.width == 0 || window.height == 0) throw std::runtime_error("kyty presentation: invalid output extent");
    auto& host = KytyShim::AttachedWindow(window);
    KytyShim::UpdateDrawableSize(host, window);
    auto& frame = host.presenter->PrepareBlankFrame(window.width, window.height, opaque);
    KytyShim::PresentFrame(host, frame, gpuReady, context);
}

extern "C" void AgcDriverReleaseWindow_nid_postfix(void* window) {
    {
        auto& state = KytyShim::State();
        std::lock_guard lock(state.mutex);
        if (state.window == nullptr || state.windowHandle != window) return;
    }
    KytyShim::ShutdownPresentation();
}

extern "C" void AgcDriverTogglePerformanceOverlay_nid_postfix() {
    static std::atomic<bool> told{false};
    if (!told.exchange(true)) {
        std::fprintf(stderr, "[kyty-videoout] the performance overlay is not built with ANYPS5_GPU_BACKEND=kyty\n");
        std::fflush(stderr);
    }
}

extern "C" void AgcDriverSetSystemOverlay_nid_postfix(const ImeKeyboardView* view) {
    if (view != nullptr) throw std::runtime_error("kyty presentation: the on-screen keyboard is not drawn with ANYPS5_GPU_BACKEND=kyty");
}

extern "C" void AgcDriverSetSystemDialog_nid_postfix(const ModalDialogView* view) {
    if (view != nullptr) throw std::runtime_error("kyty presentation: system dialogs are not drawn with ANYPS5_GPU_BACKEND=kyty");
}
