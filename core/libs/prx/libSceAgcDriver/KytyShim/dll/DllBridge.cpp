#include "prx/libSceAgcDriver/KytyShim/dll/Trampoline.hpp"
#include "prx/libSceAgcDriver/KytyShim/include/KytyDll.h"

#include "SceShaders.hpp"
#include "SceTypes.hpp"
#include "libs/agc.h"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/KeyboardOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "prx/libkernel/include/Api.hpp"
#include "prx/libkernel/Time/include/SleepWake.hpp"
#include "prx/libkernel/Time/include/Time.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" void AgcDriverSubmitGraphics_nid_postfix(std::uint32_t* commands, std::uint32_t dwords);
extern "C" void AgcDriverSuspendPoint_nid_postfix();
extern "C" void AgcDriverWaitIdle_nid_postfix();
extern "C" void AgcDriverPrestartGpu_nid_postfix();
extern "C" void AgcDriverShutdown_nid_postfix();
extern "C" void* AgcDriverStaticTable_nid_postfix();
extern "C" void AgcDriverLockVulkanLoader_nid_postfix();
extern "C" void AgcDriverUnlockVulkanLoader_nid_postfix();
extern "C" void AgcDriverRegisterShader_nid_postfix(const Shader* shader);
extern "C" void AgcDriverTrackSerials_nid_postfix();
extern "C" void AgcDriverSerialProgress_nid_postfix(std::uint64_t* accepted, std::uint64_t* completed);
extern "C" bool AgcDriverSettled_nid_postfix();
extern "C" bool AgcDriverHoldSettled_nid_postfix(std::int64_t milliseconds);
extern "C" void AgcDriverReleaseSettled_nid_postfix();
extern "C" std::size_t AgcDriverAwaitedWaits_nid_postfix(AgcDriver::AwaitedWait* waits, std::size_t capacity);
extern "C" void AgcDriverWaitForPipelineJournal_nid_postfix();
extern "C" std::size_t AgcDriverReplayCounters_nid_postfix(std::uint64_t* values, std::size_t capacity);
extern "C" void AgcDriverTracePresent_nid_postfix(std::uint32_t width, std::uint32_t height);

namespace KytyShim {
void StartPgoWriter();
}

namespace Libs::LibKernel::Memory::Shim {
void EnableView();
}

namespace {

const KytyDllHost* g_host = nullptr;
thread_local std::string t_error;

const KytyDllHost* Host(const char* caller) noexcept {
    if (g_host == nullptr) [[unlikely]] {
        std::FILE* error = __acrt_iob_func(2);
        std::fprintf(error, "kyty_gpu: %s called into AnyPS5 before KytyDllCreate (from a static initializer?)\n", caller);
        std::fflush(error);
        std::abort();
    }
    return g_host;
}

[[noreturn]] void HostFailed(KytyDllStatus status) {
    if (status == KYTY_DLL_PROCESS_SHUTDOWN) throw ProcessShutdown{};
    std::string message(1024, '\0');
    const auto bytes = Host(__func__)->take_error(message.data(), message.size());
    message.resize(std::min(bytes, message.size()));
    if (status == KYTY_DLL_EOP_FLIP_ALREADY_RESERVED) throw AgcDriver::EopFlipAlreadyReserved{};
    throw std::runtime_error(message);
}

void Check(KytyDllStatus status) {
    if (status != KYTY_DLL_OK) HostFailed(status);
}

template<class Function>
KytyDllStatus Guard(Function&& function) noexcept {
    try {
        function();
        return KYTY_DLL_OK;
    } catch (const ProcessShutdown&) {
        t_error = "process shutdown";
        return KYTY_DLL_PROCESS_SHUTDOWN;
    } catch (const AgcDriver::EopFlipAlreadyReserved& error) {
        t_error = error.what();
        return KYTY_DLL_EOP_FLIP_ALREADY_RESERVED;
    } catch (const std::exception& error) {
        t_error = error.what();
        return KYTY_DLL_ERROR;
    } catch (...) {
        t_error = "kyty_gpu: unknown exception";
        return KYTY_DLL_ERROR;
    }
}

std::size_t TakeError(char* message, std::size_t capacity) noexcept {
    const auto bytes = t_error.size();
    if (message != nullptr && capacity != 0) std::memcpy(message, t_error.data(), std::min(bytes, capacity));
    return bytes;
}

AgcDriver::FlipInfo ToFlip(const KytyDllFlipInfo& info) {
    return {info.handle, info.index, info.mode, info.argument};
}

KytyDllFlipInfo ToFlip(const AgcDriver::FlipInfo& info) {
    return {info.handle, info.index, info.mode, info.argument};
}

AgcDriver::DisplayBuffer ToBuffer(const KytyDllDisplayBuffer& buffer) {
    AgcDriver::DisplayBuffer ours{buffer.address, buffer.pixel_format, buffer.width, buffer.height};
    ours.tilingMode = buffer.tiling_mode;
    ours.pitchInPixel = buffer.pitch_in_pixel;
    ours.dccAddress = buffer.dcc_address;
    ours.dccClearColor = buffer.dcc_clear_color;
    ours.alignment = buffer.alignment;
    ours.dccControl = buffer.dcc_control;
    return ours;
}

KytyDllDisplayBuffer ToBuffer(const AgcDriver::DisplayBuffer& buffer) {
    return {buffer.address, buffer.pixelFormat, buffer.width, buffer.height, buffer.tilingMode, buffer.pitchInPixel, buffer.dccAddress, buffer.dccClearColor, buffer.alignment, buffer.dccControl};
}

AgcDriver::PresentationWindow ToWindow(const KytyDllWindow& window) {
    return {window.context, std::span<const char* const>(window.extensions, window.extension_count), window.create_surface, window.get_drawable_size, window.width, window.height, nullptr};
}

std::string HostPath(std::size_t (*resolve)(const char*, char*, std::size_t), const char* path) {
    std::string resolved(512, '\0');
    auto bytes = resolve(path, resolved.data(), resolved.size());
    if (bytes > resolved.size()) {
        resolved.resize(bytes);
        bytes = resolve(path, resolved.data(), resolved.size());
    }
    resolved.resize(std::min(bytes, resolved.size()));
    return resolved;
}

class HostRenderingWait final : public AgcDriver::IRenderingWait {
public:
    explicit HostRenderingWait(KytyDllRenderingWait* wait) : m_wait(wait) {}
    ~HostRenderingWait() override { Host(__func__)->rendering_wait_release(m_wait); }
    void Wait() override { Check(Host(__func__)->rendering_wait(m_wait)); }

private:
    KytyDllRenderingWait* m_wait;
};

class HostFlipRequest final : public AgcDriver::IFlipRequest {
public:
    explicit HostFlipRequest(KytyDllFlipRequest* request) : m_request(request) {}
    ~HostFlipRequest() override { Host(__func__)->flip_release(m_request); }

    void GpuReady(const std::shared_ptr<AgcDriver::FrameTiming>&) override {
        Check(Host(__func__)->flip_gpu_ready(m_request));
    }

    void Fail(std::exception_ptr error) noexcept override {
        Host(__func__)->flip_fail(m_request, Message(error).c_str());
    }

    std::optional<AgcDriver::DisplayBuffer> Source() override {
        KytyDllDisplayBuffer source{};
        bool has = false;
        Check(Host(__func__)->flip_source(m_request, &source, &has));
        if (!has) return std::nullopt;
        return ToBuffer(source);
    }

    static std::string Message(std::exception_ptr error) noexcept {
        try {
            if (error) std::rethrow_exception(error);
        } catch (const std::exception& failure) {
            return failure.what();
        } catch (...) {
        }
        return "kyty_gpu: flip failed";
    }

private:
    KytyDllFlipRequest* m_request;
};

class HostVideoOutput final : public AgcDriver::IVideoOutput {
public:
    explicit HostVideoOutput(KytyDllVideoOutput* output) : m_output(output) {}
    ~HostVideoOutput() override { Host(__func__)->video_release(m_output); }

    std::shared_ptr<AgcDriver::IFlipRequest> Reserve(const AgcDriver::FlipInfo& info) override { return Make(info, false); }
    std::shared_ptr<AgcDriver::IFlipRequest> TryReserve(const AgcDriver::FlipInfo& info) override { return Make(info, true); }
    void WaitForFlipRoom() override { Check(Host(__func__)->video_wait_for_flip_room(m_output)); }

    std::shared_ptr<AgcDriver::IRenderingWait> CaptureRenderingWait(std::uint32_t index) override {
        KytyDllRenderingWait* wait = nullptr;
        Check(Host(__func__)->video_capture_rendering_wait(m_output, index, &wait));
        return std::make_shared<HostRenderingWait>(wait);
    }

    void Fail(std::exception_ptr error) noexcept override {
        Host(__func__)->video_fail(m_output, HostFlipRequest::Message(error).c_str());
    }

    [[nodiscard]] KytyDllVideoOutput* Handle() const { return m_output; }

private:
    std::shared_ptr<AgcDriver::IFlipRequest> Make(const AgcDriver::FlipInfo& info, bool tryOnly) {
        const auto flip = ToFlip(info);
        KytyDllFlipRequest* request = nullptr;
        Check(Host(__func__)->video_reserve(m_output, &flip, tryOnly, &request));
        if (request == nullptr) return nullptr;
        return std::make_shared<HostFlipRequest>(request);
    }

    KytyDllVideoOutput* m_output;
};

std::mutex g_outputs_mutex;
auto& g_outputs = *new std::unordered_map<KytyDllVideoOutput*, std::weak_ptr<HostVideoOutput>>;

std::shared_ptr<HostVideoOutput> OutputFor(KytyDllVideoOutput* handle) {
    std::lock_guard lock(g_outputs_mutex);
    auto& slot = g_outputs[handle];
    if (auto output = slot.lock()) return output;
    auto output = std::make_shared<HostVideoOutput>(handle);
    slot = output;
    return output;
}

struct FilterBridge {
    KernelFilter filter;
};

struct ViewObserver {
    static constexpr const char* kName = "the memory view observer";
};

struct ProtectionObserver {
    static constexpr const char* kName = "the memory protection observer";
};

struct GpuFaultHandler {
    static constexpr const char* kName = "the GPU fault handler";
};

struct HostWriteFilter {
    static constexpr const char* kName = "the host-write filter";
};

struct HostWritePreparer {
    static constexpr const char* kName = "the host-write preparer";
};

struct ShutdownCallback {
    static constexpr const char* kName = "a shutdown callback";
};

constexpr std::size_t kShutdownCallbacks = 8;

KernelEvent ToOurs(const KytyDllKernelEvent& event) {
    KernelEvent ours;
    ours.ident = event.ident;
    ours.filter = event.filter;
    ours.flags = event.flags;
    ours.fflags = event.fflags;
    ours.data = event.data;
    ours.udata = event.udata;
    return ours;
}

KytyDllKernelEvent ToSeam(const KernelEvent& event) {
    return {event.ident, event.filter, event.flags, event.fflags, event.data, event.udata};
}

KernelEqueueEvent FromState(const FilterBridge& bridge, const KytyDllEqueueEventState& state) {
    KernelEqueueEvent event;
    event.triggered = state.triggered;
    event.deadlineNs = state.deadline_ns;
    event.intervalNs = state.interval_ns;
    event.event = ToOurs(state.event);
    for (std::size_t i = 0; i < state.pending_count; i++) event.pendingEvents.push_back(ToOurs(state.pending[i]));
    event.filter = bridge.filter;
    return event;
}

void ToState(const KernelEqueueEvent& event, KytyDllEqueueEventState& state) {
    if (event.pendingEvents.size() > state.pending_capacity) throw std::runtime_error("kyty_gpu: an event filter left more pending events than the seam carries");
    state.triggered = event.triggered;
    state.deadline_ns = event.deadlineNs;
    state.interval_ns = event.intervalNs;
    state.event = ToSeam(event.event);
    state.pending_count = 0;
    for (const auto& pending : event.pendingEvents) state.pending[state.pending_count++] = ToSeam(pending);
}

const KytyDllApi* MakeApi() {
    static KytyDllApi api = [] {
        KytyDllApi table{};
        table.abi_version = KYTY_DLL_ABI_VERSION;
        table.size = sizeof(KytyDllApi);
        table.take_error = &TakeError;
        table.submit = [](const void* packet, std::uint32_t queue) { return Guard([&] { AgcDriver::Submit(static_cast<const Packet*>(packet), queue); }); };
        table.submit_graphics = [](std::uint32_t* commands, std::uint32_t dwords) { return Guard([&] { AgcDriverSubmitGraphics_nid_postfix(commands, dwords); }); };
        table.suspend_point = [] { return Guard([] { AgcDriverSuspendPoint_nid_postfix(); }); };
        table.wait_idle = [] { return Guard([] { AgcDriverWaitIdle_nid_postfix(); }); };
        table.prestart_gpu = [] { return Guard([] { AgcDriverPrestartGpu_nid_postfix(); }); };
        table.shutdown = [] { return Guard([] { AgcDriverShutdown_nid_postfix(); }); };
        table.check_guest_memory = [](const void* pointer, std::size_t bytes, std::size_t alignment, bool writable) { return Guard([&] { AgcDriver::GuestMemory::CheckRange(pointer, bytes, alignment, writable); }); };
        table.display_buffer_size = [](const KytyDllDisplayBuffer* buffer, std::size_t* bytes) { return Guard([&] { *bytes = AgcDriver::DisplayBufferSize(ToBuffer(*buffer)); }); };
        table.static_table = [](void** pointer) { return Guard([&] { *pointer = AgcDriverStaticTable_nid_postfix(); }); };
        table.prepare_command_write = [](std::uint64_t address, std::size_t bytes) { return Guard([&] { AgcDriverPrepareCommandWrite_nid_no_patch(address, bytes); }); };
        table.note_command_segment = [](std::uint64_t address, std::size_t bytes) { return Guard([&] { AgcDriverNoteCommandSegment_nid_no_patch(address, bytes); }); };
        table.store_guest_label = [](std::uint64_t* pointer, std::uint64_t value) { return Guard([&] { AgcDriverStoreGuestLabel_nid_postfix(pointer, value); }); };
        table.lock_vulkan_loader = [] { return Guard([] { AgcDriverLockVulkanLoader_nid_postfix(); }); };
        table.unlock_vulkan_loader = [] { return Guard([] { AgcDriverUnlockVulkanLoader_nid_postfix(); }); };
        table.register_shader = [](const void* shader) { return Guard([&] { AgcDriverRegisterShader_nid_postfix(static_cast<const Shader*>(shader)); }); };
        table.submit_flip = [](const KytyDllFlipInfo* info) { return Guard([&] { AgcDriverSubmitFlip_nid_postfix(ToFlip(*info)); }); };
        table.reserve_eop_flip = [](const KytyDllFlipInfo* info, std::uint32_t* token) { return Guard([&] { *token = AgcDriverReserveEopFlip_nid_postfix(ToFlip(*info)); }); };
        table.reserve_eop_flip_at = [](const KytyDllFlipInfo* info, std::uint32_t token, std::uint32_t* reserved) { return Guard([&] { *reserved = AgcDriverReserveEopFlipAt_nid_postfix(ToFlip(*info), token); }); };
        table.register_video_output = [](std::uint32_t handle, KytyDllVideoOutput* output) {
            return Guard([&] {
                std::shared_ptr<AgcDriver::IVideoOutput> ours = OutputFor(output);
                AgcDriverRegisterVideoOutput_nid_postfix(handle, ours);
            });
        };
        table.unregister_video_output = [](std::uint32_t handle, KytyDllVideoOutput* output) {
            return Guard([&] {
                std::shared_ptr<AgcDriver::IVideoOutput> ours = OutputFor(output);
                AgcDriverUnregisterVideoOutput_nid_postfix(handle, ours);
            });
        };
        table.attach_window = [](const KytyDllWindow* window) { return Guard([&] { AgcDriverAttachWindow_nid_postfix(ToWindow(*window)); }); };
        table.present_buffer = [](const KytyDllWindow* window, const KytyDllDisplayBuffer* buffer, const KytyDllDisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
            return Guard([&] {
                const auto ours = ToBuffer(*buffer);
                std::optional<AgcDriver::DisplayBuffer> ourPlane;
                if (plane != nullptr) ourPlane = ToBuffer(*plane);
                AgcDriverPresentBuffer_nid_postfix(ToWindow(*window), ours, ourPlane ? &*ourPlane : nullptr, gpuReady, context);
            });
        };
        table.present_clear = [](const KytyDllWindow* window, bool opaque, const KytyDllDisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
            return Guard([&] {
                std::optional<AgcDriver::DisplayBuffer> ourPlane;
                if (plane != nullptr) ourPlane = ToBuffer(*plane);
                AgcDriverPresentClear_nid_postfix(ToWindow(*window), opaque, ourPlane ? &*ourPlane : nullptr, gpuReady, context);
            });
        };
        table.release_window = [](void* window) { return Guard([&] { AgcDriverReleaseWindow_nid_postfix(window); }); };
        table.track_serials = [] { return Guard([] { AgcDriverTrackSerials_nid_postfix(); }); };
        table.serial_progress = [](std::uint64_t* accepted, std::uint64_t* completed) { return Guard([&] { AgcDriverSerialProgress_nid_postfix(accepted, completed); }); };
        table.settled = [](bool* settled) { return Guard([&] { *settled = AgcDriverSettled_nid_postfix(); }); };
        table.hold_settled = [](std::int64_t milliseconds, bool* held) { return Guard([&] { *held = AgcDriverHoldSettled_nid_postfix(milliseconds); }); };
        table.release_settled = [] { return Guard([] { AgcDriverReleaseSettled_nid_postfix(); }); };
        table.awaited_waits = [](KytyDllAwaitedWait* waits, std::size_t capacity, std::size_t* count) {
            static_assert(sizeof(KytyDllAwaitedWait) == sizeof(AgcDriver::AwaitedWait));
            return Guard([&] { *count = AgcDriverAwaitedWaits_nid_postfix(reinterpret_cast<AgcDriver::AwaitedWait*>(waits), capacity); });
        };
        table.wait_for_pipeline_journal = [] { return Guard([] { AgcDriverWaitForPipelineJournal_nid_postfix(); }); };
        table.replay_counters = [](std::uint64_t* values, std::size_t capacity, std::size_t* count) { return Guard([&] { *count = AgcDriverReplayCounters_nid_postfix(values, capacity); }); };
        table.trace_present = [](std::uint32_t width, std::uint32_t height) { return Guard([&] { AgcDriverTracePresent_nid_postfix(width, height); }); };
        table.event_trigger = [](KytyDllEventFilter* filter, KytyDllEqueueEventState* state, void* data) {
            return Guard([&] {
                const auto& bridge = *reinterpret_cast<FilterBridge*>(filter);
                auto event = FromState(bridge, *state);
                bridge.filter.triggerFunc(&event, data);
                ToState(event, *state);
            });
        };
        table.event_reset = [](KytyDllEventFilter* filter, KytyDllEqueueEventState* state) {
            return Guard([&] {
                const auto& bridge = *reinterpret_cast<FilterBridge*>(filter);
                auto event = FromState(bridge, *state);
                bridge.filter.resetFunc(&event);
                ToState(event, *state);
            });
        };
        table.event_delete = [](KytyDllEventFilter* filter, std::int64_t eq, KytyDllEqueueEventState* state) {
            return Guard([&] {
                const auto& bridge = *reinterpret_cast<FilterBridge*>(filter);
                auto event = FromState(bridge, *state);
                bridge.filter.deleteEventFunc(eq, &event);
            });
        };
        table.event_filter_release = [](KytyDllEventFilter* filter) noexcept { delete reinterpret_cast<FilterBridge*>(filter); };
        return table;
    }();
    return &api;
}

}

namespace KytyShim {

std::FILE* TrampolineLog() noexcept {
    return HostStandardErrorFile_nid_no_patch();
}

bool DllHostReportSetting() {
    bool enabled = false;
    Check(Host(__func__)->host_report_setting(&enabled));
    return enabled;
}

void DllReportHost(const void* properties, const void* driver, const void* memory, std::uint32_t subgroupSize, const char* shaderStorePath) {
    Check(Host(__func__)->report_host(properties, driver, memory, subgroupSize, shaderStorePath));
}

}

extern "C" KytyDllStatus KytyDllCreate(const KytyDllHost* host, const KytyDllApi** api) noexcept {
    if (api == nullptr || !KytyDllHostMatches(host)) return KYTY_DLL_ABI_MISMATCH;
    g_host = host;
    const auto status = Guard([&] {
        static std::once_flag view;
        std::call_once(view, [] { Libs::LibKernel::Memory::Shim::EnableView(); });
        const auto* table = MakeApi();
        static std::once_flag pgo;
        std::call_once(pgo, [] { KytyShim::StartPgoWriter(); });
        *api = table;
    });
    if (status != KYTY_DLL_OK) {
        std::FILE* error = HostStandardErrorFile_nid_no_patch();
        std::fprintf(error, "kyty_gpu: KytyDllCreate failed: %s\n", t_error.c_str());
        std::fflush(error);
    }
    return status;
}

extern "C" {

bool DirectMemoryEnableView_nid_no_patch(DirectMemoryViewObserver observer, std::uint8_t** view, std::size_t* guestBytes, std::size_t* viewBytes) {
    return Host(__func__)->direct_memory_enable_view(KytyShim::Trampoline<ViewObserver, void(std::uintptr_t, std::size_t, std::uint64_t)>::Bind(observer), view, guestBytes, viewBytes);
}

void DirectMemoryWatchProtections_nid_no_patch(DirectMemoryProtectionObserver observer) {
    Host(__func__)->direct_memory_watch_protections(KytyShim::Trampoline<ProtectionObserver, void(std::uintptr_t, std::size_t, int)>::Bind(observer));
}

std::size_t DirectMemorySize_nid_no_patch() {
    return Host(__func__)->direct_memory_size();
}

bool DirectMemoryFind_nid_no_patch(std::int64_t offset, bool findNext, std::int64_t* start, std::int64_t* end, int* memoryType) {
    return Host(__func__)->direct_memory_find(offset, findNext, start, end, memoryType);
}

void GuestPageTableUsage_nid_no_patch(std::uint64_t granularity, std::uint64_t* cpuEntries, std::uint64_t* gpuEntries) {
    Host(__func__)->guest_page_table_usage(granularity, cpuEntries, gpuEntries);
}

int DoMapAnon_nid_no_patch(void** addr, std::size_t len, int prot, int flags, std::size_t alignment, bool exclusive) {
    int result = 0;
    Check(Host(__func__)->do_map_anon(addr, len, prot, flags, alignment, exclusive, &result));
    return result;
}

int DoMunmap_nid_no_patch(void* addr, std::size_t len) {
    int result = 0;
    Check(Host(__func__)->do_munmap(addr, len, &result));
    return result;
}

void* DirectMemoryMapHostView_nid_no_patch(std::uintptr_t address, std::size_t bytes, int prot) {
    return Host(__func__)->direct_memory_map_host_view(address, bytes, prot);
}

void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes) {
    Host(__func__)->guest_arena_range(base, bytes);
}

void GuestArenaNoteHostProtection_nid_no_patch(std::uintptr_t address, std::size_t bytes, bool writable) {
    Host(__func__)->guest_arena_note_host_protection(address, bytes, writable);
}

std::uint64_t GuestArenaOpenHostWrite_nid_no_patch(void* pointer, std::size_t bytes) {
    return Host(__func__)->guest_arena_open_host_write(pointer, bytes);
}

void GuestArenaCloseHostWrite_nid_no_patch(void* pointer, std::size_t bytes, std::uint64_t token) {
    Host(__func__)->guest_arena_close_host_write(pointer, bytes, token);
}

void GuestArenaSetGpuFaultHandler_nid_no_patch(bool (*handler)(void* exceptionPointers)) {
    Host(__func__)->guest_arena_set_gpu_fault_handler(KytyShim::Trampoline<GpuFaultHandler, bool(void*)>::Bind(handler));
}

void GuestArenaSetHostWriteFilter_nid_no_patch(bool (*filter)(std::uintptr_t address, std::size_t bytes)) {
    Host(__func__)->guest_arena_set_host_write_filter(KytyShim::Trampoline<HostWriteFilter, bool(std::uintptr_t, std::size_t)>::Bind(filter));
}

void GuestArenaSetHostWritePreparer_nid_no_patch(bool (*prepare)(std::uintptr_t address, std::size_t bytes)) {
    Host(__func__)->guest_arena_set_host_write_preparer(KytyShim::Trampoline<HostWritePreparer, bool(std::uintptr_t, std::size_t)>::Bind(prepare));
}

void* GuestAllocationsBegin_nid_postfix() {
    return Host(__func__)->guest_allocations_begin();
}

void GuestAllocationsAdd_nid_postfix(void* mutation, void* pointer, std::size_t bytes, bool readable, bool writable, bool gpu) {
    Host(__func__)->guest_allocations_add(mutation, pointer, bytes, readable, writable, gpu);
}

void GuestAllocationsEnd_nid_postfix(void* mutation) noexcept {
    Host(__func__)->guest_allocations_end(mutation);
}

void KernelNoteGpuSubmit_nid_no_patch() {
    Host(__func__)->kernel_note_gpu_submit();
}

int sceKernelAvailableFlexibleMemorySize(size_t* size) {
    return Host(__func__)->kernel_available_flexible_memory_size(size);
}

int sceKernelConfiguredFlexibleMemorySize(size_t* size) {
    return Host(__func__)->kernel_configured_flexible_memory_size(size);
}

uint64_t sceKernelReadTsc(void) {
    return Host(__func__)->kernel_read_tsc();
}

std::uint64_t sceKernelGetTscFrequency() {
    return Host(__func__)->kernel_tsc_frequency();
}

std::uint64_t sceKernelGetProcessTime() {
    return Host(__func__)->kernel_process_time();
}

std::uint64_t sceKernelGetProcessTimeCounter() {
    return Host(__func__)->kernel_process_time_counter();
}

KernelEqueueRef EqueuePin_nid_postfix(KernelEqueue eq) {
    auto* queue = Host(__func__)->equeue_pin(eq);
    if (queue == nullptr) return {};
    return KernelEqueueRef(reinterpret_cast<KernelEqueuePrivate*>(queue), [](KernelEqueuePrivate* pinned) { Host(__func__)->equeue_unpin(reinterpret_cast<KytyDllEqueue*>(pinned)); });
}

int EqueueAddEvent_nid_postfix(KernelEqueue eq, const KernelEqueueEvent& event) {
    std::vector<KytyDllKernelEvent> pending;
    pending.reserve(event.pendingEvents.size());
    for (const auto& item : event.pendingEvents) pending.push_back(ToSeam(item));
    KytyDllEqueueEvent seam{event.triggered, event.deadlineNs, event.intervalNs, ToSeam(event.event), pending.data(), pending.size(), nullptr, 0};
    const auto& filter = event.filter;
    if (filter.data != nullptr || filter.owner || filter.triggerFunc != nullptr || filter.resetFunc != nullptr || filter.deleteEventFunc != nullptr) {
        if (filter.pollFunc != nullptr || filter.watched) throw std::runtime_error("kyty_gpu: an event filter with a poll function or a watched queue cannot cross to libkernel");
        seam.filter = reinterpret_cast<KytyDllEventFilter*>(new FilterBridge{filter});
        seam.filter_functions = (filter.triggerFunc != nullptr ? KYTY_DLL_FILTER_TRIGGER : 0u) | (filter.resetFunc != nullptr ? KYTY_DLL_FILTER_RESET : 0u) | (filter.deleteEventFunc != nullptr ? KYTY_DLL_FILTER_DELETE : 0u);
    }
    int result = 0;
    Check(Host(__func__)->equeue_add_event(eq, &seam, &result));
    return result;
}

int EqueueTriggerEvent_nid_postfix(KernelEqueue eq, uintptr_t ident, int16_t filter, void* triggerData) {
    int result = 0;
    Check(Host(__func__)->equeue_trigger_event(eq, ident, filter, triggerData, &result));
    return result;
}

int EqueueDeleteEvent_nid_postfix(KernelEqueue eq, uintptr_t ident, int16_t filter) {
    int result = 0;
    Check(Host(__func__)->equeue_delete_event(eq, ident, filter, &result));
    return result;
}

std::FILE* HostStandardOutputFile_nid_no_patch() {
    return g_host != nullptr ? g_host->standard_output_file() : __acrt_iob_func(1);
}

std::FILE* HostStandardErrorFile_nid_no_patch() {
    return g_host != nullptr ? g_host->standard_error_file() : __acrt_iob_func(2);
}

void LibcRegisterShutdown_nid_postfix(void (*callback)()) {
    Host(__func__)->register_shutdown(KytyShim::TrampolinePool<ShutdownCallback, kShutdownCallbacks, void()>::Bind(callback));
}

std::stop_token LibcShutdownToken_nid_postfix() {
    static std::stop_source requested;
    if (Host(__func__)->shutdown_requested()) requested.request_stop();
    return requested.get_token();
}

[[noreturn]] void LibcAwaitExit_nid_postfix() {
    Host(__func__)->await_exit();
    std::abort();
}

[[noreturn]] void LibcTerminate_nid_no_patch(int code) {
    Host(__func__)->terminate(code);
    std::abort();
}

std::filesystem::path ResolvePath_nid_no_patch(const char* path) {
    return std::filesystem::u8path(HostPath(Host(__func__)->resolve_path, path));
}

std::filesystem::path ResolveDiskPath_nid_no_patch(const char* path) {
    return std::filesystem::u8path(HostPath(Host(__func__)->resolve_disk_path, path));
}

}

void AgcDriverDeliverEopInterrupt(std::uint32_t queue, std::uint32_t context) {
    Host(__func__)->deliver_eop_interrupt(queue, context);
}

namespace AgcDriver {

void RaiseWorkerThreadPriority(const char* role) {
    Host(__func__)->raise_worker_thread_priority(role);
}

bool SystemOverlayShown() {
    return Host(__func__)->system_overlay_shown();
}

std::optional<OverlayLayout> SystemOverlayLayout(VkExtent2D extent) {
    KytyDllOverlayRect bounds{};
    std::int32_t scale = 1;
    std::size_t count = 0;
    std::vector<KytyDllOverlayRect> rects(64);
    for (;;) {
        if (!Host(__func__)->system_overlay_layout(extent.width, extent.height, &bounds, &scale, rects.data(), rects.size(), &count)) return std::nullopt;
        if (count <= rects.size()) break;
        rects.resize(count);
    }
    OverlayLayout layout;
    layout.bounds = {bounds.x, bounds.y, bounds.width, bounds.height, bounds.color};
    layout.scale = scale;
    for (std::size_t i = 0; i < count; i++) layout.rects.push_back({rects[i].x, rects[i].y, rects[i].width, rects[i].height, rects[i].color});
    return layout;
}

}
