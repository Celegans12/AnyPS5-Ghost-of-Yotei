#include "prx/libSceAgcDriver/KytyShim/host/DllCreate.hpp"
#include "prx/libSceAgcDriver/KytyShim/include/KytyDll.h"

#include "SceShaders.hpp"
#include "SceTypes.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostReportSources.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostReportVulkan.hpp"
#include "prx/libSceAgcDriver/Execution/include/KeyboardOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PadStatusOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparationScope.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/RunReport.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "prx/libkernel/include/Api.hpp"
#include "prx/libkernel/Time/include/SleepWake.hpp"
#include "prx/libkernel/Time/include/Time.hpp"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

const KytyDllApi* g_api = nullptr;
thread_local std::string t_error;

struct OutputHolder {
    std::shared_ptr<AgcDriver::IVideoOutput> output;
};

struct RequestHolder {
    std::shared_ptr<AgcDriver::IFlipRequest> request;
};

struct WaitHolder {
    std::shared_ptr<AgcDriver::IRenderingWait> wait;
};

std::mutex g_outputs_mutex;
std::unordered_map<std::uint32_t, OutputHolder*> g_outputs;

std::size_t TakeError(char* message, std::size_t capacity) {
    const auto bytes = t_error.size();
    if (message != nullptr && capacity != 0) std::memcpy(message, t_error.data(), std::min(bytes, capacity));
    return bytes;
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
        t_error = "libSceAgcDriver: unknown exception";
        return KYTY_DLL_ERROR;
    }
}

template<class Function>
auto Quiet(const char* what, Function&& function) noexcept -> decltype(function()) {
    try {
        return function();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[kyty] %s failed: %s\n", what, error.what());
    } catch (...) {
        std::fprintf(stderr, "[kyty] %s failed\n", what);
    }
    std::fflush(stderr);
    LibcTerminate_nid_no_patch(1);
}

std::exception_ptr Failure(const char* message) {
    return std::make_exception_ptr(std::runtime_error(message != nullptr ? message : "kyty_gpu: failure"));
}

KytyDllFlipInfo ToSeam(const AgcDriver::FlipInfo& info) {
    return {info.handle, info.index, info.mode, info.argument};
}

AgcDriver::FlipInfo ToOurs(const KytyDllFlipInfo& info) {
    return {info.handle, info.index, info.mode, info.argument};
}

KytyDllDisplayBuffer ToSeam(const AgcDriver::DisplayBuffer& buffer) {
    return {buffer.address, buffer.pixelFormat, buffer.width, buffer.height, buffer.tilingMode, buffer.pitchInPixel, buffer.dccAddress, buffer.dccClearColor, buffer.alignment, buffer.dccControl};
}

KytyDllWindow ToSeam(const AgcDriver::PresentationWindow& window) {
    return {window.context, window.extensions.data(), window.extensions.size(), window.createSurface, window.getDrawableSize, window.width, window.height};
}

KytyDllKernelEvent ToSeam(const KernelEvent& event) {
    return {event.ident, event.filter, event.flags, event.fflags, event.data, event.udata};
}

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

struct FilterHandle {
    explicit FilterHandle(KytyDllEventFilter* handle) : filter(handle) {}
    FilterHandle(const FilterHandle&) = delete;
    FilterHandle& operator=(const FilterHandle&) = delete;
    ~FilterHandle() { g_api->event_filter_release(filter); }
    KytyDllEventFilter* filter;
};

struct EventState {
    std::vector<KytyDllKernelEvent> pending;
    KytyDllEqueueEventState state{};

    explicit EventState(const KernelEqueueEvent& event) {
        pending.resize(std::max<std::size_t>(event.pendingEvents.size(), KernelEqueuePrivate::kMaxPendingEvents));
        state.triggered = event.triggered;
        state.deadline_ns = event.deadlineNs;
        state.interval_ns = event.intervalNs;
        state.event = ToSeam(event.event);
        for (const auto& item : event.pendingEvents) pending[state.pending_count++] = ToSeam(item);
        state.pending = pending.data();
        state.pending_capacity = pending.size();
    }

    void Reflect(KernelEqueueEvent& event) const {
        event.triggered = state.triggered;
        event.deadlineNs = state.deadline_ns;
        event.intervalNs = state.interval_ns;
        event.event = ToOurs(state.event);
        event.pendingEvents.clear();
        for (std::size_t i = 0; i < state.pending_count; i++) event.pendingEvents.push_back(ToOurs(state.pending[i]));
    }
};

[[noreturn]] void DllFailed(KytyDllStatus status) {
    if (status == KYTY_DLL_PROCESS_SHUTDOWN) throw ProcessShutdown{};
    if (status == KYTY_DLL_EOP_FLIP_ALREADY_RESERVED) throw AgcDriver::EopFlipAlreadyReserved{};
    std::string message(1024, '\0');
    const auto bytes = g_api->take_error(message.data(), message.size());
    message.resize(std::min(bytes, message.size()));
    throw std::runtime_error(message);
}

void Check(KytyDllStatus status) {
    if (status != KYTY_DLL_OK) DllFailed(status);
}

void TriggerThroughDll(KernelEqueueEvent* event, void* data) {
    EventState state(*event);
    Check(g_api->event_trigger(static_cast<FilterHandle*>(event->filter.data)->filter, &state.state, data));
    state.Reflect(*event);
}

void ResetThroughDll(KernelEqueueEvent* event) {
    EventState state(*event);
    Check(g_api->event_reset(static_cast<FilterHandle*>(event->filter.data)->filter, &state.state));
    state.Reflect(*event);
}

void DeleteThroughDll(KernelEqueue eq, KernelEqueueEvent* event) {
    EventState state(*event);
    Check(g_api->event_delete(static_cast<FilterHandle*>(event->filter.data)->filter, eq, &state.state));
}

std::size_t CopyPath(const std::filesystem::path& path, char* resolved, std::size_t capacity) {
    const auto text = path.u8string();
    if (resolved != nullptr && text.size() <= capacity) std::memcpy(resolved, text.data(), text.size());
    return text.size();
}

KytyDllHost MakeHost() {
    KytyDllHost host{};
    host.abi_version = KYTY_DLL_ABI_VERSION;
    host.size = sizeof(KytyDllHost);
    host.take_error = &TakeError;
    host.direct_memory_find = [](std::int64_t offset, bool findNext, std::int64_t* start, std::int64_t* end, int* memoryType) { return Quiet("DirectMemoryFind", [&] { return DirectMemoryFind_nid_no_patch(offset, findNext, start, end, memoryType); }); };
    host.direct_memory_map_host_view = [](std::uintptr_t address, std::size_t bytes, int prot) { return Quiet("DirectMemoryMapHostView", [&] { return DirectMemoryMapHostView_nid_no_patch(address, bytes, prot); }); };
    host.direct_memory_size = [] { return Quiet("DirectMemorySize", [] { return DirectMemorySize_nid_no_patch(); }); };
    host.direct_memory_enable_view = [](void (*observer)(std::uintptr_t, std::size_t, std::uint64_t), std::uint8_t** view, std::size_t* guestBytes, std::size_t* viewBytes) { return Quiet("DirectMemoryEnableView", [&] { return DirectMemoryEnableView_nid_no_patch(observer, view, guestBytes, viewBytes); }); };
    host.direct_memory_watch_protections = [](void (*observer)(std::uintptr_t, std::size_t, int)) { Quiet("DirectMemoryWatchProtections", [&] { DirectMemoryWatchProtections_nid_no_patch(observer); }); };
    host.do_map_anon = [](void** address, std::size_t length, int prot, int flags, std::size_t alignment, bool exclusive, int* result) { return Guard([&] { *result = DoMapAnon_nid_no_patch(address, length, prot, flags, alignment, exclusive); }); };
    host.do_munmap = [](void* address, std::size_t length, int* result) { return Guard([&] { *result = DoMunmap_nid_no_patch(address, length); }); };
    host.guest_arena_range = [](std::uintptr_t* base, std::size_t* bytes) { Quiet("GuestArenaRange", [&] { GuestArena::GuestArenaRange_nid_postfix(base, bytes); }); };
    host.guest_arena_note_host_protection = [](std::uintptr_t address, std::size_t bytes, bool writable) { Quiet("GuestArenaNoteHostProtection", [&] { GuestArena::GuestArenaNoteHostProtection_nid_no_patch(address, bytes, writable); }); };
    host.guest_arena_open_host_write = [](void* pointer, std::size_t bytes) { return Quiet("GuestArenaOpenHostWrite", [&] { return GuestArena::GuestArenaOpenHostWrite_nid_no_patch(pointer, bytes); }); };
    host.guest_arena_close_host_write = [](void* pointer, std::size_t bytes, std::uint64_t token) { Quiet("GuestArenaCloseHostWrite", [&] { GuestArena::GuestArenaCloseHostWrite_nid_no_patch(pointer, bytes, token); }); };
    host.guest_arena_set_gpu_fault_handler = [](bool (*handler)(void*)) { Quiet("GuestArenaSetGpuFaultHandler", [&] { GuestArena::GuestArenaSetGpuFaultHandler_nid_no_patch(handler); }); };
    host.guest_arena_set_host_write_filter = [](bool (*filter)(std::uintptr_t, std::size_t)) { Quiet("GuestArenaSetHostWriteFilter", [&] { GuestArena::GuestArenaSetHostWriteFilter_nid_no_patch(filter); }); };
    host.guest_arena_set_host_write_preparer = [](bool (*prepare)(std::uintptr_t, std::size_t)) { Quiet("GuestArenaSetHostWritePreparer", [&] { GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(prepare); }); };
    host.guest_page_table_usage = [](std::uint64_t granularity, std::uint64_t* cpuEntries, std::uint64_t* gpuEntries) { Quiet("GuestPageTableUsage", [&] { GuestPageTableUsage_nid_no_patch(granularity, cpuEntries, gpuEntries); }); };
    host.guest_allocations_begin = [] { return Quiet("GuestAllocationsBegin", [] { return GuestAllocations::GuestAllocationsBegin_nid_postfix(); }); };
    host.guest_allocations_add = [](void* mutation, void* pointer, std::size_t bytes, bool readable, bool writable, bool gpu) { Quiet("GuestAllocationsAdd", [&] { GuestAllocations::GuestAllocationsAdd_nid_postfix(mutation, pointer, bytes, readable, writable, gpu); }); };
    host.guest_allocations_end = [](void* mutation) { GuestAllocations::GuestAllocationsEnd_nid_postfix(mutation); };
    host.kernel_note_gpu_submit = [] { Quiet("KernelNoteGpuSubmit", [] { KernelNoteGpuSubmit_nid_no_patch(); }); };
    host.kernel_available_flexible_memory_size = [](std::size_t* size) { return Quiet("sceKernelAvailableFlexibleMemorySize", [&] { return sceKernelAvailableFlexibleMemorySize(size); }); };
    host.kernel_configured_flexible_memory_size = [](std::size_t* size) { return Quiet("sceKernelConfiguredFlexibleMemorySize", [&] { return sceKernelConfiguredFlexibleMemorySize(size); }); };
    host.kernel_read_tsc = [] { return sceKernelReadTsc(); };
    host.kernel_tsc_frequency = [] { return sceKernelGetTscFrequency(); };
    host.kernel_process_time = [] { return sceKernelGetProcessTime(); };
    host.kernel_process_time_counter = [] { return sceKernelGetProcessTimeCounter(); };
    host.kernel_pause_guest_clock = [] { KernelPauseGuestClock_nid_no_patch(); };
    host.kernel_resume_guest_clock = [] { KernelResumeGuestClock_nid_no_patch(); };
    host.run_report_set_gpu = [](std::uint32_t vendorId, const char* name, const char* driver, std::uint32_t apiVersion) { LibcRunReportSetGpu_nid_no_patch(vendorId, name, driver, apiVersion); };
    host.run_report_frame_presented = [] { LibcRunReportFramePresented_nid_no_patch(); };
    host.equeue_add_event = [](std::int64_t eq, const KytyDllEqueueEvent* event, int* result) {
        return Guard([&] {
            KernelEqueueEvent ours;
            ours.triggered = event->triggered;
            ours.deadlineNs = event->deadline_ns;
            ours.intervalNs = event->interval_ns;
            ours.event = ToOurs(event->event);
            for (std::size_t i = 0; i < event->pending_count; i++) ours.pendingEvents.push_back(ToOurs(event->pending[i]));
            if (event->filter != nullptr) {
                auto handle = std::make_shared<FilterHandle>(event->filter);
                ours.filter.data = handle.get();
                if ((event->filter_functions & KYTY_DLL_FILTER_TRIGGER) != 0) ours.filter.triggerFunc = &TriggerThroughDll;
                if ((event->filter_functions & KYTY_DLL_FILTER_RESET) != 0) ours.filter.resetFunc = &ResetThroughDll;
                if ((event->filter_functions & KYTY_DLL_FILTER_DELETE) != 0) ours.filter.deleteEventFunc = &DeleteThroughDll;
                ours.filter.owner = std::move(handle);
            }
            *result = EqueueAddEvent_nid_postfix(eq, ours);
        });
    };
    host.equeue_delete_event = [](std::int64_t eq, std::uintptr_t ident, std::int16_t filter, int* result) { return Guard([&] { *result = EqueueDeleteEvent_nid_postfix(eq, ident, filter); }); };
    host.equeue_trigger_event = [](std::int64_t eq, std::uintptr_t ident, std::int16_t filter, void* data, int* result) { return Guard([&] { *result = EqueueTriggerEvent_nid_postfix(eq, ident, filter, data); }); };
    host.equeue_pin = [](std::int64_t eq) {
        return Quiet("EqueuePin", [&]() -> KytyDllEqueue* {
            auto queue = EqueuePin_nid_postfix(eq);
            if (!queue) return nullptr;
            return reinterpret_cast<KytyDllEqueue*>(new KernelEqueueRef(std::move(queue)));
        });
    };
    host.equeue_unpin = [](KytyDllEqueue* queue) { delete reinterpret_cast<KernelEqueueRef*>(queue); };
    host.deliver_eop_interrupt = [](std::uint32_t queue, std::uint32_t context) { Quiet("AgcDriverDeliverEopInterrupt", [&] { AgcDriverDeliverEopInterrupt(queue, context); }); };
    host.standard_output_file = [] { return HostStandardOutputFile_nid_no_patch(); };
    host.standard_error_file = [] { return HostStandardErrorFile_nid_no_patch(); };
    host.await_exit = [] { LibcAwaitExit_nid_postfix(); };
    host.register_shutdown = [](void (*callback)()) { Quiet("LibcRegisterShutdown", [&] { LibcRegisterShutdown_nid_postfix(callback); }); };
    host.shutdown_requested = [] { return LibcShutdownToken_nid_postfix().stop_requested(); };
    host.terminate = [](int code) { LibcTerminate_nid_no_patch(code); };
    host.resolve_path = [](const char* path, char* resolved, std::size_t capacity) { return Quiet("ResolvePath", [&] { return CopyPath(ResolvePath_nid_no_patch(path), resolved, capacity); }); };
    host.resolve_disk_path = [](const char* path, char* resolved, std::size_t capacity) { return Quiet("ResolveDiskPath", [&] { return CopyPath(ResolveDiskPath_nid_no_patch(path), resolved, capacity); }); };
    host.raise_worker_thread_priority = [](const char* role) { Quiet("RaiseWorkerThreadPriority", [&] { AgcDriver::RaiseWorkerThreadPriority(role); }); };
    host.system_overlay_shown = [] { return Quiet("SystemOverlayShown", [] { return AgcDriver::SystemOverlayShown(); }); };
    host.system_overlay_layout = [](std::uint32_t width, std::uint32_t height, KytyDllOverlayRect* bounds, std::int32_t* scale, KytyDllOverlayRect* rects, std::size_t capacity, std::size_t* count) {
        return Quiet("SystemOverlayLayout", [&] {
            const auto layout = AgcDriver::SystemOverlayLayout(VkExtent2D{width, height});
            if (!layout) return false;
            *bounds = {layout->bounds.x, layout->bounds.y, layout->bounds.width, layout->bounds.height, layout->bounds.color};
            *scale = layout->scale;
            *count = layout->rects.size();
            for (std::size_t i = 0; i < layout->rects.size() && i < capacity; i++) {
                const auto& rect = layout->rects[i];
                rects[i] = {rect.x, rect.y, rect.width, rect.height, rect.color};
            }
            return true;
        });
    };
    host.video_reserve = [](KytyDllVideoOutput* output, const KytyDllFlipInfo* info, bool tryOnly, KytyDllFlipRequest** request) {
        return Guard([&] {
            auto& ours = *reinterpret_cast<OutputHolder*>(output)->output;
            auto reserved = tryOnly ? ours.TryReserve(ToOurs(*info)) : ours.Reserve(ToOurs(*info));
            *request = reserved ? reinterpret_cast<KytyDllFlipRequest*>(new RequestHolder{std::move(reserved)}) : nullptr;
        });
    };
    host.video_wait_for_flip_room = [](KytyDllVideoOutput* output) { return Guard([&] { reinterpret_cast<OutputHolder*>(output)->output->WaitForFlipRoom(); }); };
    host.video_capture_rendering_wait = [](KytyDllVideoOutput* output, std::uint32_t index, KytyDllRenderingWait** wait) {
        return Guard([&] { *wait = reinterpret_cast<KytyDllRenderingWait*>(new WaitHolder{reinterpret_cast<OutputHolder*>(output)->output->CaptureRenderingWait(index)}); });
    };
    host.video_fail = [](KytyDllVideoOutput* output, const char* message) { reinterpret_cast<OutputHolder*>(output)->output->Fail(Failure(message)); };
    host.video_release = [](KytyDllVideoOutput* output) { delete reinterpret_cast<OutputHolder*>(output); };
    host.rendering_wait = [](KytyDllRenderingWait* wait) { return Guard([&] { reinterpret_cast<WaitHolder*>(wait)->wait->Wait(); }); };
    host.rendering_wait_release = [](KytyDllRenderingWait* wait) { delete reinterpret_cast<WaitHolder*>(wait); };
    host.flip_source = [](KytyDllFlipRequest* request, KytyDllDisplayBuffer* source, bool* hasSource) {
        return Guard([&] {
            const auto ours = reinterpret_cast<RequestHolder*>(request)->request->Source();
            *hasSource = ours.has_value();
            if (ours) *source = ToSeam(*ours);
        });
    };
    host.flip_gpu_ready = [](KytyDllFlipRequest* request) {
        static std::atomic<std::uint64_t> frames{0};
        return Guard([&] { reinterpret_cast<RequestHolder*>(request)->request->GpuReady(std::make_shared<AgcDriver::FrameTiming>(frames.fetch_add(1, std::memory_order_relaxed) + 1)); });
    };
    host.flip_fail = [](KytyDllFlipRequest* request, const char* message) { reinterpret_cast<RequestHolder*>(request)->request->Fail(Failure(message)); };
    host.flip_release = [](KytyDllFlipRequest* request) { delete reinterpret_cast<RequestHolder*>(request); };
    host.host_report_setting = [](bool* enabled) { return Guard([&] { const char* setting = std::getenv("ANYPS5_HOST_REPORT"); *enabled = setting == nullptr || AgcDriver::ParseHostReportSetting(setting); }); };
    host.report_host = [](const void* properties, const void* driver, const void* memory, std::uint32_t subgroupSize, const char* shaderStorePath) {
        return Guard([&] {
            const auto store = std::filesystem::u8path(shaderStorePath != nullptr ? shaderStorePath : "");
            auto facts = AgcDriver::CollectHostFacts(AgcDriver::SystemHostSources([&] {
                AgcDriver::HostShaderCacheFacts cache;
                if (store.empty()) return cache;
                cache.enabled = true;
                cache.present = std::filesystem::is_regular_file(store);
                if (cache.present) cache.sizeMiB = std::filesystem::file_size(store) / (1024 * 1024);
                return cache;
            }));
            facts.gpu = AgcDriver::CollectHostGpuIdentity(*static_cast<const VkPhysicalDeviceProperties*>(properties), static_cast<const VkPhysicalDeviceDriverProperties*>(driver),
                *static_cast<const VkPhysicalDeviceMemoryProperties*>(memory), subgroupSize);
            const auto text = AgcDriver::FormatHostReport(facts) + '\n';
            std::fputs(text.c_str(), stderr);
            std::fflush(stderr);
        });
    };
    return host;
}

const KytyDllApi& Api() {
    static const KytyDllApi* api = [] {
        static const KytyDllHost host = MakeHost();
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&MakeHost), &self)) throw std::runtime_error("libSceAgcDriver: its own module is not known");
        std::wstring path(32768, L'\0');
        const auto length = GetModuleFileNameW(self, path.data(), static_cast<DWORD>(path.size()));
        if (length == 0 || length >= path.size()) throw std::runtime_error("libSceAgcDriver: its own path is not known");
        path.resize(length);
        path.resize(path.find_last_of(L"\\/") + 1);
        path += KYTY_DLL_FILE_NAME;
        const auto module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (module == nullptr) throw std::runtime_error("libSceAgcDriver: kyty_gpu.prx beside libSceAgcDriver.prx could not be loaded (error " + std::to_string(GetLastError()) + ")");
        const auto create = reinterpret_cast<KytyDllCreateFunction>(reinterpret_cast<void*>(GetProcAddress(module, KYTY_DLL_CREATE_SYMBOL)));
        if (create == nullptr) throw std::runtime_error("libSceAgcDriver: kyty_gpu.prx does not export " KYTY_DLL_CREATE_SYMBOL);
        const KytyDllApi* created = KytyShim::CreateDll(create, host);
        g_api = created;
        return created;
    }();
    return *api;
}

[[maybe_unused]] const bool g_kyty_dll_loaded = (Api(), true);

void Call(KytyDllStatus (*function)()) {
    Check(function());
}

void Note(const char* what) {
    std::fprintf(stderr, "[kyty] %s is not available in the Kyty GPU driver\n", what);
    std::fflush(stderr);
}

}

namespace AgcDriver {

void Submit(const Packet* packet, std::uint32_t queue) {
    Check(Api().submit(packet, queue));
}

std::size_t DisplayBufferSize(const DisplayBuffer& buffer) {
    const auto seam = ToSeam(buffer);
    std::size_t bytes = 0;
    Check(Api().display_buffer_size(&seam, &bytes));
    return bytes;
}

namespace GuestMemory {

void CheckRange(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable) {
    Check(Api().check_guest_memory(pointer, bytes, alignment, writable));
}

}

}

extern "C" void AgcDriverSubmitGraphics_nid_postfix(std::uint32_t* commands, std::uint32_t dwords) {
    Check(Api().submit_graphics(commands, dwords));
}

extern "C" void AgcDriverSuspendPoint_nid_postfix() {
    Call(Api().suspend_point);
}

extern "C" void AgcDriverWaitIdle_nid_postfix() {
    Call(Api().wait_idle);
}

extern "C" void AgcDriverPrestartGpu_nid_postfix() {
    Call(Api().prestart_gpu);
}

extern "C" void AgcDriverShutdown_nid_postfix() {
    if (g_api != nullptr) Call(g_api->shutdown);
}

extern "C" void AgcDriverSubmitFlip_nid_postfix(const AgcDriver::FlipInfo& info) {
    const auto seam = ToSeam(info);
    Check(Api().submit_flip(&seam));
}

extern "C" std::uint32_t AgcDriverReserveEopFlip_nid_postfix(const AgcDriver::FlipInfo& info) {
    const auto seam = ToSeam(info);
    std::uint32_t token = 0;
    Check(Api().reserve_eop_flip(&seam, &token));
    return token;
}

extern "C" std::uint32_t AgcDriverReserveEopFlipAt_nid_postfix(const AgcDriver::FlipInfo& info, std::uint32_t token) {
    const auto seam = ToSeam(info);
    std::uint32_t reserved = 0;
    Check(Api().reserve_eop_flip_at(&seam, token, &reserved));
    return reserved;
}

extern "C" void AgcDriverRegisterVideoOutput_nid_postfix(std::uint32_t handle, const std::shared_ptr<AgcDriver::IVideoOutput>& output) {
    if (output == nullptr) throw std::runtime_error("KytyShim: null video output");
    auto* holder = new OutputHolder{output};
    const auto status = Api().register_video_output(handle, reinterpret_cast<KytyDllVideoOutput*>(holder));
    if (status != KYTY_DLL_OK) DllFailed(status);
    std::lock_guard lock(g_outputs_mutex);
    g_outputs[handle] = holder;
}

extern "C" void AgcDriverUnregisterVideoOutput_nid_postfix(std::uint32_t handle, const std::shared_ptr<AgcDriver::IVideoOutput>& output) {
    OutputHolder* holder = nullptr;
    {
        std::lock_guard lock(g_outputs_mutex);
        const auto found = g_outputs.find(handle);
        if (found != g_outputs.end() && found->second->output == output) {
            holder = found->second;
            g_outputs.erase(found);
        }
    }
    if (holder == nullptr) throw std::runtime_error("KytyShim: video output registration mismatch");
    Check(Api().unregister_video_output(handle, reinterpret_cast<KytyDllVideoOutput*>(holder)));
}

extern "C" void* AgcDriverStaticTable_nid_postfix() {
    void* table = nullptr;
    Check(Api().static_table(&table));
    return table;
}

extern "C" void AgcDriverPrepareCommandWrite_nid_no_patch(std::uint64_t address, std::size_t bytes) {
    Check(Api().prepare_command_write(address, bytes));
}

extern "C" void AgcDriverNoteCommandSegment_nid_no_patch(std::uint64_t address, std::size_t bytes) {
    Check(Api().note_command_segment(address, bytes));
}

extern "C" void AgcDriverStoreGuestLabel_nid_postfix(std::uint64_t* pointer, std::uint64_t value) {
    Check(Api().store_guest_label(pointer, value));
}

extern "C" void AgcDriverLockVulkanLoader_nid_postfix() {
    Call(Api().lock_vulkan_loader);
}

extern "C" void AgcDriverUnlockVulkanLoader_nid_postfix() {
    Call(Api().unlock_vulkan_loader);
}

extern "C" std::size_t AgcDriverDisplayBufferSize_nid_postfix(const AgcDriver::DisplayBuffer& buffer) {
    return AgcDriver::DisplayBufferSize(buffer);
}

extern "C" void AgcDriverCheckGuestMemory_nid_postfix(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable) {
    AgcDriver::GuestMemory::CheckRange(pointer, bytes, alignment, writable);
}

extern "C" void AgcDriverRaiseWorkerThreadPriority_nid_postfix(const char* role) {
    AgcDriver::RaiseWorkerThreadPriority(role);
}

extern "C" void AgcDriverUseGcnShaders_nid_postfix() {
    throw std::runtime_error("PS4 (GNM) titles are not supported by the Kyty GPU driver");
}

extern "C" void AgcDriverTogglePerformanceOverlay_nid_postfix() {
    Note("the performance overlay");
}

extern "C" void AgcDriverRequestScreenshot_nid_postfix() {
    Note("the screenshot key (use the vkcount layer captures)");
}

extern "C" void AgcDriverPadStatusPoll_nid_postfix(const AgcDriver::PadStatusProvider* provider) {
    static_cast<void>(provider);
}

extern "C" void AgcDriverPadStatusShow_nid_postfix() {
    Note("the controller status overlay");
}

extern "C" bool AgcDriverTraceRecording_nid_postfix() {
    return false;
}

extern "C" AgcDriver::TraceThread* AgcDriverTraceThread_nid_postfix() {
    return nullptr;
}

extern "C" void AgcDriverRegisterShader_nid_postfix(const Shader* shader) {
    Check(Api().register_shader(shader));
}

extern "C" void AgcDriverResolveShaderAbi_nid_postfix(const Shader*, std::span<const ShaderRegister>, std::span<const ShaderRegister>) {}

extern "C" void AgcDriverResolveGraphicsStagesAbi_nid_postfix(std::span<const Shader* const>, std::span<const ShaderRegister>, std::span<const ShaderRegister>) {}

extern "C" void AgcDriverResolveGraphicsAbi_nid_postfix(const Shader*, const Shader*, std::uint32_t) {}

extern "C" AgcDriver::DriverDetail::ShaderPreparationTransaction* AgcDriverBeginShaderPreparation_nid_postfix(const Shader* const*, std::size_t) {
    return nullptr;
}

extern "C" void AgcDriverCommitShaderPreparation_nid_postfix(AgcDriver::DriverDetail::ShaderPreparationTransaction*) {}

extern "C" void AgcDriverEndShaderPreparation_nid_postfix(AgcDriver::DriverDetail::ShaderPreparationTransaction*) noexcept {}

extern "C" void AgcDriverTrackSerials_nid_postfix() {
    Call(Api().track_serials);
}

extern "C" void AgcDriverSerialProgress_nid_postfix(std::uint64_t* accepted, std::uint64_t* completed) {
    Check(Api().serial_progress(accepted, completed));
}

extern "C" bool AgcDriverSettled_nid_postfix() {
    bool settled = false;
    Check(Api().settled(&settled));
    return settled;
}

extern "C" bool AgcDriverHoldSettled_nid_postfix(std::int64_t milliseconds) {
    bool held = false;
    Check(Api().hold_settled(milliseconds, &held));
    return held;
}

extern "C" void AgcDriverReleaseSettled_nid_postfix() {
    Call(Api().release_settled);
}

extern "C" std::size_t AgcDriverAwaitedWaits_nid_postfix(AgcDriver::AwaitedWait* waits, std::size_t capacity) {
    static_assert(sizeof(KytyDllAwaitedWait) == sizeof(AgcDriver::AwaitedWait));
    std::size_t count = 0;
    Check(Api().awaited_waits(reinterpret_cast<KytyDllAwaitedWait*>(waits), capacity, &count));
    return count;
}

extern "C" void AgcDriverWaitForPipelineJournal_nid_postfix() {
    Call(Api().wait_for_pipeline_journal);
}

extern "C" std::size_t AgcDriverReplayCounters_nid_postfix(std::uint64_t* values, std::size_t capacity) {
    std::size_t count = 0;
    Check(Api().replay_counters(values, capacity, &count));
    return count;
}

extern "C" void AgcDriverTracePresent_nid_postfix(std::uint32_t width, std::uint32_t height) {
    Check(Api().trace_present(width, height));
}

extern "C" void AgcDriverAttachWindow_nid_postfix(const AgcDriver::PresentationWindow& window) {
    const auto seam = ToSeam(window);
    Check(Api().attach_window(&seam));
}

extern "C" void AgcDriverPresentBuffer_nid_postfix(const AgcDriver::PresentationWindow& window, const AgcDriver::DisplayBuffer& buffer, const AgcDriver::DisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
    const auto seamWindow = ToSeam(window);
    const auto seamBuffer = ToSeam(buffer);
    KytyDllDisplayBuffer seamPlane{};
    if (plane != nullptr) seamPlane = ToSeam(*plane);
    Check(Api().present_buffer(&seamWindow, &seamBuffer, plane != nullptr ? &seamPlane : nullptr, gpuReady, context));
}

extern "C" void AgcDriverPresentClear_nid_postfix(const AgcDriver::PresentationWindow& window, bool opaque, const AgcDriver::DisplayBuffer* plane, void (*gpuReady)(void*), void* context) {
    const auto seamWindow = ToSeam(window);
    KytyDllDisplayBuffer seamPlane{};
    if (plane != nullptr) seamPlane = ToSeam(*plane);
    Check(Api().present_clear(&seamWindow, opaque, plane != nullptr ? &seamPlane : nullptr, gpuReady, context));
}

extern "C" void AgcDriverReleaseWindow_nid_postfix(void* window) {
    Check(Api().release_window(window));
}
