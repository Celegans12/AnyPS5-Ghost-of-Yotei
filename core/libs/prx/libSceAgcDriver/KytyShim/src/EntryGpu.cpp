#include "prx/libSceAgcDriver/KytyShim/src/EntryApi.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/EntryPure.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/Replay.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/HostThread.hpp"
#include "prx/libc/include/RunReport.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/Time/include/SleepWake.hpp"

#include "common/emulatorConfig.h"
#include "common/threads.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "prx/libSceAgcDriver/Execution/include/HostReportSources.hpp"
#include "prx/libSceAgcDriver/Execution/include/HostReportVulkan.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#ifdef _WIN32
namespace {
constexpr int PROT_READ = 1;
constexpr int PROT_WRITE = 2;
}
#endif

#if defined(ANYPS5_KYTY_DLL)
namespace KytyShim {
bool DllHostReportSetting();
void DllReportHost(const void* properties, const void* driver, const void* memory, std::uint32_t subgroupSize, const char* shaderStorePath);
}
#endif

namespace KytyShim {

void StartRuntime();
void ReportNativeRayTracing(const Libs::Graphics::GraphicContext& graphics);
void RouteInterrupts(Libs::Graphics::RenderContext& renderer);
void InstallFaultChain();
void RemoveFaultChain();

}

namespace KytyShim::Present {

void SubmitCpuFlip(Libs::Graphics::GuestGpu& gpu, const AgcDriver::FlipInfo& info);

}

namespace KytyShim::Entry {

namespace {

Libs::Graphics::RenderContext* g_renderer = nullptr;

bool HostWriteWatched(std::uintptr_t address, std::size_t bytes) {
    return Libs::LibKernel::Memory::GuestRangeWatched(address, bytes);
}

void NoteRunReportGpu(const Libs::Graphics::GraphicContext& graphics) {
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(graphics.physical_device, &properties);
    std::string driverText = std::string(driver.driverName) + " " + driver.driverInfo;
    if (driver.driverName[0] == '\0' && driver.driverInfo[0] == '\0') driverText = std::to_string(properties.properties.driverVersion);
    LibcRunReportSetGpu_nid_no_patch(properties.properties.vendorID, properties.properties.deviceName, driverText.c_str(), properties.properties.apiVersion);
}

void ReportKytyHost(const Libs::Graphics::GraphicContext& graphics) {
    static std::atomic<bool> printed{false};
    if (printed.exchange(true)) return;
#if defined(ANYPS5_KYTY_DLL)
    VkPhysicalDeviceDriverProperties driverProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 deviceProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driverProperties};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(graphics.physical_device, &deviceProperties);
    driverProperties.pNext = nullptr;
    const auto store = Libs::Graphics::ShaderStorePath().u8string();
    const std::string storeText(reinterpret_cast<const char*>(store.data()), store.size());
    DllReportHost(&deviceProperties.properties, &driverProperties, &graphics.physical_device_memory_properties, graphics.subgroup_size, storeText.c_str());
#else
    auto facts = AgcDriver::CollectHostFacts(AgcDriver::SystemHostSources([] {
        AgcDriver::HostShaderCacheFacts cache;
        const auto path = Libs::Graphics::ShaderStorePath();
        if (path.empty()) return cache;
        cache.enabled = true;
        cache.present = std::filesystem::is_regular_file(path);
        if (cache.present) cache.sizeMiB = std::filesystem::file_size(path) / (1024 * 1024);
        return cache;
    }));
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties2(graphics.physical_device, &properties);
    facts.gpu = AgcDriver::CollectHostGpuIdentity(properties.properties, &driver, graphics.physical_device_memory_properties, graphics.subgroup_size);
    const auto text = AgcDriver::FormatHostReport(facts) + '\n';
    std::fputs(text.c_str(), stderr);
    std::fflush(stderr);
#endif
}

}

}

namespace Libs::Graphics {

using KytyShim::Entry::g_renderer;

void Initialize() {
    if (g_renderer != nullptr) throw std::runtime_error("kyty GPU backend: Libs::Graphics::Initialize called twice");
#if defined(ANYPS5_KYTY_DLL)
    const bool hostReport = KytyShim::DllHostReportSetting();
#else
    const char* hostReportSetting = std::getenv("ANYPS5_HOST_REPORT");
    const bool hostReport = hostReportSetting == nullptr || AgcDriver::ParseHostReportSetting(hostReportSetting);
#endif
    KytyShim::InstallFaultChain();
    GuestArena::GuestArenaSetHostWriteFilter_nid_no_patch(&KytyShim::Entry::HostWriteWatched);
    const auto width = Config::GetScreenWidth();
    const auto height = Config::GetScreenHeight();
    auto& presenter = WindowInit(width, height);
    KytyShim::ReportNativeRayTracing(presenter.Renderer().GetGraphics());
    KytyShim::Entry::NoteRunReportGpu(presenter.Renderer().GetGraphics());
    if (hostReport) KytyShim::Entry::ReportKytyHost(presenter.Renderer().GetGraphics());
    auto& video_out = VideoOut::VideoOutInit(width, height, presenter);
    g_renderer = &presenter.Renderer();
    g_renderer->InitializeGpu(&video_out);
    KytyShim::RouteInterrupts(*g_renderer);
}

void Shutdown() {
    if (g_renderer == nullptr) throw std::runtime_error("kyty GPU backend: Libs::Graphics::Shutdown without Initialize");
    g_renderer->ShutdownGpu();
    VideoOut::VideoOutShutdown();
    WindowShutdown();
    KytyShim::RemoveFaultChain();
    GuestArena::GuestArenaSetHostWriteFilter_nid_no_patch(nullptr);
    g_renderer = nullptr;
}

}

namespace KytyShim::Entry {

namespace {

constexpr std::uint32_t ComputeQueueBase = 0x20;
constexpr std::uint32_t ComputeQueueCount = 7 * 8;
constexpr std::uint64_t DriverMemoryAddress = 0xfe0000000ull + GuestArena::GuestAddressOffset;
constexpr std::size_t DriverMemoryBytes = 2u << 20u;
constexpr std::uint64_t FsTableAddress = DriverMemoryAddress + 0x40000ull;
constexpr std::uint64_t FsTableSlots = 6u;
constexpr std::uint64_t FsTableSlotBytes = 16u;
constexpr std::uint64_t FsRoutineRecords = 8u;
constexpr std::uint64_t FsRoutineRecordBytes = 16u;
constexpr std::uint64_t FsRoutineTableAddress = FsTableAddress + FsTableSlots * FsTableSlotBytes;
constexpr std::uint64_t FsRoutineAddress = FsRoutineTableAddress + FsRoutineRecords * FsRoutineRecordBytes;
constexpr std::uint32_t ScalarSetpcB64LinkS14 = 0xbe80200eu;
constexpr std::uint32_t ScalarEndProgram = 0xbf810000u;
constexpr std::uint64_t FsTableDword3 = 0x5204u;
constexpr std::uint64_t SerialLabelAddress = DriverMemoryAddress + 0x80000ull;
constexpr std::size_t FlipSlotCount = 1024;
constexpr auto StopGateTimeout = std::chrono::seconds(2);

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

struct FlipSlots {
    alignas(64) std::array<std::array<std::uint32_t, AgcDriver::FlipPacketWords>, FlipSlotCount> words{};
    std::array<std::atomic<bool>, FlipSlotCount> busy{};
    std::atomic<std::size_t> next{0};
    std::mutex mutex;
    std::condition_variable freed;
};

struct PendingEopFlip {
    std::uint32_t token = 0;
    AgcDriver::FlipInfo info{};
};

struct State {
    std::mutex startMutex;
    std::exception_ptr startFailure;
    std::mutex prestartMutex;
    std::thread prestart;
    std::exception_ptr prestartFailure;
    std::atomic<Libs::Graphics::RenderContext*> renderer{nullptr};
    std::atomic<bool> stopped{false};
    std::shared_timed_mutex gate;
    std::mutex outputsMutex;
    std::map<std::uint32_t, std::shared_ptr<AgcDriver::IVideoOutput>> outputs;
    std::mutex eopMutex;
    std::uint32_t nextEopToken = AgcDriver::EopFlipTokenBase;
    std::vector<PendingEopFlip> eopFlips;
    FlipSlots flipSlots;
    std::mutex vulkanLoader;
};

State& state() {
    static auto* instance = new State();
    return *instance;
}

void CheckStopping() {
    if (state().stopped.load(std::memory_order_acquire) || LibcShutdownToken_nid_postfix().stop_requested()) throw ProcessShutdown{};
}

void MapDriverMemory() {
    void* mapped = DirectMemoryMapHostView_nid_no_patch(DriverMemoryAddress, DriverMemoryBytes, PROT_READ | PROT_WRITE);
    if (reinterpret_cast<std::uint64_t>(mapped) != DriverMemoryAddress) throw std::runtime_error("the AGC driver memory was not mapped at its fixed address");
    auto* routineTable = reinterpret_cast<std::uint64_t*>(FsRoutineTableAddress);
    for (std::uint64_t record = 0; record < FsRoutineRecords; ++record) {
        routineTable[record * 2u] = FsRoutineAddress;
        routineTable[record * 2u + 1u] = 0u;
    }
    auto* routine = reinterpret_cast<std::uint32_t*>(FsRoutineAddress);
    routine[0] = ScalarSetpcB64LinkS14;
    routine[1] = ScalarEndProgram;
    auto* slot = reinterpret_cast<std::uint64_t*>(FsTableAddress);
    slot[0] = (FsRoutineTableAddress & 0xffffffffffffull) | 16ull << 48u;
    slot[1] = FsTableDword3 << 32u | FsRoutineRecords;
    GuestAllocations::Mutation mutation;
    mutation.Add(mapped, DriverMemoryBytes, true, true, true);
}

void EnsureDriverMemory() {
    static std::once_flag once;
    std::call_once(once, MapDriverMemory);
}

void StopGpu() {
    auto& s = state();
    JoinPrestart(false);
    std::unique_lock gate(s.gate, std::defer_lock);
    const bool exclusive = gate.try_lock_for(StopGateTimeout);
    s.stopped.store(true, std::memory_order_release);
    std::lock_guard start(s.startMutex);
    if (s.renderer.load(std::memory_order_acquire) == nullptr) return;
    if (!exclusive) {
        std::fprintf(stderr, "[kyty] GPU shutdown: a submission was still inside the GPU 2 s into the shutdown; leaving the Kyty GPU running until the process exits\n");
        std::fflush(stderr);
        return;
    }
    std::exception_ptr failure;
    auto* renderer = s.renderer.load(std::memory_order_acquire);
    auto thread = Libc::StartHostThread([&failure, renderer] {
        try {
            renderer->ShutdownGpu();
        } catch (...) {
            failure = std::current_exception();
        }
    });
    thread.join();
    if (failure) std::rethrow_exception(failure);
}

void ReportPrestartFailure() {
    auto& s = state();
    std::lock_guard lock(s.prestartMutex);
    if (s.prestart.joinable() || !s.prestartFailure) return;
    try {
        std::rethrow_exception(s.prestartFailure);
    } catch (const ProcessShutdown&) {
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[kyty] GPU start failed: %s\n", error.what());
        std::fflush(stderr);
    } catch (...) {
        std::fprintf(stderr, "[kyty] GPU start failed\n");
        std::fflush(stderr);
    }
}

void StopGpuAtExit() {
    try {
        StopGpu();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[kyty] GPU shutdown failed: %s\n", error.what());
        std::fflush(stderr);
    }
    ReportPrestartFailure();
}

void RegisterStopAtExit() {
    static std::once_flag registered;
    std::call_once(registered, [] { LibcRegisterShutdown_nid_postfix(&StopGpuAtExit); });
}

std::uint32_t* AcquireFlipSlot() {
    auto& slots = state().flipSlots;
    for (;;) {
        const auto start = slots.next.fetch_add(1, std::memory_order_relaxed);
        for (std::size_t probe = 0; probe < FlipSlotCount; ++probe) {
            const auto index = (start + probe) % FlipSlotCount;
            bool expected = false;
            if (slots.busy[index].compare_exchange_strong(expected, true, std::memory_order_acq_rel)) return slots.words[index].data();
        }
        CheckStopping();
        std::unique_lock lock(slots.mutex);
        slots.freed.wait_for(lock, std::chrono::milliseconds(100));
    }
}

void SubmitWords(Libs::Graphics::GuestGpu& gpu, std::span<const std::uint32_t> commands, std::uint32_t queue) {
    if (commands.empty()) return;
    if (queue == 0) gpu.Submit(commands);
    else gpu.SubmitCompute(queue, commands);
}

void SubmitFlipPacket(Libs::Graphics::GuestGpu& gpu, const AgcDriver::FlipInfo& info) {
    auto* words = AcquireFlipSlot();
    const auto packet = EncodeFlipPacket(info);
    std::copy(packet.begin(), packet.end(), words);
    gpu.Submit(std::span<const std::uint32_t>(words, packet.size()));
}

void RequireOutput(std::uint32_t handle, const char* reason) {
    auto& s = state();
    std::lock_guard lock(s.outputsMutex);
    require(s.outputs.contains(handle), reason);
}

void Submit(std::span<const std::uint32_t> commands, std::uint32_t queue) {
    auto& s = state();
    std::shared_lock gate(s.gate);
    CheckStopping();
    std::vector<EopFlipPoint> points;
    std::vector<AgcDriver::FlipInfo> flips;
    {
        std::lock_guard lock(s.eopMutex);
        if (!s.eopFlips.empty()) {
            std::vector<std::uint32_t> tokens;
            tokens.reserve(s.eopFlips.size());
            for (const auto& flip : s.eopFlips) tokens.push_back(flip.token);
            points = FindEopFlipPoints(commands, tokens);
            require(queue == 0 || points.empty(), "EOP flips on async queues are not implemented");
            if (points.empty()) {
                if (const auto inner = FindIndirectEopToken(commands, tokens)) {
                    char text[200];
                    std::snprintf(text, sizeof(text), "kyty GPU backend: the end-of-pipe release of EOP flip token 0x%08x is inside an indirect buffer; only releases in the submitted command buffer itself are supported", *inner);
                    throw std::runtime_error(text);
                }
            }
            for (const auto& point : points) {
                const auto found = std::find_if(s.eopFlips.begin(), s.eopFlips.end(), [&](const PendingEopFlip& flip) { return flip.token == point.token; });
                flips.push_back(found->info);
                s.eopFlips.erase(found);
            }
        }
    }
    auto& gpu = Gpu();
    std::size_t begin = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        SubmitWords(gpu, commands.subspan(begin, points[i].end - begin), queue);
        SubmitFlipPacket(gpu, flips[i]);
        begin = points[i].end;
    }
    SubmitWords(gpu, commands.subspan(begin), queue);
    if (queue == 0 && (!points.empty() || HasFlipPacket(commands))) gpu.NoteFrameSubmitted();
    KytyShim::Replay::NoteSubmission(gpu, queue);
}

}

Libs::Graphics::RenderContext& Renderer() {
    auto& s = state();
    if (auto* renderer = s.renderer.load(std::memory_order_acquire)) return *renderer;
    std::lock_guard lock(s.startMutex);
    if (auto* renderer = s.renderer.load(std::memory_order_acquire)) return *renderer;
    if (s.startFailure) std::rethrow_exception(s.startFailure);
    CheckStopping();
    EnsureDriverMemory();
    RegisterStopAtExit();
    std::exception_ptr failure;
    Libs::Graphics::RenderContext* started = nullptr;
    auto thread = Libc::StartHostThread([&failure, &started] {
        try {
            KytyShim::StartRuntime();
            Common::InitializeThreads();
            Libs::Graphics::Initialize();
            started = g_renderer;
        } catch (...) {
            failure = std::current_exception();
        }
    });
    thread.join();
    if (failure) {
        s.startFailure = failure;
        std::rethrow_exception(failure);
    }
    s.renderer.store(started, std::memory_order_release);
    return *started;
}

Libs::Graphics::GuestGpu& Gpu() {
    return Renderer().GetGpu();
}

bool GpuStarted() {
    return state().renderer.load(std::memory_order_acquire) != nullptr;
}

namespace {

bool PrestartEnabled() {
    const char* value = std::getenv("ANYPS5_KYTY_PRESTART");
    if (value == nullptr || *value == '\0') return true;
    const std::string_view text(value);
    if (text != "0" && text != "1") throw std::runtime_error(std::string("KytyShim: ANYPS5_KYTY_PRESTART=") + value + " is neither 0 nor 1");
    return text == "1";
}

}

void PrestartGpu() {
    if (!PrestartEnabled()) return;
    auto& s = state();
    std::lock_guard lock(s.prestartMutex);
    if (s.prestart.joinable() || s.prestartFailure || GpuStarted()) return;
    CheckStopping();
    RegisterStopAtExit();
    s.prestart = Libc::StartHostThread([&s] {
        try {
            Renderer();
        } catch (...) {
            s.prestartFailure = std::current_exception();
        }
    });
}

void JoinPrestart(bool rethrow) {
    auto& s = state();
    std::lock_guard lock(s.prestartMutex);
    if (s.prestart.joinable()) {
        if (s.prestart.get_id() == std::this_thread::get_id()) throw std::runtime_error("kyty GPU backend: the GPU start thread cannot wait for itself");
        s.prestart.join();
    }
    if (rethrow && s.prestartFailure) std::rethrow_exception(s.prestartFailure);
}

std::shared_ptr<AgcDriver::IVideoOutput> VideoOutput(std::uint32_t handle) {
    auto& s = state();
    std::lock_guard lock(s.outputsMutex);
    const auto found = s.outputs.find(handle);
    return found == s.outputs.end() ? nullptr : found->second;
}

bool OwnsFlipPacket(const std::uint32_t* packet) {
    const auto& words = state().flipSlots.words;
    const auto* first = words.front().data();
    const auto* end = words.back().data() + AgcDriver::FlipPacketWords;
    return packet >= first && packet < end && (packet - first) % AgcDriver::FlipPacketWords == 0;
}

std::uint64_t* SerialLabels() {
    EnsureDriverMemory();
    return reinterpret_cast<std::uint64_t*>(SerialLabelAddress);
}

void ReleaseFlipPacket(const std::uint32_t* packet) {
    auto& slots = state().flipSlots;
    const auto index = static_cast<std::size_t>(packet - slots.words.front().data()) / AgcDriver::FlipPacketWords;
    slots.busy[index].store(false, std::memory_order_release);
    std::lock_guard lock(slots.mutex);
    slots.freed.notify_one();
}

}

namespace KytyShim::Entry {

void CheckGuestRange(const void* pointer, std::size_t bytes, std::size_t alignment) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (address != 0 && alignment != 0 && address % alignment == 0 && bytes != 0 && Libs::LibKernel::Memory::IsFullyMapped(address, bytes)) return;
    CheckMappedRange(pointer, bytes, alignment);
}

namespace {

void SubmitDescriptor(const Packet& descriptor, std::uint32_t queue) {
    if (descriptor.flags != 0) throw std::runtime_error("nonzero submission flags are not implemented");
    if (queue != 0 && (queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount)) {
        char text[160];
        std::snprintf(text, sizeof(text), "kyty GPU backend: queue 0x%x is outside Kyty's graphics queue and its 56 compute queues (0x20-0x57)", queue);
        throw std::runtime_error(text);
    }
    if (descriptor.dw_num == 0) return;
    if (descriptor.addr == nullptr) throw std::runtime_error("null command buffer");
    CheckGuestRange(descriptor.addr, std::size_t{descriptor.dw_num} * sizeof(std::uint32_t), alignof(std::uint32_t));
    Submit(std::span<const std::uint32_t>(descriptor.addr, descriptor.dw_num), queue);
}

}

}

namespace AgcDriver {

void Submit(const Packet* packet, std::uint32_t queue) try {
    KernelNoteGpuSubmit_nid_no_patch();
    KytyShim::Entry::CheckGuestRange(packet, sizeof(Packet), alignof(Packet));
    KytyShim::Entry::SubmitDescriptor(*packet, queue);
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

}

extern "C" void AgcDriverSubmitGraphics_nid_postfix(std::uint32_t* commands, std::uint32_t dwords) try {
    KernelNoteGpuSubmit_nid_no_patch();
    KytyShim::Entry::SubmitDescriptor(Packet{commands, dwords, 0, {}}, 0);
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

extern "C" void AgcDriverSuspendPoint_nid_postfix() try {
    auto& s = KytyShim::Entry::state();
    std::shared_lock gate(s.gate);
    KytyShim::Entry::CheckStopping();
    auto& gpu = KytyShim::Entry::Gpu();
    gpu.SuspendPoint();
    KytyShim::Replay::NoteSubmission(gpu, 0);
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

extern "C" void AgcDriverWaitIdle_nid_postfix() try {
    KytyShim::Replay::RequireNotHeld("AgcDriverWaitIdle");
    auto& s = KytyShim::Entry::state();
    std::shared_lock gate(s.gate);
    KytyShim::Entry::CheckStopping();
    if (!KytyShim::Entry::GpuStarted()) return;
    auto& renderer = KytyShim::Entry::Renderer();
    auto& gpu = renderer.GetGpu();
    gpu.WaitForIdle();
    gpu.SendCommandSync([&renderer] {
        auto& scheduler = renderer.GetCommandScheduler();
        if (scheduler.Active()) scheduler.Finish();
    });
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

extern "C" void AgcDriverPrestartGpu_nid_postfix() try {
    KytyShim::Entry::PrestartGpu();
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

extern "C" void AgcDriverShutdown_nid_postfix() {
    KytyShim::Entry::StopGpu();
}

extern "C" void AgcDriverSubmitFlip_nid_postfix(const AgcDriver::FlipInfo& info) try {
    auto& s = KytyShim::Entry::state();
    std::shared_lock gate(s.gate);
    KytyShim::Entry::CheckStopping();
    KytyShim::Entry::RequireOutput(info.handle, "flip references an unregistered video output");
    auto& gpu = KytyShim::Entry::Gpu();
    KytyShim::Present::SubmitCpuFlip(gpu, info);
    KytyShim::Replay::NoteSubmission(gpu, 0);
} catch (const ProcessShutdown&) {
    LibcAwaitExit_nid_postfix();
}

extern "C" std::uint32_t AgcDriverReserveEopFlip_nid_postfix(const AgcDriver::FlipInfo& info) {
    auto& s = KytyShim::Entry::state();
    KytyShim::Entry::CheckStopping();
    KytyShim::Entry::RequireOutput(info.handle, "EOP flip references an unregistered video output");
    std::lock_guard lock(s.eopMutex);
    if (s.eopFlips.size() >= KytyShim::Entry::PendingEopFlipLimit) return 0;
    KytyShim::Entry::require(s.nextEopToken <= AgcDriver::EopFlipTokenLimit, "EOP flip token space exhausted");
    const auto token = s.nextEopToken++;
    s.eopFlips.push_back({token, info});
    return token;
}

extern "C" std::uint32_t AgcDriverReserveEopFlipAt_nid_postfix(const AgcDriver::FlipInfo& info, std::uint32_t token) {
    auto& s = KytyShim::Entry::state();
    KytyShim::Entry::CheckStopping();
    KytyShim::Entry::require(token >= AgcDriver::EopFlipTokenBase && token <= AgcDriver::EopFlipTokenLimit, "EOP flip token is out of range");
    KytyShim::Entry::RequireOutput(info.handle, "EOP flip references an unregistered video output");
    std::lock_guard lock(s.eopMutex);
    const auto found = std::find_if(s.eopFlips.begin(), s.eopFlips.end(), [token](const KytyShim::Entry::PendingEopFlip& flip) { return flip.token == token; });
    if (found != s.eopFlips.end()) {
        const auto& reserved = found->info;
        KytyShim::Entry::require(reserved.handle == info.handle && reserved.index == info.index && reserved.mode == info.mode && reserved.argument == info.argument, "EOP flip token is reserved for a different flip");
        throw AgcDriver::EopFlipAlreadyReserved{};
    }
    if (s.eopFlips.size() >= KytyShim::Entry::PendingEopFlipLimit) return 0;
    s.eopFlips.push_back({token, info});
    s.nextEopToken = std::max(s.nextEopToken, token + 1);
    return token;
}

extern "C" void AgcDriverRegisterVideoOutput_nid_postfix(std::uint32_t handle, const std::shared_ptr<AgcDriver::IVideoOutput>& output) {
    KytyShim::Entry::require(output != nullptr, "null video output");
    KytyShim::Entry::CheckStopping();
    auto& s = KytyShim::Entry::state();
    std::lock_guard lock(s.outputsMutex);
    KytyShim::Entry::require(s.outputs.emplace(handle, output).second, "video output already registered");
}

extern "C" void AgcDriverUnregisterVideoOutput_nid_postfix(std::uint32_t handle, const std::shared_ptr<AgcDriver::IVideoOutput>& output) {
    auto& s = KytyShim::Entry::state();
    {
        std::lock_guard lock(s.outputsMutex);
        const auto found = s.outputs.find(handle);
        KytyShim::Entry::require(found != s.outputs.end() && found->second == output, "video output registration mismatch");
        s.outputs.erase(found);
    }
    std::lock_guard lock(s.eopMutex);
    std::erase_if(s.eopFlips, [handle](const KytyShim::Entry::PendingEopFlip& flip) { return flip.info.handle == handle; });
}

extern "C" void* AgcDriverStaticTable_nid_postfix() {
    KytyShim::Entry::EnsureDriverMemory();
    return reinterpret_cast<void*>(KytyShim::Entry::FsTableAddress);
}

extern "C" void AgcDriverPrepareCommandWrite_nid_no_patch(std::uint64_t address, std::size_t bytes) {
    static_cast<void>(address);
    static_cast<void>(bytes);
}

extern "C" void AgcDriverNoteCommandSegment_nid_no_patch(std::uint64_t address, std::size_t bytes) {
    if (auto* renderer = KytyShim::Entry::state().renderer.load(std::memory_order_acquire)) {
        renderer->GetBufferCache().NoteCommandSegment(address, bytes);
    }
}

extern "C" void AgcDriverStoreGuestLabel_nid_postfix(std::uint64_t* pointer, std::uint64_t value) {
    KytyShim::Entry::CheckMappedRange(pointer, sizeof(*pointer), std::atomic_ref<std::uint64_t>::required_alignment);
    const auto address = reinterpret_cast<std::uint64_t>(pointer);
    {
        const GuestArena::HostWrite prepared(pointer, sizeof(*pointer));
        std::atomic_ref(*pointer).store(value, std::memory_order_release);
    }
    Libs::LibKernel::Memory::NoteEmulatorWrite(address, sizeof(*pointer));
}

extern "C" void AgcDriverLockVulkanLoader_nid_postfix() {
    KytyShim::Entry::state().vulkanLoader.lock();
}

extern "C" void AgcDriverUnlockVulkanLoader_nid_postfix() {
    KytyShim::Entry::state().vulkanLoader.unlock();
}
