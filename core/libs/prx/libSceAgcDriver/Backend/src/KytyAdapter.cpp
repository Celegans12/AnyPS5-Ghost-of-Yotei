#include "prx/libSceAgcDriver/Backend/include/KytyAdapter.hpp"
#include "prx/libSceAgcDriver/Backend/include/GpuBackend.h"
#include "prx/libSceAgcDriver/Execution/include/AsyncLog.hpp"
#include "prx/libSceAgcDriver/Execution/include/FrameDumpIo.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libc/include/GpuBackendSwitch.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/Shutdown.hpp"
#include "prx/libkernel/AppMetadata/include/AppMetadata.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "prx/libkernel/Time/include/SleepWake.hpp"
#include "prx/libkernel/Time/include/Time.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

extern "C" std::uint64_t APS5_VABI sceKernelReadTsc(void);
extern "C" bool IsGuestPthread_nid_no_patch();

namespace AgcDriver::Kyty {
namespace {

constexpr std::uint64_t DriverBlockAddress = 0xfe0000000ull + GuestArena::GuestAddressOffset;
constexpr std::size_t DriverBlockBytes = 2u << 20u;
constexpr std::uint64_t DriverTableOffset = 0x40000u;
constexpr int DriverBlockMemoryType = 12;
constexpr int DriverBlockProtection = 0x33;
constexpr std::int16_t EvfiltGraphicsCore = -14;

struct Counts {
    std::atomic<std::uint64_t> submits{0};
    std::atomic<std::uint64_t> submitDwords{0};
    std::atomic<std::uint64_t> queues{0};
    std::atomic<std::uint64_t> cpuFlips{0};
    std::atomic<std::uint64_t> flipReservations{0};
    std::atomic<std::uint64_t> flipsFull{0};
    std::atomic<std::uint64_t> flipsReady{0};
    std::atomic<std::uint64_t> flipsCancelled{0};
    std::atomic<std::uint64_t> renderingWaits{0};
    std::atomic<std::uint64_t> displayBuffers{0};
    std::atomic<std::uint64_t> interrupts{0};
    std::atomic<std::uint64_t> interruptsDelivered{0};
    std::atomic<std::uint64_t> eqAdds{0};
    std::atomic<std::uint64_t> eqDeletes{0};
    std::atomic<std::uint64_t> shaders{0};
    std::atomic<std::uint64_t> maps{0};
    std::atomic<std::uint64_t> unmaps{0};
    std::atomic<std::uint64_t> protects{0};
    std::atomic<std::uint64_t> protectGuest{0};
    std::atomic<std::uint64_t> hostWrites{0};
    std::atomic<std::uint64_t> hostReads{0};
    std::atomic<std::uint64_t> faults{0};
    std::atomic<std::uint64_t> faultsServed{0};
    std::atomic<std::uint64_t> presents{0};
    std::atomic<std::uint64_t> clears{0};
    std::atomic<std::uint64_t> dumps{0};
    std::atomic<std::uint64_t> suspends{0};
    std::atomic<std::uint64_t> idles{0};
    std::atomic<std::uint64_t> labelStores{0};
    std::atomic<std::uint64_t> checks{0};
};

struct WindowCopy {
    void* context = nullptr;
    std::vector<std::string> names;
    std::vector<const char*> extensions;
    VkSurfaceKHR (*createSurface)(void*, VkInstance) = nullptr;
    void (*drawableSize)(void*, std::uint32_t*, std::uint32_t*) = nullptr;
    Aps5GpuWindow window{};
};

struct EqRegistration {
    KernelEqueue eq;
    int id;
};

void Print(const char* format, auto... args) {
    std::fprintf(stderr, format, args...);
    std::fflush(stderr);
}

enum class PendingKind { Unmap, Map, Shader };

struct PendingChange {
    PendingKind kind = PendingKind::Map;
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    std::uint32_t protection = 0;
    std::uint64_t directOffset = 0;
    Aps5GpuShaderInfo shader{};
};

enum class AbSwitch : std::uint32_t { ProtectSrw, LeanCounters, Count };

constexpr std::array<const char*, static_cast<std::size_t>(AbSwitch::Count)> AbSwitchNames{"protect-srw", "lean-counters"};

std::array<std::atomic<bool>, static_cast<std::size_t>(AbSwitch::Count)> g_abSwitches{};

bool AbOn(AbSwitch which) {
    return g_abSwitches[static_cast<std::size_t>(which)].load(std::memory_order_relaxed);
}

std::uint64_t EnvNumber(const char* name, std::uint64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    std::uint64_t parsed = 0;
    const auto end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, parsed);
    if (result.ec != std::errc{} || result.ptr != end) throw std::invalid_argument(std::string(name) + " must be a whole number");
    return parsed;
}

std::vector<AbSwitch> ParseAbSwitches(const char* variable, char separator) {
    std::vector<AbSwitch> list;
    const char* value = std::getenv(variable);
    if (value == nullptr) return list;
    const std::string text(value);
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = std::min(text.find(separator, begin), text.size());
        const auto item = text.substr(begin, end - begin);
        begin = end + 1;
        if (item.empty()) continue;
        const auto found = std::find_if(AbSwitchNames.begin(), AbSwitchNames.end(), [&](const char* known) { return item == known; });
        if (found == AbSwitchNames.end()) throw std::invalid_argument(std::string(variable) + ": unknown adapter switch '" + item + "' (known: protect-srw, lean-counters)");
        list.push_back(static_cast<AbSwitch>(found - AbSwitchNames.begin()));
    }
    return list;
}

class AdapterAb {
public:
    void Configure() {
        for (const auto which : ParseAbSwitches("APS5_ADAPTER_SWITCHES", ',')) {
            g_abSwitches[static_cast<std::size_t>(which)].store(true, std::memory_order_relaxed);
            fixed += std::string(fixed.empty() ? "" : ",") + AbSwitchNames[static_cast<std::size_t>(which)];
        }
        if (!fixed.empty()) Print("[kyty-ab] adapter switches on for the whole run: %s\n", fixed.c_str());
        tested = ParseAbSwitches("APS5_ADAPTER_AB", '+');
        if (tested.empty()) return;
        period = std::chrono::seconds(EnvNumber("APS5_ADAPTER_AB_PERIOD_S", 60));
        startPresent = EnvNumber("APS5_ADAPTER_AB_START", 1500);
        warmup = EnvNumber("APS5_ADAPTER_AB_WARMUP", 20);
        for (const auto which : tested) label += std::string(label.empty() ? "" : "+") + AbSwitchNames[static_cast<std::size_t>(which)];
        Print("[kyty-ab] in-run A/B of %s: phases of %lld s from present %llu, the first %llu presents of each phase not measured, order off/on then on/off\n", label.c_str(), static_cast<long long>(period.count()), static_cast<unsigned long long>(startPresent), static_cast<unsigned long long>(warmup));
    }

    void AtPresent(std::uint64_t present) {
        if (tested.empty() || present < startPresent) return;
        const auto now = std::chrono::steady_clock::now();
        if (!running) {
            running = true;
            Begin(now);
            return;
        }
        ++phasePresents;
        if (phasePresents == warmup) measureStart = now;
        if (now - phaseStart < period) return;
        const double seconds = std::chrono::duration<double>(now - measureStart).count();
        const auto measured = phasePresents > warmup ? phasePresents - warmup : 0;
        Print("[kyty-ab] cycle=%u phase=%s presents=%llu seconds=%.3f rate=%.3f present=%llu\n", cycle, on ? label.c_str() : "off", static_cast<unsigned long long>(measured), seconds, seconds > 0 ? static_cast<double>(measured) * 10.0 / seconds : 0.0, static_cast<unsigned long long>(present));
        if (++phaseInCycle == 2) {
            phaseInCycle = 0;
            ++cycle;
        }
        Begin(now);
    }

private:
    void Begin(std::chrono::steady_clock::time_point now) {
        on = (phaseInCycle == 1) != (cycle % 2 == 1);
        for (const auto which : tested) g_abSwitches[static_cast<std::size_t>(which)].store(on, std::memory_order_relaxed);
        phaseStart = now;
        measureStart = now;
        phasePresents = 0;
    }

    std::string fixed;
    std::string label;
    std::vector<AbSwitch> tested;
    std::chrono::steady_clock::duration period{};
    std::uint64_t startPresent = 0;
    std::uint64_t warmup = 0;
    bool running = false;
    bool on = false;
    std::uint32_t cycle = 0;
    std::uint32_t phaseInCycle = 0;
    std::uint64_t phasePresents = 0;
    std::chrono::steady_clock::time_point phaseStart{};
    std::chrono::steady_clock::time_point measureStart{};
};

AdapterAb& Ab() {
    static auto* ab = new AdapterAb;
    return *ab;
}

std::uint32_t DumpScale() {
    static const std::uint32_t scale = [] {
        const char* value = std::getenv("APS5_DUMP_SCALE");
        if (value == nullptr) return std::uint32_t{4};
        std::uint32_t parsed = 0;
        const auto end = value + std::strlen(value);
        const auto result = std::from_chars(value, end, parsed);
        if (result.ec != std::errc{} || result.ptr != end || parsed == 0 || parsed > 16384) throw std::invalid_argument("APS5_DUMP_SCALE must be between 1 and 16384");
        return parsed;
    }();
    return scale;
}

int CreateBackend(const Aps5GpuHost& host, const Aps5GpuBackendApi** api, Aps5GpuBackend** backend, std::string& source);

class Adapter;
std::atomic<Adapter*> g_adapter{nullptr};
std::atomic<void*> g_selfCheckPage{nullptr};
std::atomic<bool> g_selfCheckServed{false};
thread_local int g_changeDepth = 0;

class Adapter {
public:
    static Adapter& Get() {
        static auto* adapter = new Adapter;
        return *adapter;
    }

    Counts counts;
    Aps5GpuHost host{};
    const Aps5GpuBackendApi* api = nullptr;
    Aps5GpuBackend* backend = nullptr;
    std::atomic<bool> created{false};
    std::shared_mutex protectLock;
    SRWLOCK protectSrw = SRWLOCK_INIT;
    std::atomic<std::uint64_t> generation{1};
    FrameDumpOutput frameDumps;

    bool Created() const {
        return created.load(std::memory_order_acquire);
    }

    void EnsureCreated() {
        if (Created()) return;
        std::lock_guard lock(pendingMutex);
        if (created.load(std::memory_order_relaxed)) return;
        if (CreateBackend(host, &api, &backend, backendSource) != APS5_GPU_OK || api == nullptr || backend == nullptr) throw std::runtime_error("the GPU backend could not be created");
        if (api->abi_version != APS5_GPU_BACKEND_ABI_VERSION || api->size < sizeof(Aps5GpuBackendApi)) throw std::runtime_error("the GPU backend has another interface version");
        for (const auto& change : pending) Send(change);
        const auto replayed = pending.size();
        pending.clear();
        pending.shrink_to_fit();
        created.store(true, std::memory_order_release);
        Print("[kyty] backend '%s' created at its first use (%s); %zu earlier memory and shader notifications replayed\n", api->name, backendSource.c_str(), replayed);
    }

    void Deliver(const PendingChange& change) {
        if (!Created()) {
            std::lock_guard lock(pendingMutex);
            if (!created.load(std::memory_order_relaxed)) {
                pending.push_back(change);
                return;
            }
        }
        Send(change);
    }

    void EnsureStarted() {
        EnsureCreated();
        std::call_once(started, [this] {
            if (api->start(backend) != APS5_GPU_OK) throw std::runtime_error("the GPU backend did not start");
            Print("[kyty] backend '%s' started\n", api->name);
            reporter = std::thread([this] { reportLoop(); });
            reporter.detach();
            LibcRegisterShutdown_nid_postfix([] { Adapter::Get().Shutdown(); });
        });
    }

    void Shutdown() {
        if (!Created() || stopped.exchange(true)) return;
        reportCounts("shutdown");
        api->shutdown(backend);
    }

    void Send(const PendingChange& change) {
        switch (change.kind) {
        case PendingKind::Unmap:
            api->memory_unmap(backend, change.address, change.bytes);
            break;
        case PendingKind::Map:
            api->memory_map(backend, change.address, change.bytes, change.protection, change.directOffset);
            break;
        case PendingKind::Shader:
            if (api->register_shader(backend, &change.shader) != APS5_GPU_OK) throw std::runtime_error("the GPU backend refused a shader");
            break;
        }
    }

    std::string backendSource;

    std::shared_ptr<IVideoOutput> Output(std::uint32_t handle) {
        std::lock_guard lock(outputsMutex);
        const auto found = outputs.find(handle);
        if (found == outputs.end()) return {};
        return found->second;
    }

    void AddOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
        std::lock_guard lock(outputsMutex);
        outputs[handle] = output;
    }

    void RemoveOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
        std::lock_guard lock(outputsMutex);
        const auto found = outputs.find(handle);
        if (found != outputs.end() && found->second == output) outputs.erase(found);
    }

    std::uint64_t KeepFlip(std::shared_ptr<IFlipRequest> request) {
        std::lock_guard lock(flipsMutex);
        const auto token = ++nextFlipToken;
        flips.emplace(token, std::move(request));
        return token;
    }

    std::shared_ptr<IFlipRequest> TakeFlip(std::uint64_t token) {
        std::lock_guard lock(flipsMutex);
        const auto found = flips.find(token);
        if (found == flips.end()) return {};
        auto request = std::move(found->second);
        flips.erase(found);
        return request;
    }

    std::uint64_t NextFrameSerial() {
        return ++frameSerial;
    }

    void AddRegistration(KernelEqueue eq, int id) {
        std::lock_guard lock(eqMutex);
        if (std::none_of(registrations.begin(), registrations.end(), [&](const EqRegistration& r) { return r.eq == eq && r.id == id; })) registrations.push_back({eq, id});
    }

    void RemoveRegistration(KernelEqueue eq, int id) {
        std::lock_guard lock(eqMutex);
        std::erase_if(registrations, [&](const EqRegistration& r) { return r.eq == eq && r.id == id; });
    }

    void RaiseInterrupt(std::uint32_t eventId, std::uint32_t contextId) {
        counts.interrupts.fetch_add(1, std::memory_order_relaxed);
        std::array<EqRegistration, 16> local{};
        std::size_t count = 0;
        {
            std::lock_guard lock(eqMutex);
            for (const auto& registration : registrations) {
                if (registration.id != static_cast<int>(eventId)) continue;
                if (count == local.size()) throw std::runtime_error("more than 16 event queues wait for one GPU interrupt");
                local[count++] = registration;
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            const auto result = EqueueTriggerEvent_nid_postfix(local[i].eq, static_cast<std::uintptr_t>(local[i].id), EvfiltGraphicsCore, reinterpret_cast<void*>(static_cast<std::uintptr_t>(contextId)));
            if (result == SCE_KERNEL_ERROR_EBADF || result == SCE_KERNEL_ERROR_ENOENT) {
                RemoveRegistration(local[i].eq, local[i].id);
                continue;
            }
            if (result != EQUEUE_OK) throw std::runtime_error("GPU interrupt delivery to an event queue failed");
            if (counts.interruptsDelivered.fetch_add(1, std::memory_order_relaxed) == 0) Print("[kyty] first GPU interrupt delivered to a guest event queue: event id 0x%x, context id 0x%x\n", eventId, contextId);
        }
    }

    WindowCopy& Window(const PresentationWindow& source) {
        std::lock_guard lock(windowsMutex);
        auto& copy = windows[source.context];
        if (!copy) {
            copy = std::make_unique<WindowCopy>();
            copy->context = source.context;
            for (const auto* name : source.extensions) copy->names.emplace_back(name);
            for (const auto& name : copy->names) copy->extensions.push_back(name.c_str());
            copy->createSurface = source.createSurface;
            copy->drawableSize = source.getDrawableSize;
            copy->window.context = copy.get();
            copy->window.instance_extensions = copy->extensions.data();
            copy->window.instance_extension_count = static_cast<std::uint32_t>(copy->extensions.size());
            copy->window.create_surface = [](void* context, void* instance) -> std::uint64_t {
                auto* window = static_cast<WindowCopy*>(context);
                return reinterpret_cast<std::uint64_t>(window->createSurface(window->context, static_cast<VkInstance>(instance)));
            };
            copy->window.drawable_size = [](void* context, std::uint32_t* width, std::uint32_t* height) {
                auto* window = static_cast<WindowCopy*>(context);
                window->drawableSize(window->context, width, height);
            };
        }
        copy->window.width = source.width;
        copy->window.height = source.height;
        return *copy;
    }

    std::unique_ptr<WindowCopy> TakeWindow(void* context) {
        std::lock_guard lock(windowsMutex);
        const auto found = windows.find(context);
        if (found == windows.end()) return {};
        auto window = std::move(found->second);
        windows.erase(found);
        return window;
    }

    int NextDumpIndex() {
        static const int limit = [] {
            const char* value = std::getenv("APS5_DUMP_FRAMES");
            return value ? std::atoi(value) : 0;
        }();
        static const int every = [] {
            const char* value = std::getenv("APS5_DUMP_FRAMES_EVERY");
            return value ? std::max(std::atoi(value), 1) : 1;
        }();
        std::lock_guard lock(dumpMutex);
        if (!frameDumps.Enabled_nid_no_patch() || dumped >= limit) return -1;
        if (++dumpPresents % static_cast<std::uint64_t>(every) != 0) return -1;
        return dumped++;
    }

    void NotePresent() {
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        if (!profile) return;
        std::lock_guard lock(presentMutex);
        const auto now = std::chrono::steady_clock::now();
        if (now - lastPresentReport < std::chrono::seconds(10)) return;
        lastPresentReport = now;
        const auto total = counts.presents.load() + counts.clears.load();
        Print("[present] %llu presents over 10 s (kyty backend '%s'; %llu clears in total)\n", static_cast<unsigned long long>(total - lastPresentTotal), api->name, static_cast<unsigned long long>(counts.clears.load()));
        lastPresentTotal = total;
    }

    void* StaticTable() {
        std::call_once(driverBlock, [] {
            const auto physical = DirectMemoryMapSystemBlock_nid_no_patch(DriverBlockAddress, DriverBlockBytes, DriverBlockMemoryType, DriverBlockProtection);
            std::memset(reinterpret_cast<void*>(DriverBlockAddress), 0, DriverBlockBytes);
            struct FunctionBinding {
                std::uint64_t code;
                std::uint32_t userData[2];
            };
            static_assert(sizeof(FunctionBinding) == 16);
            static constexpr std::array<std::uint32_t, 16> validator{
                0xbeeb03ffu, 0x00000021u, 0xb96a1818u, 0xbf06106au,
                0xbf800000u, 0x8590807eu, 0xb96a0a18u, 0xbf066a80u,
                0xbf800000u, 0x85ea807eu, 0x88ea106au, 0xbf870003u,
                0xbefc03ffu, 0x8a6ca000u, 0xbf920009u, 0xbefd210eu,
            };
            const auto base = DriverBlockAddress + DriverTableOffset;
            const auto functionTable = base + 0x60u;
            const auto code = base + 0x100u;
            std::array<FunctionBinding, 8> functions{};
            functions[2].code = code;
            const std::array<std::uint32_t, 4> descriptor{static_cast<std::uint32_t>(functionTable), static_cast<std::uint32_t>(functionTable >> 32u) | 0x00100000u, static_cast<std::uint32_t>(functions.size()), 0x5204u};
            std::memcpy(reinterpret_cast<void*>(functionTable), functions.data(), sizeof(functions));
            std::memcpy(reinterpret_cast<void*>(code), validator.data(), sizeof(validator));
            std::memcpy(reinterpret_cast<void*>(base), descriptor.data(), sizeof(descriptor));
            std::uint8_t* backing = nullptr;
            std::uint64_t backingBytes = 0;
            if (!DirectMemoryBackingView_nid_no_patch(&backing, &backingBytes) || static_cast<std::uint64_t>(physical) + DriverBlockBytes > backingBytes) throw std::runtime_error("the AGC driver block is not in the direct memory backing view");
            const bool same = std::memcmp(backing + physical + DriverTableOffset, reinterpret_cast<const void*>(base), 0x100u + sizeof(validator)) == 0;
            Print("[kyty] AGC driver block: 2 MiB direct memory at 0x%llx (physical 0x%llx), Kyty's static table layout at +0x40000; backing view %s the guest view's bytes\n", static_cast<unsigned long long>(DriverBlockAddress), static_cast<unsigned long long>(physical), same ? "holds" : "DOES NOT hold");
            if (!same) throw std::runtime_error("the direct memory backing view does not see the guest's writes");
        });
        return reinterpret_cast<void*>(DriverBlockAddress + DriverTableOffset);
    }

    void reportCounts(const char* why) {
        auto load = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
        Print("[kyty] adapter (%s): submits %llu (%llu dwords, queues seen 0x%llx); flips cpu %llu, reserved %llu (queue full %llu), ready %llu, cancelled %llu; rendering waits %llu, buffer queries %llu; interrupts %llu raised, %llu delivered; eq add %llu delete %llu; shaders %llu; memory map %llu unmap %llu protect %llu (backend protects %llu), host writes %llu reads %llu; faults %llu (%llu served); presents %llu clears %llu dumps %llu; suspend %llu idle %llu; labels %llu, range checks %llu\n", why,
            load(counts.submits), load(counts.submitDwords), load(counts.queues), load(counts.cpuFlips), load(counts.flipReservations), load(counts.flipsFull), load(counts.flipsReady), load(counts.flipsCancelled), load(counts.renderingWaits), load(counts.displayBuffers),
            load(counts.interrupts), load(counts.interruptsDelivered), load(counts.eqAdds), load(counts.eqDeletes), load(counts.shaders), load(counts.maps), load(counts.unmaps), load(counts.protects), load(counts.protectGuest), load(counts.hostWrites), load(counts.hostReads),
            load(counts.faults), load(counts.faultsServed), load(counts.presents), load(counts.clears), load(counts.dumps), load(counts.suspends), load(counts.idles), load(counts.labelStores), load(counts.checks));
    }

private:
    Adapter();

    void reportLoop() {
        std::uint64_t previous = 0;
        while (!stopped.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            const auto activity = counts.submits.load() + counts.maps.load() + counts.unmaps.load() + counts.presents.load() + counts.flipsReady.load() + counts.hostWrites.load() + counts.hostReads.load() + counts.interrupts.load();
            if (activity == previous) continue;
            previous = activity;
            reportCounts("10 s");
        }
    }

    std::once_flag started;
    std::once_flag driverBlock;
    std::mutex pendingMutex;
    std::vector<PendingChange> pending;
    std::atomic<bool> stopped{false};
    std::thread reporter;
    std::mutex outputsMutex;
    std::map<std::uint32_t, std::shared_ptr<IVideoOutput>> outputs;
    std::mutex flipsMutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<IFlipRequest>> flips;
    std::uint64_t nextFlipToken = 0;
    std::atomic<std::uint64_t> frameSerial{0};
    std::mutex eqMutex;
    std::vector<EqRegistration> registrations;
    std::mutex windowsMutex;
    std::map<void*, std::unique_ptr<WindowCopy>> windows;
    std::mutex dumpMutex;
    int dumped = 0;
    std::uint64_t dumpPresents = 0;
    std::mutex presentMutex;
    std::chrono::steady_clock::time_point lastPresentReport = std::chrono::steady_clock::now();
    std::uint64_t lastPresentTotal = 0;
};

Adapter& A() {
    return Adapter::Get();
}

[[noreturn]] void Fatal(const char* what) {
    Print("[kyty] fatal: %s\n", what);
    FlushAsyncLog_nid_no_patch();
    std::abort();
}

template <typename TWork>
auto Guarded(const char* where, TWork&& work) noexcept -> decltype(work()) {
    try {
        return work();
    } catch (const ProcessShutdown&) {
        LibcAwaitExit_nid_postfix();
    } catch (const std::exception& error) {
        char text[512];
        std::snprintf(text, sizeof(text), "%s: %s", where, error.what());
        Fatal(text);
    } catch (...) {
        Fatal(where);
    }
}

void HostLog(void*, const char* line) {
    Print("[kyty] %s\n", line);
}

void HostFatal(void*, const char* message) {
    Fatal(message);
}

void HostRaiseInterrupt(void*, std::uint32_t eventId, std::uint32_t contextId) {
    Guarded("raise_interrupt", [&] { A().RaiseInterrupt(eventId, contextId); });
}

int HostFlipReserve(void*, const Aps5GpuFlip* flip, std::uint64_t* token) {
    return Guarded("flip_reserve", [&] {
        auto& adapter = A();
        const auto output = adapter.Output(flip->handle);
        if (!output) throw std::runtime_error("GPU flip on a VideoOut handle that is not open");
        auto request = output->Reserve(FlipInfo{flip->handle, flip->index, flip->mode, flip->argument});
        if (!request) {
            adapter.counts.flipsFull.fetch_add(1, std::memory_order_relaxed);
            return static_cast<int>(APS5_GPU_FLIP_QUEUE_FULL);
        }
        adapter.counts.flipReservations.fetch_add(1, std::memory_order_relaxed);
        *token = adapter.KeepFlip(std::move(request));
        return static_cast<int>(APS5_GPU_OK);
    });
}

void HostFlipWaitRoom(void*, std::uint32_t handle) {
    Guarded("flip_wait_room", [&] {
        const auto output = A().Output(handle);
        if (!output) throw std::runtime_error("flip room wait on a VideoOut handle that is not open");
        output->WaitForFlipRoom();
    });
}

void HostFlipReady(void*, std::uint64_t token) {
    Guarded("flip_ready", [&] {
        auto& adapter = A();
        const auto request = adapter.TakeFlip(token);
        if (!request) throw std::runtime_error("flip_ready for an unknown flip token");
        request->GpuReady(std::make_shared<FrameTiming>(adapter.NextFrameSerial()));
        adapter.counts.flipsReady.fetch_add(1, std::memory_order_relaxed);
    });
}

void HostFlipCancel(void*, std::uint64_t token) {
    Guarded("flip_cancel", [&] {
        auto& adapter = A();
        if (!adapter.TakeFlip(token)) throw std::runtime_error("flip_cancel for an unknown flip token");
        adapter.counts.flipsCancelled.fetch_add(1, std::memory_order_relaxed);
    });
}

int HostDisplayBuffer(void*, std::uint32_t handle, std::int32_t index, Aps5GpuDisplayBuffer* buffer) {
    return Guarded("display_buffer", [&] {
        auto& adapter = A();
        adapter.counts.displayBuffers.fetch_add(1, std::memory_order_relaxed);
        const auto output = adapter.Output(handle);
        if (!output) return static_cast<int>(APS5_GPU_ERROR);
        const auto display = output->Buffer(index);
        if (!display) return static_cast<int>(APS5_GPU_ERROR);
        *buffer = Aps5GpuDisplayBuffer{display->address, display->pixelFormat, display->dccAddress, display->dccClearColor, display->width, display->height, display->tilingMode, display->pitchInPixel, display->alignment, display->dccControl};
        return static_cast<int>(APS5_GPU_OK);
    });
}

void HostRenderingWait(void*, std::uint32_t handle, std::int32_t index) {
    Guarded("rendering_wait", [&] {
        auto& adapter = A();
        adapter.counts.renderingWaits.fetch_add(1, std::memory_order_relaxed);
        const auto output = adapter.Output(handle);
        if (!output) throw std::runtime_error("rendering wait on a VideoOut handle that is not open");
        if (index < 0) throw std::runtime_error("rendering wait for a negative buffer index");
        output->CaptureRenderingWait(static_cast<std::uint32_t>(index))->Wait();
    });
}

void HostFrameReady(void*, std::int32_t dumpIndex, const void* bgra, std::uint32_t fullWidth, std::uint32_t fullHeight, std::uint32_t rowBytes) {
    Guarded("frame_ready", [&] {
        auto& adapter = A();
        if (dumpIndex < 0 || bgra == nullptr || fullWidth == 0 || fullHeight == 0 || rowBytes < fullWidth * 4u) throw std::runtime_error("invalid frame dump");
        const auto scale = DumpScale();
        const auto width = (fullWidth + scale - 1) / scale;
        const auto height = (fullHeight + scale - 1) / scale;
        std::vector<std::byte> pixels(static_cast<std::size_t>(width) * height * 4u);
        const auto* source = static_cast<const std::byte*>(bgra);
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) std::memcpy(pixels.data() + (static_cast<std::size_t>(y) * width + x) * 4u, source + static_cast<std::size_t>(y) * scale * rowBytes + static_cast<std::size_t>(x) * scale * 4u, 4);
        }
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%03d.bmp", dumpIndex);
        adapter.frameDumps.Write_nid_no_patch(name, width, height, pixels);
        adapter.counts.dumps.fetch_add(1, std::memory_order_relaxed);
        static const auto start = std::chrono::steady_clock::now();
        Print("[gpu] frame_%03d.bmp: %ux%u display read back on the GPU at %.1f s\n", dumpIndex, fullWidth, fullHeight, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    });
}

int HostBackingView(void*, std::uint8_t** base, std::uint64_t* bytes) {
    return DirectMemoryBackingView_nid_no_patch(base, bytes) ? APS5_GPU_OK : APS5_GPU_UNSUPPORTED;
}

int HostDirectMapping(void*, std::uint64_t address, std::uint64_t* start, std::uint64_t* end, std::uint64_t* offset, std::int32_t* memoryType) {
    std::uintptr_t first = 0;
    std::uintptr_t last = 0;
    std::uint64_t physical = 0;
    int type = 0;
    if (!DirectMemoryQueryMapping_nid_no_patch(static_cast<std::uintptr_t>(address), &first, &last, &physical, &type)) return APS5_GPU_ERROR;
    *start = first;
    *end = last;
    *offset = physical;
    *memoryType = type;
    return APS5_GPU_OK;
}

std::uint64_t HostMappingGeneration(void*) {
    return A().generation.load(std::memory_order_acquire);
}

int HostProtectGuest(void*, std::uint64_t address, std::uint64_t bytes, std::uint32_t protection) {
    auto& adapter = A();
    if (!AbOn(AbSwitch::LeanCounters)) adapter.counts.protectGuest.fetch_add(1, std::memory_order_relaxed);
    DWORD previous = 0;
    const auto protect = [&] { return VirtualProtect(reinterpret_cast<void*>(address), static_cast<SIZE_T>(bytes), protection, &previous) ? APS5_GPU_OK : APS5_GPU_ERROR; };
    if (g_changeDepth != 0) return protect();
    if (AbOn(AbSwitch::ProtectSrw)) {
        AcquireSRWLockShared(&adapter.protectSrw);
        const int result = protect();
        ReleaseSRWLockShared(&adapter.protectSrw);
        return result;
    }
    std::shared_lock lock(adapter.protectLock);
    return protect();
}

std::uint64_t HostReadTsc(void*) {
    return sceKernelReadTsc();
}

std::uint64_t HostTscFrequency(void*) {
    return sceKernelGetTscFrequency();
}

std::uint64_t HostProcessTimeUs(void*) {
    return sceKernelGetProcessTime();
}

std::uint64_t HostProcessTimeCounter(void*) {
    return sceKernelGetProcessTimeCounter();
}

int HostIsGuestThread(void*) {
    return IsGuestPthread_nid_no_patch() ? 1 : 0;
}

const char* HostAppTitleId(void*) {
    static const AppTitleId id = GetAppTitleId_nid_postfix();
    return id.value;
}

const char* HostAppContentVersion(void*) {
    static const AppContentVersion version = GetAppContentVersion_nid_no_patch();
    return version.value;
}

void ChangeBegin(std::uintptr_t address, std::size_t bytes, std::uint32_t kind, bool hadPages) {
    auto& adapter = A();
    if (kind != GuestArena::GuestArenaChangeProtect && hadPages) {
        adapter.Deliver(PendingChange{PendingKind::Unmap, address, bytes, 0, 0, {}});
        adapter.counts.unmaps.fetch_add(1, std::memory_order_relaxed);
        adapter.generation.fetch_add(1, std::memory_order_acq_rel);
    }
    if (g_changeDepth++ == 0) {
        adapter.protectLock.lock();
        AcquireSRWLockExclusive(&adapter.protectSrw);
    }
}

void ChangeEnd(std::uintptr_t address, std::size_t bytes, std::uint32_t kind, std::uint32_t protection, std::uint64_t directOffset, bool succeeded) {
    auto& adapter = A();
    if (--g_changeDepth == 0) {
        ReleaseSRWLockExclusive(&adapter.protectSrw);
        adapter.protectLock.unlock();
    }
    if (!succeeded) return;
    if (kind == GuestArena::GuestArenaChangeMap) {
        adapter.Deliver(PendingChange{PendingKind::Map, address, bytes, protection, directOffset == GuestArena::GuestArenaNoSectionOffset ? APS5_GPU_NOT_DIRECT : directOffset, {}});
        adapter.counts.maps.fetch_add(1, std::memory_order_relaxed);
        adapter.generation.fetch_add(1, std::memory_order_acq_rel);
    } else if (kind == GuestArena::GuestArenaChangeProtect) {
        if (adapter.Created()) adapter.api->memory_protect(adapter.backend, address, bytes, protection);
        adapter.counts.protects.fetch_add(1, std::memory_order_relaxed);
    }
}

void HostWriteBegins(std::uintptr_t address, std::size_t bytes) {
    auto& adapter = A();
    adapter.counts.hostWrites.fetch_add(1, std::memory_order_relaxed);
    if (adapter.Created()) adapter.api->host_write_begin(adapter.backend, address, bytes);
}

void HostWriteEnds(std::uintptr_t address, std::size_t bytes) {
    auto& adapter = A();
    if (adapter.Created()) adapter.api->host_write_end(adapter.backend, address, bytes);
}

void HostReads(std::uintptr_t address, std::size_t bytes) {
    auto& adapter = A();
    adapter.counts.hostReads.fetch_add(1, std::memory_order_relaxed);
    if (adapter.Created()) adapter.api->host_read(adapter.backend, address, bytes);
}

LONG WINAPI FaultHook(EXCEPTION_POINTERS* info) {
    const auto* record = info->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    auto* adapter = g_adapter.load(std::memory_order_acquire);
    if (adapter == nullptr) return EXCEPTION_CONTINUE_SEARCH;
    const auto address = static_cast<std::uint64_t>(record->ExceptionInformation[1]);
    const auto kind = record->ExceptionInformation[0];
    const bool counted = !AbOn(AbSwitch::LeanCounters);
    if (counted) adapter->counts.faults.fetch_add(1, std::memory_order_relaxed);
    if (auto* page = g_selfCheckPage.load(std::memory_order_acquire); page != nullptr && address - reinterpret_cast<std::uint64_t>(page) < 4096u) {
        DWORD previous = 0;
        if (!VirtualProtect(page, 4096, PAGE_READWRITE, &previous)) return EXCEPTION_CONTINUE_SEARCH;
        g_selfCheckServed.store(true, std::memory_order_release);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (!adapter->Created()) return EXCEPTION_CONTINUE_SEARCH;
    const std::uint32_t access = kind == 1 ? APS5_GPU_FAULT_WRITE : kind == 8 ? APS5_GPU_FAULT_EXECUTE : APS5_GPU_FAULT_READ;
    if (adapter->api->handle_fault(adapter->backend, address, access) != APS5_GPU_FAULT_HANDLED) return EXCEPTION_CONTINUE_SEARCH;
    if (counted) adapter->counts.faultsServed.fetch_add(1, std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_EXECUTION;
}

bool FaultHookFirst() {
    auto* page = VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READONLY);
    if (page == nullptr) return false;
    g_selfCheckPage.store(page, std::memory_order_release);
    *static_cast<volatile std::uint32_t*>(page) = 1u;
    g_selfCheckPage.store(nullptr, std::memory_order_release);
    VirtualFree(page, 0, MEM_RELEASE);
    return g_selfCheckServed.load(std::memory_order_acquire);
}

int CreateBackend(const Aps5GpuHost& host, const Aps5GpuBackendApi** api, Aps5GpuBackend** backend, std::string& source) {
    const char* name = std::getenv("APS5_GPU_BACKEND");
    if (name != nullptr && std::strcmp(name, "null") == 0) {
        source = "in-tree null backend";
        return Aps5NullGpuBackendCreate(&host, api, backend);
    }
#ifdef APS5_KYTY_GPU_BACKEND
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, reinterpret_cast<LPCWSTR>(&CreateBackend), &self)) throw std::runtime_error("APS5_GPU_BACKEND=kyty: the AGC driver's own module is not known");
    std::wstring path(32768, L'\0');
    const auto length = GetModuleFileNameW(self, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 || length >= path.size()) throw std::runtime_error("APS5_GPU_BACKEND=kyty: the AGC driver's path is not known");
    path.resize(length);
    path.resize(path.find_last_of(L"\\/") + 1);
    path += L"kyty_gpu.prx";
    const auto module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (module == nullptr) throw std::runtime_error("APS5_GPU_BACKEND=kyty: kyty_gpu.prx beside libSceAgcDriver.prx could not be loaded (error " + std::to_string(GetLastError()) + ")");
    const auto create = reinterpret_cast<Aps5GpuBackendCreateFunction>(reinterpret_cast<void*>(GetProcAddress(module, APS5_GPU_BACKEND_CREATE_SYMBOL)));
    if (create == nullptr) throw std::runtime_error("APS5_GPU_BACKEND=kyty: kyty_gpu.prx does not export " APS5_GPU_BACKEND_CREATE_SYMBOL);
    source = "kyty_gpu.prx loaded beside libSceAgcDriver.prx";
    return create(&host, api, backend);
#else
    throw std::runtime_error("APS5_GPU_BACKEND=kyty: this build has no Kyty backend (configure with APS5_KYTY_GPU_DLL)");
#endif
}

Adapter::Adapter() {
    host.abi_version = APS5_GPU_BACKEND_ABI_VERSION;
    host.size = sizeof(Aps5GpuHost);
    host.context = this;
    host.log = &HostLog;
    host.fatal = &HostFatal;
    host.raise_interrupt = &HostRaiseInterrupt;
    host.flip_reserve = &HostFlipReserve;
    host.flip_wait_room = &HostFlipWaitRoom;
    host.flip_ready = &HostFlipReady;
    host.flip_cancel = &HostFlipCancel;
    host.display_buffer = &HostDisplayBuffer;
    host.rendering_wait = &HostRenderingWait;
    host.frame_ready = &HostFrameReady;
    host.backing_view = &HostBackingView;
    host.direct_mapping = &HostDirectMapping;
    host.mapping_generation = &HostMappingGeneration;
    host.protect_guest = &HostProtectGuest;
    host.read_tsc = &HostReadTsc;
    host.tsc_frequency = &HostTscFrequency;
    host.process_time_us = &HostProcessTimeUs;
    host.process_time_counter = &HostProcessTimeCounter;
    host.is_guest_thread = &HostIsGuestThread;
    host.app_title_id = &HostAppTitleId;
    host.app_content_version = &HostAppContentVersion;
}

bool Install() {
    auto& adapter = Adapter::Get();
    g_adapter.store(&adapter, std::memory_order_release);
    GuestArena::GuestArenaSetBackendObserver_nid_no_patch(&ChangeBegin, &ChangeEnd);
    GuestArena::GuestArenaSetHostWriteHook_nid_no_patch(&HostWriteBegins);
    GuestArena::GuestArenaSetHostWriteEndHook_nid_no_patch(&HostWriteEnds);
    GuestArena::GuestArenaSetHostReadHook_nid_no_patch(&HostReads);
    if (AddVectoredExceptionHandler(1, &FaultHook) == nullptr) throw std::runtime_error("the GPU backend's fault hook could not be installed");
    const bool first = FaultHookFirst();
    std::uint8_t* backingBase = nullptr;
    std::uint64_t backingBytes = 0;
    const bool backing = DirectMemoryBackingView_nid_no_patch(&backingBase, &backingBytes);
    const char* name = std::getenv("APS5_GPU_BACKEND");
    Print("[kyty] adapter installed: backend '%s' (interface v%u; created at its first use, outside the loader lock); guest memory observers on; fault hook %s; direct memory backing view %p+0x%llx%s\n", name != nullptr ? name : "?", APS5_GPU_BACKEND_ABI_VERSION, first ? "first in the exception chain (self-check served)" : "NOT first in the exception chain", static_cast<void*>(backingBase), static_cast<unsigned long long>(backingBytes), backing ? "" : " (not in kyty memory mode)");
    if (!first) throw std::runtime_error("the GPU backend's fault hook is not first in the exception chain");
    Ab().Configure();
    return true;
}

bool InstallOrReport() {
    try {
        return Install();
    } catch (const std::exception& error) {
        Print("[kyty] adapter installation failed: %s\n", error.what());
        throw;
    }
}

const bool g_installed = Active() && InstallOrReport();

}

bool Active() {
    static const bool active = LibcGpuBackendKyty_nid_no_patch();
    return active;
}

void Submit(const Packet* packet, std::uint32_t queue) {
    auto& adapter = A();
    adapter.EnsureStarted();
    if (packet == nullptr || packet->addr == nullptr) throw std::runtime_error("AGC driver: null command buffer");
    if (adapter.counts.submits.fetch_add(1, std::memory_order_relaxed) == 0) Print("[kyty] first guest submission: queue 0x%x, %u dwords, straight to the backend\n", queue, packet->dw_num);
    adapter.counts.submitDwords.fetch_add(packet->dw_num, std::memory_order_relaxed);
    if (queue < 64) adapter.counts.queues.fetch_or(std::uint64_t{1} << queue, std::memory_order_relaxed);
    if (adapter.api->submit(adapter.backend, queue, packet->addr, packet->dw_num) != APS5_GPU_OK) throw std::runtime_error("the GPU backend refused a submission");
}

void SubmitFlip(const FlipInfo& info) {
    auto& adapter = A();
    adapter.EnsureStarted();
    adapter.counts.cpuFlips.fetch_add(1, std::memory_order_relaxed);
    const Aps5GpuFlip flip{info.handle, info.index, info.mode, 0, info.argument};
    if (adapter.api->submit_flip(adapter.backend, &flip) != APS5_GPU_OK) throw std::runtime_error("the GPU backend refused a CPU flip");
}

void SuspendPoint() {
    auto& adapter = A();
    adapter.EnsureStarted();
    adapter.counts.suspends.fetch_add(1, std::memory_order_relaxed);
    adapter.api->suspend_point(adapter.backend);
}

void WaitIdle() {
    auto& adapter = A();
    adapter.counts.idles.fetch_add(1, std::memory_order_relaxed);
    if (adapter.Created()) adapter.api->wait_idle(adapter.backend);
}

void Shutdown() {
    A().Shutdown();
}

void RegisterShader(const Shader* shader) {
    auto& adapter = A();
    if (shader == nullptr) throw std::runtime_error("AGC driver: null shader");
    const Aps5GpuShaderInfo info{reinterpret_cast<std::uint64_t>(shader->code), shader->user_data, shader->input_semantics, shader->num_input_semantics, shader->shader_size, shader->scratch_size_dw_per_thread, shader->type, {}};
    if (adapter.counts.shaders.fetch_add(1, std::memory_order_relaxed) == 0) Print("[kyty] first shader registration: code 0x%llx, %u bytes\n", static_cast<unsigned long long>(info.code), info.code_size_bytes);
    adapter.Deliver(PendingChange{PendingKind::Shader, 0, 0, 0, 0, info});
}

void RegisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    A().AddOutput(handle, output);
    Print("[kyty] VideoOut output %u registered\n", handle);
}

void UnregisterVideoOutput(std::uint32_t handle, const std::shared_ptr<IVideoOutput>& output) {
    A().RemoveOutput(handle, output);
}

void AttachWindow(const PresentationWindow& window) {
    auto& adapter = A();
    adapter.EnsureStarted();
    auto& copy = adapter.Window(window);
    if (adapter.api->attach_window(adapter.backend, &copy.window) != APS5_GPU_OK) throw std::runtime_error("the GPU backend could not attach the window");
}

void Present(const PresentationWindow& window, const DisplayBuffer* buffer, bool opaque, void (*gpuReady)(void*), void* context) {
    auto& adapter = A();
    adapter.EnsureStarted();
    auto& copy = adapter.Window(window);
    Aps5GpuDisplayBuffer display{};
    if (buffer != nullptr) display = Aps5GpuDisplayBuffer{buffer->address, buffer->pixelFormat, buffer->dccAddress, buffer->dccClearColor, buffer->width, buffer->height, buffer->tilingMode, buffer->pitchInPixel, buffer->alignment, buffer->dccControl};
    const int dump = buffer != nullptr ? adapter.NextDumpIndex() : -1;
    const auto presents = (buffer != nullptr ? adapter.counts.presents : adapter.counts.clears).fetch_add(1, std::memory_order_relaxed) + 1;
    adapter.NotePresent();
    if (buffer != nullptr) Ab().AtPresent(presents);
    static const std::uint64_t exitAt = EnvNumber("APS5_ADAPTER_TEST_EXIT_AT_PRESENT", 0);
    if (buffer != nullptr && exitAt != 0 && presents == exitAt) {
        Print("[kyty] APS5_ADAPTER_TEST_EXIT_AT_PRESENT: present %llu; another thread now ends the process with ExitProcess(0), every library's detach and static destructors included\n", static_cast<unsigned long long>(presents));
        std::thread([] {
            std::fflush(nullptr);
            ExitProcess(0);
        }).detach();
    }
    if (adapter.api->present(adapter.backend, &copy.window, buffer != nullptr ? &display : nullptr, opaque ? 1 : 0, dump, gpuReady, context) != APS5_GPU_OK) throw std::runtime_error("the GPU backend could not present");
}

void ReleaseWindow(void* window) {
    auto& adapter = A();
    if (auto copy = adapter.TakeWindow(window); copy && adapter.Created()) adapter.api->release_window(adapter.backend, copy.get());
}

void ReportFailure(std::exception_ptr error) {
    try {
        std::rethrow_exception(error);
    } catch (const std::exception& cause) {
        Print("[kyty] asynchronous failure: %s\n", cause.what());
    } catch (...) {
        Print("[kyty] asynchronous failure with a non-standard exception\n");
    }
}

void* StaticTable() {
    return A().StaticTable();
}

void CheckGuestMemory(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable) {
    A().counts.checks.fetch_add(1, std::memory_order_relaxed);
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    if (alignment == 0 || address == 0 || address % alignment != 0) throw std::runtime_error("guest memory check: null or misaligned address");
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) throw std::runtime_error("guest memory check: address range overflow");
    for (auto cursor = address; cursor < address + bytes;) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0 || info.State != MEM_COMMIT) throw std::runtime_error("guest memory is not readable");
        const auto protect = info.Protect & 0xffu;
        const bool readable = protect != PAGE_NOACCESS && protect != PAGE_EXECUTE && (info.Protect & PAGE_GUARD) == 0;
        const bool canWrite = protect == PAGE_READWRITE || protect == PAGE_EXECUTE_READWRITE || protect == PAGE_WRITECOPY || protect == PAGE_EXECUTE_WRITECOPY;
        std::uint32_t guest = 0;
        const bool guestWritable = GuestArena::GuestArenaProtection_nid_postfix(cursor, &guest) && (guest == PAGE_READWRITE || guest == PAGE_EXECUTE_READWRITE);
        const bool guestReadable = guestWritable || (guest == PAGE_READONLY || guest == PAGE_EXECUTE_READ);
        if (!readable && !guestReadable) throw std::runtime_error("guest memory is not readable");
        if (writable && !canWrite && !guestWritable) throw std::runtime_error("guest memory has no write permission");
        const auto next = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
        if (next <= cursor) break;
        cursor = next;
    }
}

void StoreGuestLabel(std::uint64_t* pointer, std::uint64_t value) {
    CheckGuestMemory(pointer, sizeof(*pointer), alignof(std::uint64_t), true);
    A().counts.labelStores.fetch_add(1, std::memory_order_relaxed);
    __atomic_store_n(pointer, value, __ATOMIC_RELEASE);
}

int AddEqEvent(KernelEqueue eq, int id, void* udata) {
    if (eq == 0 || !EqueuePin_nid_postfix(eq)) throw std::runtime_error("sceAgcDriverAddEqEvent: invalid event queue");
    KernelEqueueEvent event{};
    event.event.ident = static_cast<std::uintptr_t>(id);
    event.event.filter = EvfiltGraphicsCore;
    event.event.flags = EV_ADD | EV_CLEAR;
    event.event.fflags = 0;
    event.event.data = id;
    event.event.udata = udata;
    event.filter.triggerFunc = [](KernelEqueueEvent* e, void* data) {
        auto triggered = e->event;
        ++triggered.fflags;
        triggered.data = static_cast<std::intptr_t>(reinterpret_cast<std::uintptr_t>(data));
        if (e->triggered) {
            e->pendingEvents.push_back(triggered);
        } else {
            e->event = triggered;
            e->triggered = true;
        }
    };
    event.filter.resetFunc = [](KernelEqueueEvent* e) {
        e->triggered = false;
        e->event.fflags = 0;
        e->event.data = 0;
    };
    if (EqueueAddEvent_nid_postfix(eq, event) != EQUEUE_OK) throw std::runtime_error("sceAgcDriverAddEqEvent: event queue rejected the event");
    auto& adapter = A();
    adapter.AddRegistration(eq, id);
    if (adapter.counts.eqAdds.fetch_add(1, std::memory_order_relaxed) < 8) Print("[kyty] guest event queue 0x%llx registered for GPU interrupt id 0x%x\n", static_cast<unsigned long long>(eq), static_cast<unsigned>(id));
    return 0;
}

int DeleteEqEvent(KernelEqueue eq, int id) {
    auto& adapter = A();
    adapter.RemoveRegistration(eq, id);
    adapter.counts.eqDeletes.fetch_add(1, std::memory_order_relaxed);
    const auto result = EqueueDeleteEvent_nid_postfix(eq, static_cast<std::uintptr_t>(id), EvfiltGraphicsCore);
    if (result != EQUEUE_OK && result != SCE_KERNEL_ERROR_ENOENT) throw std::runtime_error("sceAgcDriverDeleteEqEvent: event is not registered");
    return 0;
}

std::uint32_t EqContextId(const KernelEvent* event) {
    if (event->data < 0 || static_cast<std::uint64_t>(event->data) > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("sceAgcDriverGetEqContextId: context id out of range");
    return static_cast<std::uint32_t>(event->data);
}

void Unsupported(const char* function) {
    throw std::runtime_error(std::string(function) + " is not supported with APS5_GPU_BACKEND=kyty");
}

}
