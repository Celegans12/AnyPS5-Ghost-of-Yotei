#include "prx/libSceAgcDriver/Backend/include/GpuBackend.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <vector>

struct Aps5GpuBackend {
    const Aps5GpuHost* host = nullptr;
    std::atomic<bool> started{false};
    std::atomic<std::uint64_t> graphicsSubmits{0};
    std::atomic<std::uint64_t> computeSubmits{0};
    std::atomic<std::uint64_t> dwords{0};
    std::atomic<std::uint64_t> packets{0};
    std::atomic<std::uint64_t> chains{0};
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> malformed{0};
    std::atomic<std::uint64_t> gpuFlips{0};
    std::atomic<std::uint64_t> cpuFlips{0};
    std::atomic<std::uint64_t> flipWaits{0};
    std::atomic<std::uint64_t> releases{0};
    std::atomic<std::uint64_t> labelWrites{0};
    std::atomic<std::uint64_t> interrupts{0};
    std::atomic<std::uint64_t> writeData{0};
    std::atomic<std::uint64_t> renderingWaits{0};
    std::atomic<std::uint64_t> unsupportedData{0};
    std::atomic<std::uint64_t> shaders{0};
    std::atomic<std::uint64_t> maps{0};
    std::atomic<std::uint64_t> unmaps{0};
    std::atomic<std::uint64_t> protects{0};
    std::atomic<std::uint64_t> hostWrites{0};
    std::atomic<std::uint64_t> hostReads{0};
    std::atomic<std::uint64_t> faults{0};
    std::atomic<std::uint64_t> presents{0};
    std::atomic<std::uint64_t> clears{0};
    std::atomic<std::uint64_t> dumps{0};
    std::atomic<std::uint64_t> windows{0};
    std::atomic<std::uint64_t> suspends{0};
    std::atomic<std::uint64_t> idles{0};
    std::array<std::atomic<std::uint64_t>, 256> opcodes{};
    std::array<std::atomic<std::uint64_t>, 64> customs{};
    std::mutex reportMutex;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

namespace {

void Log(Aps5GpuBackend* backend, const char* format, auto... args) {
    char line[1024];
    std::snprintf(line, sizeof(line), format, args...);
    backend->host->log(backend->host->context, line);
}

void Report(Aps5GpuBackend* backend, const char* why) {
    char opcodes[512] = "";
    std::size_t used = 0;
    for (std::size_t op = 0; op < backend->opcodes.size() && used + 24 < sizeof(opcodes); ++op) {
        const auto count = backend->opcodes[op].load(std::memory_order_relaxed);
        if (count == 0) continue;
        used += static_cast<std::size_t>(std::snprintf(opcodes + used, sizeof(opcodes) - used, " %02zx:%llu", op, static_cast<unsigned long long>(count)));
    }
    Log(backend, "null backend (%s): submits dcb %llu acb %llu, %llu dwords, %llu packets (%llu chained and %llu called buffers, %llu malformed); flips gpu %llu cpu %llu (room waits %llu); release_mem %llu, label writes %llu, interrupts %llu, write_data %llu, rendering waits %llu, unsupported data %llu; shaders %llu; memory map %llu unmap %llu protect %llu, host writes %llu reads %llu, faults %llu; presents %llu clears %llu dumps %llu; windows %llu; suspends %llu idles %llu; opcodes%s", why,
        static_cast<unsigned long long>(backend->graphicsSubmits.load()), static_cast<unsigned long long>(backend->computeSubmits.load()), static_cast<unsigned long long>(backend->dwords.load()), static_cast<unsigned long long>(backend->packets.load()), static_cast<unsigned long long>(backend->chains.load()), static_cast<unsigned long long>(backend->calls.load()), static_cast<unsigned long long>(backend->malformed.load()),
        static_cast<unsigned long long>(backend->gpuFlips.load()), static_cast<unsigned long long>(backend->cpuFlips.load()), static_cast<unsigned long long>(backend->flipWaits.load()),
        static_cast<unsigned long long>(backend->releases.load()), static_cast<unsigned long long>(backend->labelWrites.load()), static_cast<unsigned long long>(backend->interrupts.load()), static_cast<unsigned long long>(backend->writeData.load()), static_cast<unsigned long long>(backend->renderingWaits.load()), static_cast<unsigned long long>(backend->unsupportedData.load()),
        static_cast<unsigned long long>(backend->shaders.load()),
        static_cast<unsigned long long>(backend->maps.load()), static_cast<unsigned long long>(backend->unmaps.load()), static_cast<unsigned long long>(backend->protects.load()), static_cast<unsigned long long>(backend->hostWrites.load()), static_cast<unsigned long long>(backend->hostReads.load()), static_cast<unsigned long long>(backend->faults.load()),
        static_cast<unsigned long long>(backend->presents.load()), static_cast<unsigned long long>(backend->clears.load()), static_cast<unsigned long long>(backend->dumps.load()),
        static_cast<unsigned long long>(backend->windows.load()), static_cast<unsigned long long>(backend->suspends.load()), static_cast<unsigned long long>(backend->idles.load()), opcodes);
}

void MaybeReport(Aps5GpuBackend* backend) {
    const auto now = std::chrono::steady_clock::now();
    std::unique_lock lock(backend->reportMutex, std::try_to_lock);
    if (!lock.owns_lock() || now - backend->lastReport < std::chrono::seconds(10)) return;
    backend->lastReport = now;
    lock.unlock();
    Report(backend, "10 s");
}

bool First(std::atomic<std::uint64_t>& counter) {
    return counter.fetch_add(1, std::memory_order_relaxed) == 0;
}

std::uint64_t ReferenceClock(Aps5GpuBackend* backend) {
    const auto frequency = backend->host->tsc_frequency(backend->host->context);
    const auto ticks = backend->host->read_tsc(backend->host->context);
    if (frequency == 0) return 0;
    constexpr std::uint64_t ReferenceFrequency = 100000000;
    return ticks / frequency * ReferenceFrequency + ticks % frequency * ReferenceFrequency / frequency;
}

void Store32(std::uint64_t address, std::uint32_t value) {
    __atomic_store_n(reinterpret_cast<std::uint32_t*>(address), value, __ATOMIC_RELEASE);
}

void Store64(std::uint64_t address, std::uint64_t value) {
    if ((address & 7u) == 0) {
        __atomic_store_n(reinterpret_cast<std::uint64_t*>(address), value, __ATOMIC_RELEASE);
        return;
    }
    std::memcpy(reinterpret_cast<void*>(address), &value, sizeof(value));
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

void Flip(Aps5GpuBackend* backend, const Aps5GpuFlip& flip, std::uint32_t queue, bool gpu) {
    const auto* host = backend->host;
    std::uint64_t token = 0;
    for (;;) {
        const int result = host->flip_reserve(host->context, &flip, &token);
        if (result == APS5_GPU_OK) break;
        if (result != APS5_GPU_FLIP_QUEUE_FULL) {
            host->fatal(host->context, "null backend: flip reservation failed");
            return;
        }
        backend->flipWaits.fetch_add(1, std::memory_order_relaxed);
        host->flip_wait_room(host->context, flip.handle);
    }
    host->flip_ready(host->context, token);
    if (gpu) {
        host->raise_interrupt(host->context, queue, 0);
        backend->interrupts.fetch_add(1, std::memory_order_relaxed);
    }
}

void ReleaseMem(Aps5GpuBackend* backend, std::uint32_t queue, const std::uint32_t* packet) {
    backend->releases.fetch_add(1, std::memory_order_relaxed);
    const auto dataSelect = packet[2] >> 29u;
    const auto interruptSelect = (packet[2] >> 24u) & 7u;
    const auto address = static_cast<std::uint64_t>(packet[3]) | static_cast<std::uint64_t>(packet[4]) << 32u;
    const auto value = static_cast<std::uint64_t>(packet[5]) | static_cast<std::uint64_t>(packet[6]) << 32u;
    const auto contextId = packet[7] & 0x07ffffffu;
    if (dataSelect != 0 && address != 0) {
        if (dataSelect == 1) Store32(address, static_cast<std::uint32_t>(value));
        else if (dataSelect == 2) Store64(address, value);
        else if (dataSelect == 3) Store64(address, ReferenceClock(backend));
        else backend->unsupportedData.fetch_add(1, std::memory_order_relaxed);
        if (dataSelect <= 3) backend->labelWrites.fetch_add(1, std::memory_order_relaxed);
    }
    if (interruptSelect == 1 || interruptSelect == 2 || interruptSelect == 4) {
        backend->host->raise_interrupt(backend->host->context, queue, contextId);
        backend->interrupts.fetch_add(1, std::memory_order_relaxed);
    }
}

void WriteData(Aps5GpuBackend* backend, const std::uint32_t* packet, std::uint32_t words) {
    backend->writeData.fetch_add(1, std::memory_order_relaxed);
    if (words < 5) return;
    const auto control = packet[1];
    if (((control >> 8u) & 0xfu) == 0) return;
    const auto address = static_cast<std::uint64_t>(packet[2]) | static_cast<std::uint64_t>(packet[3]) << 32u;
    if (address == 0) return;
    const bool oneAddress = (control & 0x10000u) != 0;
    for (std::uint32_t i = 4; i < words; ++i) Store32(oneAddress ? address : address + 4ull * (i - 4), packet[i]);
    backend->labelWrites.fetch_add(1, std::memory_order_relaxed);
}

void Walk(Aps5GpuBackend* backend, std::uint32_t queue, const std::uint32_t* dwords, std::uint32_t count) {
    struct Segment {
        const std::uint32_t* words;
        std::uint32_t count;
        std::uint32_t at;
    };
    std::array<Segment, 8> stack{};
    std::size_t depth = 0;
    stack[depth++] = {dwords, count, 0};
    std::uint64_t budget = std::uint64_t{1} << 27u;
    while (depth != 0) {
        auto& segment = stack[depth - 1];
        if (segment.at >= segment.count) {
            --depth;
            continue;
        }
        const auto at = segment.at;
        const auto header = segment.words[at];
        const auto type = header >> 30u;
        if (type == 2u) {
            ++segment.at;
            continue;
        }
        const auto words = ((header >> 16u) & 0x3fffu) + 2u;
        if ((type != 3u && type != 0u) || words > segment.count - at || words > budget) {
            if (First(backend->malformed)) Log(backend, "null backend: malformed packet 0x%08x at dword %u of %u on queue 0x%x (depth %zu); rest of the submission skipped", header, at, segment.count, queue, depth);
            return;
        }
        budget -= words;
        segment.at += words;
        backend->packets.fetch_add(1, std::memory_order_relaxed);
        if (type == 3u) {
            const auto* packet = segment.words + at;
            const auto opcode = (header >> 8u) & 0xffu;
            backend->opcodes[opcode].fetch_add(1, std::memory_order_relaxed);
            if (opcode == 0x3fu && words == 4u) {
                const auto* target = reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(packet[1] & ~3u) | static_cast<std::uintptr_t>(packet[2] & 0xffffu) << 32u);
                const auto targetWords = packet[3] & 0xfffffu;
                if (target == nullptr || targetWords == 0) continue;
                if ((packet[3] & (1u << 20u)) != 0) {
                    backend->chains.fetch_add(1, std::memory_order_relaxed);
                    segment = {target, targetWords, 0};
                } else if (depth < stack.size()) {
                    backend->calls.fetch_add(1, std::memory_order_relaxed);
                    stack[depth++] = {target, targetWords, 0};
                } else {
                    if (First(backend->malformed)) Log(backend, "null backend: indirect buffers nested deeper than %zu on queue 0x%x; rest of the submission skipped", stack.size(), queue);
                    return;
                }
                continue;
            }
            if (opcode == 0x10u && (header & 0xfcu) != 0) {
                const auto custom = (header >> 2u) & 0x3fu;
                backend->customs[custom].fetch_add(1, std::memory_order_relaxed);
                if (custom == 0x17u && words == 6u) {
                    const Aps5GpuFlip flip{packet[1], static_cast<std::int32_t>(packet[2]), packet[3], 0, static_cast<std::int64_t>(static_cast<std::uint64_t>(packet[4]) | static_cast<std::uint64_t>(packet[5]) << 32u)};
                    if (First(backend->gpuFlips)) Log(backend, "null backend: first GPU flip on queue 0x%x: handle %u index %d mode %u", queue, flip.handle, flip.index, flip.mode);
                    Flip(backend, flip, queue, true);
                } else if (custom == 0x06u && words >= 4u) {
                    if (First(backend->renderingWaits)) Log(backend, "null backend: first rendering wait: handle %u index %d", packet[1], static_cast<std::int32_t>(packet[2]));
                    backend->host->rendering_wait(backend->host->context, packet[1], static_cast<std::int32_t>(packet[2]));
                }
            } else if (opcode == 0x49u && words == 8u) {
                ReleaseMem(backend, queue, packet);
            } else if (opcode == 0x37u) {
                WriteData(backend, packet, words);
            }
        }
    }
}

int Start(Aps5GpuBackend* backend) {
    if (backend->started.exchange(true)) return APS5_GPU_OK;
    Log(backend, "null backend started: submissions are walked for flips, RELEASE_MEM and WRITE_DATA on the submitting thread; nothing is drawn");
    return APS5_GPU_OK;
}

void Shutdown(Aps5GpuBackend* backend) {
    Report(backend, "shutdown");
}

void Destroy(Aps5GpuBackend* backend) {
    delete backend;
}

int Submit(Aps5GpuBackend* backend, std::uint32_t queue, const std::uint32_t* dwords, std::uint32_t count) {
    if (First(queue == 0 ? backend->graphicsSubmits : backend->computeSubmits)) Log(backend, "null backend: first submission on queue 0x%x (%u dwords)", queue, count);
    backend->dwords.fetch_add(count, std::memory_order_relaxed);
    Walk(backend, queue, dwords, count);
    MaybeReport(backend);
    return APS5_GPU_OK;
}

int SubmitFlip(Aps5GpuBackend* backend, const Aps5GpuFlip* flip) {
    if (First(backend->cpuFlips)) Log(backend, "null backend: first CPU flip: handle %u index %d mode %u", flip->handle, flip->index, flip->mode);
    Flip(backend, *flip, 0, false);
    return APS5_GPU_OK;
}

void SuspendPoint(Aps5GpuBackend* backend) {
    backend->suspends.fetch_add(1, std::memory_order_relaxed);
}

void WaitIdle(Aps5GpuBackend* backend) {
    backend->idles.fetch_add(1, std::memory_order_relaxed);
}

int RegisterShader(Aps5GpuBackend* backend, const Aps5GpuShaderInfo* shader) {
    if (First(backend->shaders)) Log(backend, "null backend: first shader: code 0x%llx type %u, %u bytes", static_cast<unsigned long long>(shader->code), static_cast<unsigned>(shader->type), shader->code_size_bytes);
    return APS5_GPU_OK;
}

void MemoryUnmap(Aps5GpuBackend* backend, std::uint64_t, std::uint64_t) {
    backend->unmaps.fetch_add(1, std::memory_order_relaxed);
}

void MemoryMap(Aps5GpuBackend* backend, std::uint64_t, std::uint64_t, std::uint32_t) {
    backend->maps.fetch_add(1, std::memory_order_relaxed);
}

void MemoryProtect(Aps5GpuBackend* backend, std::uint64_t, std::uint64_t, std::uint32_t) {
    backend->protects.fetch_add(1, std::memory_order_relaxed);
}

void HostWriteBegin(Aps5GpuBackend* backend, std::uint64_t, std::uint64_t) {
    backend->hostWrites.fetch_add(1, std::memory_order_relaxed);
}

void HostWriteEnd(Aps5GpuBackend*, std::uint64_t, std::uint64_t) {}

void HostRead(Aps5GpuBackend* backend, std::uint64_t, std::uint64_t) {
    backend->hostReads.fetch_add(1, std::memory_order_relaxed);
}

int HandleFault(Aps5GpuBackend* backend, std::uint64_t, std::uint32_t) {
    backend->faults.fetch_add(1, std::memory_order_relaxed);
    return APS5_GPU_FAULT_NOT_MINE;
}

int AttachWindow(Aps5GpuBackend* backend, const Aps5GpuWindow* window) {
    if (First(backend->windows)) Log(backend, "null backend: window attached, %ux%u, %u instance extensions", window->width, window->height, window->instance_extension_count);
    return APS5_GPU_OK;
}

void ReleaseWindow(Aps5GpuBackend*, void*) {}

int Present(Aps5GpuBackend* backend, const Aps5GpuWindow* window, const Aps5GpuDisplayBuffer* buffer, std::int32_t, std::int32_t dumpIndex, void (*presented)(void*), void* presentedContext) {
    if (buffer != nullptr) backend->presents.fetch_add(1, std::memory_order_relaxed);
    else backend->clears.fetch_add(1, std::memory_order_relaxed);
    if (dumpIndex >= 0) {
        const auto width = buffer != nullptr ? buffer->width : window->width;
        const auto height = buffer != nullptr ? buffer->height : window->height;
        if (width != 0 && height != 0 && width <= 16384 && height <= 16384) {
            std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 4u, 0x20u);
            for (std::size_t i = 3; i < pixels.size(); i += 4) pixels[i] = 0xffu;
            backend->host->frame_ready(backend->host->context, dumpIndex, pixels.data(), width, height, width * 4u);
            backend->dumps.fetch_add(1, std::memory_order_relaxed);
        }
    }
    presented(presentedContext);
    return APS5_GPU_OK;
}

const Aps5GpuBackendApi NullApi{
    APS5_GPU_BACKEND_ABI_VERSION,
    sizeof(Aps5GpuBackendApi),
    "null",
    &Start,
    &Shutdown,
    &Destroy,
    &Submit,
    &SubmitFlip,
    &SuspendPoint,
    &WaitIdle,
    &RegisterShader,
    &MemoryUnmap,
    &MemoryMap,
    &MemoryProtect,
    &HostWriteBegin,
    &HostWriteEnd,
    &HostRead,
    &HandleFault,
    &AttachWindow,
    &ReleaseWindow,
    &Present,
};

}

extern "C" int Aps5NullGpuBackendCreate(const Aps5GpuHost* host, const Aps5GpuBackendApi** api, Aps5GpuBackend** backend) {
    if (host == nullptr || api == nullptr || backend == nullptr) return APS5_GPU_ERROR;
    if (host->abi_version != APS5_GPU_BACKEND_ABI_VERSION || host->size < sizeof(Aps5GpuHost)) return APS5_GPU_ABI_MISMATCH;
    auto* created = new (std::nothrow) Aps5GpuBackend;
    if (created == nullptr) return APS5_GPU_ERROR;
    created->host = host;
    *api = &NullApi;
    *backend = created;
    return APS5_GPU_OK;
}
