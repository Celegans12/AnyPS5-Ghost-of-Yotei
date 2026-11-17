#include "prx/libSceAgcDriver/Graphics/include/OffThreadEncoding.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libc/include/HostThread.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace AgcDriver::Graphics::OffThreadEncoding {

namespace {

constexpr std::size_t RingBytes = std::size_t{16} << 20u;
constexpr std::size_t LargestRecord = std::size_t{1} << 20u;
constexpr auto ReportEvery = std::chrono::seconds(10);

enum Kind : std::uint32_t { Command = 0, Opening = 1, Closing = 2 };

using Replay = void (*)(const std::byte* body, VkCommandBuffer commands);

struct Header {
    Replay replay;
    VkCommandBuffer commands;
    std::uint32_t bytes;
    std::uint32_t kind;
};

constexpr std::size_t Round(std::size_t bytes) {
    return (bytes + 7u) & ~std::size_t{7};
}

template<typename T>
constexpr std::size_t Bytes(std::size_t count) {
    return Round(sizeof(T) * count);
}

struct State {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr base = nullptr;
    PFN_vkCreateCommandPool createCommandPool = nullptr;
    PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer beginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
    std::unordered_map<std::string_view, PFN_vkVoidFunction> wrappers;
    std::byte* ring = nullptr;
    alignas(64) std::atomic<VkCommandBuffer> current{VK_NULL_HANDLE};
    VkCommandBuffer detached = VK_NULL_HANDLE;
    std::atomic<bool> disabled{false};
    alignas(64) std::atomic<std::uint64_t> published{0};
    alignas(64) std::atomic<std::uint64_t> done{0};
    alignas(64) std::atomic<bool> asleep{false};
    std::atomic<std::uint32_t> wakes{0};
    alignas(64) std::uint64_t written = 0;
    Statistics produced;
    Statistics reportedCounts;
    std::chrono::steady_clock::time_point reportedAt{};
    const char* firstDirect = nullptr;
    alignas(64) std::atomic<std::uint64_t> encodeNs{0};
    std::atomic<VkResult> failure{VK_SUCCESS};
    std::atomic<VkResult> ended{VK_SUCCESS};
    std::mutex poolMutex;
    VkCommandPool pool = VK_NULL_HANDLE;
    std::uint32_t poolFamily = 0;
    std::vector<VkCommandBuffer> spare;
};

State* active = nullptr;

bool Deferring(VkCommandBuffer commands) {
    return commands != VK_NULL_HANDLE && commands == active->current.load(std::memory_order_relaxed);
}

std::uint64_t ElapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

void Wake(State& state) {
    state.asleep.store(false, std::memory_order_relaxed);
    state.wakes.fetch_add(1, std::memory_order_release);
    state.wakes.notify_one();
}

void WaitDone(State& state, std::uint64_t target) {
    while (state.done.load(std::memory_order_acquire) < target) {
        if (state.asleep.load(std::memory_order_seq_cst)) Wake(state);
        SpinPause();
    }
}

void Drain(State& state) {
    if (state.done.load(std::memory_order_acquire) == state.written) return;
    const auto start = std::chrono::steady_clock::now();
    WaitDone(state, state.written);
    ++state.produced.drains;
    state.produced.drainNs += ElapsedNs(start);
}

void Direct(const char* name) {
    auto& state = *active;
    Drain(state);
    ++state.produced.directCalls;
    if (state.firstDirect == nullptr) state.firstDirect = name;
}

std::byte* Reserve(State& state, std::size_t bytes) {
    auto offset = static_cast<std::size_t>(state.written % RingBytes);
    const std::size_t skip = RingBytes - offset < bytes ? RingBytes - offset : 0;
    const auto end = state.written + skip + bytes;
    if (end - state.done.load(std::memory_order_acquire) > RingBytes) WaitDone(state, end - RingBytes);
    if (skip != 0) {
        if (skip >= sizeof(Header)) new (state.ring + offset) Header{nullptr, VK_NULL_HANDLE, 0, Command};
        state.written += skip;
        offset = 0;
    }
    return state.ring + offset;
}

void Publish(State& state, std::size_t bytes) {
    state.written += bytes;
    ++state.produced.records;
    state.produced.bytes += bytes;
    state.published.store(state.written, std::memory_order_seq_cst);
    if (state.asleep.load(std::memory_order_seq_cst)) Wake(state);
}

class Tail {
public:
    explicit Tail(std::byte* at) : at(at) {}

    template<typename T>
    const T* Copy(const T* source, std::size_t count) {
        if (source == nullptr || count == 0) return nullptr;
        auto* target = reinterpret_cast<T*>(at);
        std::memcpy(static_cast<void*>(target), source, sizeof(T) * count);
        at += Bytes<T>(count);
        return target;
    }

    const void* CopyBytes(const void* source, std::size_t bytes) {
        if (source == nullptr || bytes == 0) return nullptr;
        std::memcpy(at, source, bytes);
        const void* target = at;
        at += Round(bytes);
        return target;
    }

    template<typename T>
    T* Place(const T& value) {
        auto* target = reinterpret_cast<T*>(at);
        std::memcpy(static_cast<void*>(target), &value, sizeof(T));
        at += Bytes<T>(1);
        return target;
    }

private:
    std::byte* at;
};

template<typename Fn, typename Tuple>
struct Packed {
    Fn real;
    Tuple arguments;

    static void Replay(const std::byte* body, VkCommandBuffer commands) {
        const auto& self = *reinterpret_cast<const Packed*>(body);
        std::apply([&](const auto&... values) { self.real(commands, values...); }, self.arguments);
    }
};

template<typename Fn, typename Fill>
bool Defer(VkCommandBuffer commands, Fn real, std::size_t tail, Fill&& fill) {
    using Tuple = std::invoke_result_t<Fill&, Tail&>;
    using Body = Packed<Fn, Tuple>;
    static_assert(alignof(Body) <= 8);
    static_assert(std::is_trivially_destructible_v<Body>);
    const auto bytes = sizeof(Header) + Round(sizeof(Body)) + tail;
    if (bytes > LargestRecord) return false;
    auto& state = *active;
    auto* at = Reserve(state, bytes);
    Tail copies(at + sizeof(Header) + Round(sizeof(Body)));
    new (at + sizeof(Header)) Body{real, fill(copies)};
    new (at) Header{&Body::Replay, commands, static_cast<std::uint32_t>(bytes), Command};
    Publish(state, bytes);
    return true;
}

template<typename Body>
void Emit(VkCommandBuffer commands, Replay replay, Kind kind, const Body& body) {
    auto& state = *active;
    const auto bytes = sizeof(Header) + Bytes<Body>(1);
    auto* at = Reserve(state, bytes);
    new (at + sizeof(Header)) Body(body);
    new (at) Header{replay, commands, static_cast<std::uint32_t>(bytes), kind};
    Publish(state, bytes);
}

struct OpeningBody {
    VkCommandBufferUsageFlags flags;
};

struct ClosingBody {
    std::uint32_t unused;
};

void ReplayOpening(const std::byte* body, VkCommandBuffer commands) {
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags = reinterpret_cast<const OpeningBody*>(body)->flags;
    const auto result = active->beginCommandBuffer(commands, &info);
    if (result != VK_SUCCESS) {
        auto expected = VK_SUCCESS;
        active->failure.compare_exchange_strong(expected, result, std::memory_order_relaxed);
    }
}

void ReplayClosing(const std::byte*, VkCommandBuffer commands) {
    active->ended.store(active->endCommandBuffer(commands), std::memory_order_relaxed);
}

void Run(State& state) {
    DriverDetail::NameDriverThread("draw-encode");
    DriverDetail::ReserveEncoderCore();
    const auto window = HandoffSpinWindowOr(std::getenv("APS5_ENCODE_SPIN_US"), HandoffSpinWindow());
    std::uint64_t read = 0;
    bool skipping = false;
    for (;;) {
        const auto available = state.published.load(std::memory_order_acquire);
        if (available == read) {
            if (SpinUntil(window, [&] { return state.published.load(std::memory_order_acquire) != read; })) continue;
            const auto ticket = state.wakes.load(std::memory_order_acquire);
            state.asleep.store(true, std::memory_order_seq_cst);
            if (state.published.load(std::memory_order_seq_cst) == read) state.wakes.wait(ticket, std::memory_order_acquire);
            state.asleep.store(false, std::memory_order_relaxed);
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        while (read != available) {
            const auto offset = static_cast<std::size_t>(read % RingBytes);
            const auto* header = reinterpret_cast<const Header*>(state.ring + offset);
            if (RingBytes - offset < sizeof(Header) || header->replay == nullptr) {
                read += RingBytes - offset;
                continue;
            }
            const auto* body = state.ring + offset + sizeof(Header);
            if (header->kind == Opening) {
                header->replay(body, header->commands);
                skipping = state.failure.load(std::memory_order_relaxed) != VK_SUCCESS;
            } else if (header->kind == Closing) {
                if (skipping) state.ended.store(state.failure.load(std::memory_order_relaxed), std::memory_order_relaxed);
                else header->replay(body, header->commands);
                skipping = false;
            } else if (!skipping) {
                header->replay(body, header->commands);
            }
            read += header->bytes;
        }
        state.done.store(read, std::memory_order_release);
        state.encodeNs.fetch_add(ElapsedNs(start), std::memory_order_relaxed);
    }
}

void Report(State& state) {
    const auto now = std::chrono::steady_clock::now();
    if (state.reportedAt.time_since_epoch().count() == 0) {
        state.reportedAt = now;
        state.reportedCounts = Counts();
        return;
    }
    if (now - state.reportedAt < ReportEvery) return;
    const auto counts = Counts();
    const auto& last = state.reportedCounts;
    const auto batches = counts.batches - last.batches;
    const auto records = counts.records - last.records;
    const double seconds = std::chrono::duration<double>(now - state.reportedAt).count();
    std::fprintf(stderr, "[encode] %.1f s: %llu batches, %llu records (%.1f per batch), %.1f MiB listed; draw-encode busy %.1f ms (%.3f us per record); committer waited %llu times, %.2f ms; %llu direct calls%s%s\n", seconds, static_cast<unsigned long long>(batches), static_cast<unsigned long long>(records),
                 batches != 0 ? static_cast<double>(records) / static_cast<double>(batches) : 0.0, static_cast<double>(counts.bytes - last.bytes) / (1024.0 * 1024.0), static_cast<double>(counts.encodeNs - last.encodeNs) / 1e6,
                 records != 0 ? static_cast<double>(counts.encodeNs - last.encodeNs) / 1e3 / static_cast<double>(records) : 0.0, static_cast<unsigned long long>(counts.drains - last.drains), static_cast<double>(counts.drainNs - last.drainNs) / 1e6,
                 static_cast<unsigned long long>(counts.directCalls - last.directCalls), state.firstDirect != nullptr ? ", first " : "", state.firstDirect != nullptr ? state.firstDirect : "");
    state.reportedAt = now;
    state.reportedCounts = counts;
}

void Unknown(State& state, const char* name) {
    if (!state.disabled.exchange(true, std::memory_order_relaxed)) std::fprintf(stderr, "[encode] %s has no off-thread form: encoding stays on the recording thread from the next batch\n", name);
    if (!GuestMemory::GpuMutex().HeldByThisThread()) return;
    const auto commands = state.current.load(std::memory_order_relaxed);
    if (commands == VK_NULL_HANDLE) return;
    Drain(state);
    state.current.store(VK_NULL_HANDLE, std::memory_order_relaxed);
    state.detached = commands;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Resolve(VkDevice device, const char* name) {
    auto& state = *active;
    if (device == state.device && name != nullptr && std::strncmp(name, "vkCmd", 5) == 0) {
        if (const auto found = state.wrappers.find(std::string_view(name)); found != state.wrappers.end()) return found->second;
        const auto real = state.base(device, name);
        if (real != nullptr) Unknown(state, name);
        return real;
    }
    return state.base(device, name);
}

#define APS5_ENCODE_TAG(name)                     \
    struct Tag_##name {                           \
        using Fn = PFN_##name;                    \
        static constexpr const char* text = #name; \
        static inline Fn real = nullptr;          \
    };

#define APS5_ENCODE_PLAIN(X) \
    X(vkCmdBindPipeline) \
    X(vkCmdDraw) \
    X(vkCmdDrawIndexed) \
    X(vkCmdDrawIndirect) \
    X(vkCmdDrawIndexedIndirect) \
    X(vkCmdDrawIndirectCount) \
    X(vkCmdDrawIndirectCountKHR) \
    X(vkCmdDrawIndexedIndirectCount) \
    X(vkCmdDrawIndexedIndirectCountKHR) \
    X(vkCmdDrawMeshTasksEXT) \
    X(vkCmdDrawMeshTasksIndirectEXT) \
    X(vkCmdDrawMeshTasksIndirectCountEXT) \
    X(vkCmdDispatch) \
    X(vkCmdDispatchIndirect) \
    X(vkCmdDispatchBase) \
    X(vkCmdEndRenderPass) \
    X(vkCmdNextSubpass) \
    X(vkCmdEndRendering) \
    X(vkCmdEndRenderingKHR) \
    X(vkCmdFillBuffer) \
    X(vkCmdSetLineWidth) \
    X(vkCmdSetDepthBounds) \
    X(vkCmdSetDepthBias) \
    X(vkCmdSetCullMode) \
    X(vkCmdSetCullModeEXT) \
    X(vkCmdSetFrontFace) \
    X(vkCmdSetFrontFaceEXT) \
    X(vkCmdSetDepthTestEnable) \
    X(vkCmdSetDepthTestEnableEXT) \
    X(vkCmdSetDepthWriteEnable) \
    X(vkCmdSetDepthWriteEnableEXT) \
    X(vkCmdSetDepthCompareOp) \
    X(vkCmdSetDepthCompareOpEXT) \
    X(vkCmdSetDepthBoundsTestEnable) \
    X(vkCmdSetDepthBoundsTestEnableEXT) \
    X(vkCmdSetStencilTestEnable) \
    X(vkCmdSetStencilTestEnableEXT) \
    X(vkCmdSetStencilOp) \
    X(vkCmdSetStencilOpEXT) \
    X(vkCmdSetStencilCompareMask) \
    X(vkCmdSetStencilWriteMask) \
    X(vkCmdSetStencilReference) \
    X(vkCmdSetPrimitiveTopology) \
    X(vkCmdSetPrimitiveTopologyEXT) \
    X(vkCmdSetPrimitiveRestartEnable) \
    X(vkCmdSetPrimitiveRestartEnableEXT) \
    X(vkCmdSetRasterizerDiscardEnable) \
    X(vkCmdSetRasterizerDiscardEnableEXT) \
    X(vkCmdSetDepthBiasEnable) \
    X(vkCmdSetDepthBiasEnableEXT) \
    X(vkCmdBindIndexBuffer) \
    X(vkCmdResetQueryPool) \
    X(vkCmdBeginQuery) \
    X(vkCmdEndQuery) \
    X(vkCmdWriteTimestamp) \
    X(vkCmdCopyQueryPoolResults) \
    X(vkCmdWriteBufferMarkerAMD) \
    X(vkCmdSetCheckpointNV) \
    X(vkCmdSetEvent) \
    X(vkCmdResetEvent)

#define APS5_ENCODE_SPECIAL(X) \
    X(vkCmdSetViewport) \
    X(vkCmdSetScissor) \
    X(vkCmdSetViewportWithCount) \
    X(vkCmdSetViewportWithCountEXT) \
    X(vkCmdSetScissorWithCount) \
    X(vkCmdSetScissorWithCountEXT) \
    X(vkCmdSetColorWriteMaskEXT) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) \
    X(vkCmdBindVertexBuffers) \
    X(vkCmdCopyBuffer) \
    X(vkCmdCopyBufferToImage) \
    X(vkCmdCopyImageToBuffer) \
    X(vkCmdCopyImage) \
    X(vkCmdBlitImage) \
    X(vkCmdResolveImage) \
    X(vkCmdClearColorImage) \
    X(vkCmdClearDepthStencilImage) \
    X(vkCmdClearAttachments) \
    X(vkCmdUpdateBuffer) \
    X(vkCmdPipelineBarrier) \
    X(vkCmdBeginRenderPass) \
    X(vkCmdBeginRendering) \
    X(vkCmdBeginRenderingKHR)

#define APS5_ENCODE_FORWARD(X) \
    X(vkCmdExecuteCommands) \
    X(vkCmdBindShadersEXT) \
    X(vkCmdPushDescriptorSetKHR) \
    X(vkCmdSetDepthBias2EXT) \
    X(vkCmdPipelineBarrier2) \
    X(vkCmdPipelineBarrier2KHR) \
    X(vkCmdExecuteGeneratedCommandsEXT) \
    X(vkCmdPreprocessGeneratedCommandsEXT) \
    X(vkCmdSetSampleLocationsEXT) \
    X(vkCmdSetBlendConstants) \
    X(vkCmdWaitEvents) \
    X(vkCmdBindVertexBuffers2) \
    X(vkCmdBindVertexBuffers2EXT) \
    X(vkCmdSetVertexInputEXT) \
    X(vkCmdBeginDebugUtilsLabelEXT) \
    X(vkCmdEndDebugUtilsLabelEXT) \
    X(vkCmdInsertDebugUtilsLabelEXT) \
    X(vkCmdCopyBuffer2) \
    X(vkCmdCopyImage2) \
    X(vkCmdCopyBufferToImage2) \
    X(vkCmdCopyImageToBuffer2) \
    X(vkCmdBlitImage2) \
    X(vkCmdResolveImage2) \
    X(vkCmdBeginRenderPass2) \
    X(vkCmdEndRenderPass2) \
    X(vkCmdNextSubpass2)

APS5_ENCODE_PLAIN(APS5_ENCODE_TAG)
APS5_ENCODE_SPECIAL(APS5_ENCODE_TAG)
APS5_ENCODE_FORWARD(APS5_ENCODE_TAG)

template<typename Tag, typename Fn = typename Tag::Fn>
struct Plain;

template<typename Tag, typename... A>
struct Plain<Tag, void(VKAPI_PTR*)(VkCommandBuffer, A...)> {
    static VKAPI_ATTR void VKAPI_CALL Call(VkCommandBuffer commands, A... arguments) {
        if (!Deferring(commands)) return Tag::real(commands, arguments...);
        Defer(commands, Tag::real, 0, [&](Tail&) { return std::tuple<A...>{arguments...}; });
    }
};

template<typename Tag, typename Fn = typename Tag::Fn>
struct Forward;

template<typename Tag, typename... A>
struct Forward<Tag, void(VKAPI_PTR*)(VkCommandBuffer, A...)> {
    static VKAPI_ATTR void VKAPI_CALL Call(VkCommandBuffer commands, A... arguments) {
        if (Deferring(commands)) Direct(Tag::text);
        Tag::real(commands, arguments...);
    }
};

template<typename T>
bool Chained(const T* items, std::size_t count) {
    if (count != 0 && items == nullptr) return true;
    for (std::size_t i = 0; i < count; ++i) {
        if (items[i].pNext != nullptr) return true;
    }
    return false;
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL SetViewport(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkViewport* viewports) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkViewport>(count), [&](Tail& tail) { return std::tuple{first, count, tail.Copy(viewports, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, first, count, viewports);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL SetScissor(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkRect2D* scissors) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkRect2D>(count), [&](Tail& tail) { return std::tuple{first, count, tail.Copy(scissors, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, first, count, scissors);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL SetViewportWithCount(VkCommandBuffer commands, std::uint32_t count, const VkViewport* viewports) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkViewport>(count), [&](Tail& tail) { return std::tuple{count, tail.Copy(viewports, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, count, viewports);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL SetScissorWithCount(VkCommandBuffer commands, std::uint32_t count, const VkRect2D* scissors) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkRect2D>(count), [&](Tail& tail) { return std::tuple{count, tail.Copy(scissors, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, count, scissors);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL SetColorWriteMask(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkColorComponentFlags* masks) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkColorComponentFlags>(count), [&](Tail& tail) { return std::tuple{first, count, tail.Copy(masks, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, first, count, masks);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL BindDescriptorSets(VkCommandBuffer commands, VkPipelineBindPoint point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets, std::uint32_t dynamicCount, const std::uint32_t* offsets) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkDescriptorSet>(count) + Bytes<std::uint32_t>(dynamicCount), [&](Tail& tail) { return std::tuple{point, layout, first, count, tail.Copy(sets, count), dynamicCount, tail.Copy(offsets, dynamicCount)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, point, layout, first, count, sets, dynamicCount, offsets);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL PushConstants(VkCommandBuffer commands, VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset, std::uint32_t size, const void* values) {
    if (Deferring(commands) && Defer(commands, Tag::real, Round(size), [&](Tail& tail) { return std::tuple{layout, stages, offset, size, tail.CopyBytes(values, size)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, layout, stages, offset, size, values);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL BindVertexBuffers(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkBuffer>(count) + Bytes<VkDeviceSize>(count), [&](Tail& tail) { return std::tuple{first, count, tail.Copy(buffers, count), tail.Copy(offsets, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, first, count, buffers, offsets);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL CopyBuffer(VkCommandBuffer commands, VkBuffer source, VkBuffer target, std::uint32_t count, const VkBufferCopy* regions) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkBufferCopy>(count), [&](Tail& tail) { return std::tuple{source, target, count, tail.Copy(regions, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, source, target, count, regions);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL CopyBufferToImage(VkCommandBuffer commands, VkBuffer source, VkImage target, VkImageLayout layout, std::uint32_t count, const VkBufferImageCopy* regions) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkBufferImageCopy>(count), [&](Tail& tail) { return std::tuple{source, target, layout, count, tail.Copy(regions, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, source, target, layout, count, regions);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL CopyImageToBuffer(VkCommandBuffer commands, VkImage source, VkImageLayout layout, VkBuffer target, std::uint32_t count, const VkBufferImageCopy* regions) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkBufferImageCopy>(count), [&](Tail& tail) { return std::tuple{source, layout, target, count, tail.Copy(regions, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, source, layout, target, count, regions);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL CopyImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage target, VkImageLayout targetLayout, std::uint32_t count, const VkImageCopy* regions) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkImageCopy>(count), [&](Tail& tail) { return std::tuple{source, sourceLayout, target, targetLayout, count, tail.Copy(regions, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, source, sourceLayout, target, targetLayout, count, regions);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL BlitImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage target, VkImageLayout targetLayout, std::uint32_t count, const VkImageBlit* regions, VkFilter filter) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkImageBlit>(count), [&](Tail& tail) { return std::tuple{source, sourceLayout, target, targetLayout, count, tail.Copy(regions, count), filter}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, source, sourceLayout, target, targetLayout, count, regions, filter);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL ResolveImage(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage target, VkImageLayout targetLayout, std::uint32_t count, const VkImageResolve* regions) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkImageResolve>(count), [&](Tail& tail) { return std::tuple{source, sourceLayout, target, targetLayout, count, tail.Copy(regions, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, source, sourceLayout, target, targetLayout, count, regions);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL ClearColorImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearColorValue* color, std::uint32_t count, const VkImageSubresourceRange* ranges) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkClearColorValue>(1) + Bytes<VkImageSubresourceRange>(count), [&](Tail& tail) { return std::tuple{image, layout, tail.Copy(color, 1), count, tail.Copy(ranges, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, image, layout, color, count, ranges);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL ClearDepthStencilImage(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, std::uint32_t count, const VkImageSubresourceRange* ranges) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkClearDepthStencilValue>(1) + Bytes<VkImageSubresourceRange>(count), [&](Tail& tail) { return std::tuple{image, layout, tail.Copy(value, 1), count, tail.Copy(ranges, count)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, image, layout, value, count, ranges);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL ClearAttachments(VkCommandBuffer commands, std::uint32_t count, const VkClearAttachment* attachments, std::uint32_t rectangles, const VkClearRect* areas) {
    if (Deferring(commands) && Defer(commands, Tag::real, Bytes<VkClearAttachment>(count) + Bytes<VkClearRect>(rectangles), [&](Tail& tail) { return std::tuple{count, tail.Copy(attachments, count), rectangles, tail.Copy(areas, rectangles)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, count, attachments, rectangles, areas);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL UpdateBuffer(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void* data) {
    if (Deferring(commands) && size <= LargestRecord && Defer(commands, Tag::real, Round(static_cast<std::size_t>(size)), [&](Tail& tail) { return std::tuple{buffer, offset, size, tail.CopyBytes(data, static_cast<std::size_t>(size))}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, buffer, offset, size, data);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL PipelineBarrier(VkCommandBuffer commands, VkPipelineStageFlags source, VkPipelineStageFlags target, VkDependencyFlags dependency, std::uint32_t memoryCount, const VkMemoryBarrier* memory, std::uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, std::uint32_t imageCount, const VkImageMemoryBarrier* images) {
    if (Deferring(commands) && !Chained(memory, memoryCount) && !Chained(buffers, bufferCount) && !Chained(images, imageCount) &&
        Defer(commands, Tag::real, Bytes<VkMemoryBarrier>(memoryCount) + Bytes<VkBufferMemoryBarrier>(bufferCount) + Bytes<VkImageMemoryBarrier>(imageCount), [&](Tail& tail) { return std::tuple{source, target, dependency, memoryCount, tail.Copy(memory, memoryCount), bufferCount, tail.Copy(buffers, bufferCount), imageCount, tail.Copy(images, imageCount)}; })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, source, target, dependency, memoryCount, memory, bufferCount, buffers, imageCount, images);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL BeginRenderPass(VkCommandBuffer commands, const VkRenderPassBeginInfo* info, VkSubpassContents contents) {
    if (Deferring(commands) && info != nullptr && info->pNext == nullptr &&
        Defer(commands, Tag::real, Bytes<VkRenderPassBeginInfo>(1) + Bytes<VkClearValue>(info->clearValueCount), [&](Tail& tail) {
            auto* copy = tail.Place(*info);
            copy->pClearValues = tail.Copy(info->pClearValues, info->clearValueCount);
            return std::tuple{static_cast<const VkRenderPassBeginInfo*>(copy), contents};
        })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, info, contents);
}

template<typename Tag>
VKAPI_ATTR void VKAPI_CALL BeginRendering(VkCommandBuffer commands, const VkRenderingInfo* info) {
    const bool plain = info != nullptr && info->pNext == nullptr && !Chained(info->pColorAttachments, info->colorAttachmentCount) && (info->pDepthAttachment == nullptr || info->pDepthAttachment->pNext == nullptr) && (info->pStencilAttachment == nullptr || info->pStencilAttachment->pNext == nullptr);
    if (Deferring(commands) && plain &&
        Defer(commands, Tag::real, Bytes<VkRenderingInfo>(1) + Bytes<VkRenderingAttachmentInfo>(info->colorAttachmentCount) + Bytes<VkRenderingAttachmentInfo>(2), [&](Tail& tail) {
            auto* copy = tail.Place(*info);
            copy->pColorAttachments = tail.Copy(info->pColorAttachments, info->colorAttachmentCount);
            copy->pDepthAttachment = info->pDepthAttachment != nullptr ? tail.Place(*info->pDepthAttachment) : nullptr;
            copy->pStencilAttachment = info->pStencilAttachment != nullptr ? tail.Place(*info->pStencilAttachment) : nullptr;
            return std::tuple{static_cast<const VkRenderingInfo*>(copy)};
        })) return;
    if (Deferring(commands)) Direct(Tag::text);
    Tag::real(commands, info);
}

template<typename Tag>
void Register(State& state, PFN_vkVoidFunction wrapper) {
    Tag::real = reinterpret_cast<typename Tag::Fn>(state.base(state.device, Tag::text));
    state.wrappers.emplace(std::string_view(Tag::text), Tag::real != nullptr ? wrapper : nullptr);
}

template<typename Tag, typename Fn>
PFN_vkVoidFunction Entry(Fn wrapper) {
    static_assert(std::is_same_v<Fn, typename Tag::Fn>);
    return reinterpret_cast<PFN_vkVoidFunction>(wrapper);
}

void RegisterAll(State& state) {
#define APS5_ENCODE_REGISTER_PLAIN(name) Register<Tag_##name>(state, Entry<Tag_##name>(&Plain<Tag_##name>::Call));
#define APS5_ENCODE_REGISTER_FORWARD(name) Register<Tag_##name>(state, Entry<Tag_##name>(&Forward<Tag_##name>::Call));
    APS5_ENCODE_PLAIN(APS5_ENCODE_REGISTER_PLAIN)
    APS5_ENCODE_FORWARD(APS5_ENCODE_REGISTER_FORWARD)
#undef APS5_ENCODE_REGISTER_PLAIN
#undef APS5_ENCODE_REGISTER_FORWARD
    Register<Tag_vkCmdSetViewport>(state, Entry<Tag_vkCmdSetViewport>(&SetViewport<Tag_vkCmdSetViewport>));
    Register<Tag_vkCmdSetScissor>(state, Entry<Tag_vkCmdSetScissor>(&SetScissor<Tag_vkCmdSetScissor>));
    Register<Tag_vkCmdSetViewportWithCount>(state, Entry<Tag_vkCmdSetViewportWithCount>(&SetViewportWithCount<Tag_vkCmdSetViewportWithCount>));
    Register<Tag_vkCmdSetViewportWithCountEXT>(state, Entry<Tag_vkCmdSetViewportWithCountEXT>(&SetViewportWithCount<Tag_vkCmdSetViewportWithCountEXT>));
    Register<Tag_vkCmdSetScissorWithCount>(state, Entry<Tag_vkCmdSetScissorWithCount>(&SetScissorWithCount<Tag_vkCmdSetScissorWithCount>));
    Register<Tag_vkCmdSetScissorWithCountEXT>(state, Entry<Tag_vkCmdSetScissorWithCountEXT>(&SetScissorWithCount<Tag_vkCmdSetScissorWithCountEXT>));
    Register<Tag_vkCmdSetColorWriteMaskEXT>(state, Entry<Tag_vkCmdSetColorWriteMaskEXT>(&SetColorWriteMask<Tag_vkCmdSetColorWriteMaskEXT>));
    Register<Tag_vkCmdBindDescriptorSets>(state, Entry<Tag_vkCmdBindDescriptorSets>(&BindDescriptorSets<Tag_vkCmdBindDescriptorSets>));
    Register<Tag_vkCmdPushConstants>(state, Entry<Tag_vkCmdPushConstants>(&PushConstants<Tag_vkCmdPushConstants>));
    Register<Tag_vkCmdBindVertexBuffers>(state, Entry<Tag_vkCmdBindVertexBuffers>(&BindVertexBuffers<Tag_vkCmdBindVertexBuffers>));
    Register<Tag_vkCmdCopyBuffer>(state, Entry<Tag_vkCmdCopyBuffer>(&CopyBuffer<Tag_vkCmdCopyBuffer>));
    Register<Tag_vkCmdCopyBufferToImage>(state, Entry<Tag_vkCmdCopyBufferToImage>(&CopyBufferToImage<Tag_vkCmdCopyBufferToImage>));
    Register<Tag_vkCmdCopyImageToBuffer>(state, Entry<Tag_vkCmdCopyImageToBuffer>(&CopyImageToBuffer<Tag_vkCmdCopyImageToBuffer>));
    Register<Tag_vkCmdCopyImage>(state, Entry<Tag_vkCmdCopyImage>(&CopyImage<Tag_vkCmdCopyImage>));
    Register<Tag_vkCmdBlitImage>(state, Entry<Tag_vkCmdBlitImage>(&BlitImage<Tag_vkCmdBlitImage>));
    Register<Tag_vkCmdResolveImage>(state, Entry<Tag_vkCmdResolveImage>(&ResolveImage<Tag_vkCmdResolveImage>));
    Register<Tag_vkCmdClearColorImage>(state, Entry<Tag_vkCmdClearColorImage>(&ClearColorImage<Tag_vkCmdClearColorImage>));
    Register<Tag_vkCmdClearDepthStencilImage>(state, Entry<Tag_vkCmdClearDepthStencilImage>(&ClearDepthStencilImage<Tag_vkCmdClearDepthStencilImage>));
    Register<Tag_vkCmdClearAttachments>(state, Entry<Tag_vkCmdClearAttachments>(&ClearAttachments<Tag_vkCmdClearAttachments>));
    Register<Tag_vkCmdUpdateBuffer>(state, Entry<Tag_vkCmdUpdateBuffer>(&UpdateBuffer<Tag_vkCmdUpdateBuffer>));
    Register<Tag_vkCmdPipelineBarrier>(state, Entry<Tag_vkCmdPipelineBarrier>(&PipelineBarrier<Tag_vkCmdPipelineBarrier>));
    Register<Tag_vkCmdBeginRenderPass>(state, Entry<Tag_vkCmdBeginRenderPass>(&BeginRenderPass<Tag_vkCmdBeginRenderPass>));
    Register<Tag_vkCmdBeginRendering>(state, Entry<Tag_vkCmdBeginRendering>(&BeginRendering<Tag_vkCmdBeginRendering>));
    Register<Tag_vkCmdBeginRenderingKHR>(state, Entry<Tag_vkCmdBeginRenderingKHR>(&BeginRendering<Tag_vkCmdBeginRenderingKHR>));
}

}

bool Enabled() {
#ifdef _WIN32
    return false;
#else
    static const bool enabled = std::getenv("APS5_NO_ENCODE_THREAD") == nullptr;
    return enabled;
#endif
}

namespace {

bool PlacementOnly() {
    static const bool placement = std::getenv("APS5_ENCODE_PLACE_ONLY") != nullptr;
    return placement;
}

}

PFN_vkGetDeviceProcAddr Wrap(VkDevice device, PFN_vkGetDeviceProcAddr resolver) {
    if (!Enabled() || device == VK_NULL_HANDLE || resolver == nullptr) return resolver;
    if (active != nullptr) {
        std::fprintf(stderr, "[encode] a second Vulkan device records on its own threads\n");
        return resolver;
    }
    auto* state = new State();
    state->device = device;
    state->base = resolver;
    state->createCommandPool = reinterpret_cast<PFN_vkCreateCommandPool>(resolver(device, "vkCreateCommandPool"));
    state->destroyCommandPool = reinterpret_cast<PFN_vkDestroyCommandPool>(resolver(device, "vkDestroyCommandPool"));
    state->allocateCommandBuffers = reinterpret_cast<PFN_vkAllocateCommandBuffers>(resolver(device, "vkAllocateCommandBuffers"));
    state->beginCommandBuffer = reinterpret_cast<PFN_vkBeginCommandBuffer>(resolver(device, "vkBeginCommandBuffer"));
    state->endCommandBuffer = reinterpret_cast<PFN_vkEndCommandBuffer>(resolver(device, "vkEndCommandBuffer"));
    if (state->createCommandPool == nullptr || state->destroyCommandPool == nullptr || state->allocateCommandBuffers == nullptr || state->beginCommandBuffer == nullptr || state->endCommandBuffer == nullptr) {
        std::fprintf(stderr, "[encode] command buffer entry points missing: encoding stays on the recording threads\n");
        delete state;
        return resolver;
    }
    state->ring = static_cast<std::byte*>(::operator new(RingBytes, std::align_val_t{64}));
    RegisterAll(*state);
    active = state;
    Libc::StartHostThread([state] { Run(*state); }).detach();
    std::size_t wrapped = 0;
    for (const auto& [name, wrapper] : state->wrappers) wrapped += wrapper != nullptr ? 1u : 0u;
    std::fprintf(stderr, "[encode] off-thread encoding %s: %zu of %zu command entry points listed or forwarded, ring %zu MiB\n", PlacementOnly() ? "placed, every batch recorded on its own thread" : "on", wrapped, state->wrappers.size(), RingBytes >> 20u);
    return &Resolve;
}

bool Active() {
    return active != nullptr && !active->disabled.load(std::memory_order_relaxed) && !PlacementOnly();
}

VkCommandBuffer Acquire(std::uint32_t queueFamily) {
    if (!Active()) return VK_NULL_HANDLE;
    auto& state = *active;
    if (state.current.load(std::memory_order_relaxed) != VK_NULL_HANDLE || state.detached != VK_NULL_HANDLE) return VK_NULL_HANDLE;
    std::lock_guard lock(state.poolMutex);
    if (state.pool != VK_NULL_HANDLE && state.poolFamily != queueFamily) return VK_NULL_HANDLE;
    if (!state.spare.empty()) {
        const auto commands = state.spare.back();
        state.spare.pop_back();
        return commands;
    }
    Drain(state);
    if (state.pool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        info.queueFamilyIndex = queueFamily;
        if (state.createCommandPool(state.device, &info, nullptr, &state.pool) != VK_SUCCESS) {
            state.pool = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
        state.poolFamily = queueFamily;
    }
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = state.pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    if (state.allocateCommandBuffers(state.device, &allocation, &commands) != VK_SUCCESS) return VK_NULL_HANDLE;
    return commands;
}

void Begin(VkCommandBuffer commands, VkCommandBufferUsageFlags flags) {
    auto& state = *active;
    state.current.store(commands, std::memory_order_relaxed);
    Emit(commands, &ReplayOpening, Opening, OpeningBody{flags});
    ++state.produced.batches;
}

VkResult End(VkCommandBuffer commands) {
    auto& state = *active;
    if (state.detached == commands) {
        state.detached = VK_NULL_HANDLE;
        state.failure.store(VK_SUCCESS, std::memory_order_relaxed);
        return state.endCommandBuffer(commands);
    }
    if (state.current.load(std::memory_order_relaxed) != commands) return state.endCommandBuffer(commands);
    Emit(commands, &ReplayClosing, Closing, ClosingBody{0});
    Drain(state);
    state.current.store(VK_NULL_HANDLE, std::memory_order_relaxed);
    const auto failed = state.failure.exchange(VK_SUCCESS, std::memory_order_relaxed);
    const auto result = state.ended.exchange(VK_SUCCESS, std::memory_order_relaxed);
    Report(state);
    return failed != VK_SUCCESS ? failed : result;
}

void Retire(VkCommandBuffer commands) noexcept {
    if (active == nullptr || commands == VK_NULL_HANDLE) return;
    auto& state = *active;
    if (state.current.load(std::memory_order_relaxed) == commands || state.detached == commands) static_cast<void>(End(commands));
    std::lock_guard lock(state.poolMutex);
    try {
        state.spare.push_back(commands);
    } catch (...) {
    }
}

void Forget(VkDevice device) noexcept {
    if (active == nullptr || active->device != device) return;
    auto& state = *active;
    state.disabled.store(true, std::memory_order_relaxed);
    if (state.current.load(std::memory_order_relaxed) != VK_NULL_HANDLE || state.detached != VK_NULL_HANDLE) return;
    Drain(state);
    std::lock_guard lock(state.poolMutex);
    if (state.pool != VK_NULL_HANDLE) state.destroyCommandPool(state.device, state.pool, nullptr);
    state.pool = VK_NULL_HANDLE;
    state.spare.clear();
}

Statistics Counts() {
    if (active == nullptr) return {};
    auto counts = active->produced;
    counts.encodeNs = active->encodeNs.load(std::memory_order_relaxed);
    return counts;
}

}
