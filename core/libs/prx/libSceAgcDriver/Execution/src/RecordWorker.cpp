#include "prx/libSceAgcDriver/Execution/include/RecordWorker.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Execution/include/VkCallProfile.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace AgcDriver::RecordWorker {

namespace {

using RunFunction = void (*)(std::byte* payload);

constexpr std::size_t BlockBytes = std::size_t{1} << 20u;
constexpr std::size_t Alignment = 16;

constexpr std::size_t Rounded(std::size_t bytes) {
    return (bytes + Alignment - 1) & ~(Alignment - 1);
}

template<typename T>
constexpr std::size_t Bytes(std::size_t count = 1) {
    return Rounded(sizeof(T) * count);
}

struct Header {
    RunFunction run;
    std::uint32_t bytes;
    std::uint32_t jump;
};
static_assert(sizeof(Header) == Alignment);

struct Block {
    Block* next = nullptr;
    alignas(Alignment) std::byte data[BlockBytes];
};

enum Kind : std::uint8_t { QueuedCall, CustomCall, DrainedCall, DrainedReporting };

#define APS5_V3E_RECORD_CALLS(X) \
    X(vkCmdBindPipeline, QueuedCall) X(vkCmdDispatch, QueuedCall) X(vkCmdDraw, QueuedCall) X(vkCmdDrawIndexed, QueuedCall) \
    X(vkCmdDrawIndirect, QueuedCall) X(vkCmdDrawIndexedIndirect, QueuedCall) X(vkCmdDispatchIndirect, QueuedCall) \
    X(vkCmdDispatchBase, QueuedCall) X(vkCmdEndRenderPass, QueuedCall) X(vkCmdEndRenderingKHR, QueuedCall) \
    X(vkCmdSetDepthBounds, QueuedCall) X(vkCmdSetDepthBias, QueuedCall) X(vkCmdSetStencilReference, QueuedCall) \
    X(vkCmdSetStencilWriteMask, QueuedCall) X(vkCmdSetStencilCompareMask, QueuedCall) X(vkCmdSetStencilTestEnableEXT, QueuedCall) \
    X(vkCmdSetStencilOpEXT, QueuedCall) X(vkCmdSetFrontFaceEXT, QueuedCall) X(vkCmdSetDepthWriteEnableEXT, QueuedCall) \
    X(vkCmdSetDepthTestEnableEXT, QueuedCall) X(vkCmdSetDepthCompareOpEXT, QueuedCall) X(vkCmdSetDepthBoundsTestEnableEXT, QueuedCall) \
    X(vkCmdSetCullModeEXT, QueuedCall) X(vkCmdFillBuffer, QueuedCall) X(vkCmdBindIndexBuffer, QueuedCall) \
    X(vkCmdResetQueryPool, QueuedCall) X(vkCmdBeginQuery, QueuedCall) X(vkCmdEndQuery, QueuedCall) X(vkCmdWriteTimestamp, QueuedCall) \
    X(vkCmdCopyQueryPoolResults, QueuedCall) X(vkCmdDrawMeshTasksEXT, QueuedCall) X(vkCmdDrawMeshTasksIndirectEXT, QueuedCall) \
    X(vkCmdDrawIndirectCountKHR, QueuedCall) X(vkCmdDrawIndexedIndirectCountKHR, QueuedCall) X(vkCmdWriteBufferMarkerAMD, QueuedCall) \
    X(vkCmdSetCheckpointNV, QueuedCall) X(vkEndCommandBuffer, QueuedCall) X(vkResetCommandBuffer, QueuedCall) \
    X(vkCmdPipelineBarrier, CustomCall) X(vkCmdCopyBuffer, CustomCall) X(vkCmdPushConstants, CustomCall) \
    X(vkCmdBindDescriptorSets, CustomCall) X(vkCmdCopyImageToBuffer, CustomCall) X(vkCmdUpdateBuffer, CustomCall) \
    X(vkCmdCopyBufferToImage, CustomCall) X(vkCmdClearColorImage, CustomCall) X(vkCmdCopyImage, CustomCall) \
    X(vkCmdBeginRenderPass, CustomCall) X(vkCmdSetViewport, CustomCall) X(vkCmdSetScissor, CustomCall) X(vkCmdBlitImage, CustomCall) \
    X(vkCmdClearDepthStencilImage, CustomCall) X(vkCmdBindVertexBuffers, CustomCall) X(vkCmdBeginRenderingKHR, CustomCall) \
    X(vkBeginCommandBuffer, CustomCall) \
    X(vkQueueSubmit, DrainedReporting) X(vkQueueSubmit2KHR, DrainedReporting) X(vkQueueBindSparse, DrainedReporting) \
    X(vkQueuePresentKHR, DrainedReporting) X(vkQueueWaitIdle, DrainedCall) X(vkDeviceWaitIdle, DrainedCall) \
    X(vkAllocateCommandBuffers, DrainedCall) X(vkFreeCommandBuffers, DrainedCall) X(vkResetCommandPool, DrainedCall) \
    X(vkTrimCommandPool, DrainedCall) X(vkDestroyCommandPool, DrainedCall) X(vkDestroyDevice, DrainedCall) \
    X(vkDestroyBuffer, DrainedCall) X(vkDestroyImage, DrainedCall) X(vkDestroyImageView, DrainedCall) X(vkFreeMemory, DrainedCall) \
    X(vkDestroyPipeline, DrainedCall) X(vkDestroyPipelineLayout, DrainedCall) X(vkDestroyDescriptorSetLayout, DrainedCall) \
    X(vkDestroyDescriptorPool, DrainedCall) X(vkFreeDescriptorSets, DrainedCall) X(vkResetDescriptorPool, DrainedCall) \
    X(vkDestroySampler, DrainedCall) X(vkDestroyFramebuffer, DrainedCall) X(vkDestroyRenderPass, DrainedCall) \
    X(vkDestroyQueryPool, DrainedCall) X(vkDestroyBufferView, DrainedCall) X(vkDestroyEvent, DrainedCall)

enum CallId : std::size_t {
#define APS5_V3E_RECORD_ID(name, kind) Id_##name,
    APS5_V3E_RECORD_CALLS(APS5_V3E_RECORD_ID)
#undef APS5_V3E_RECORD_ID
    CallCount
};

constexpr std::array<const char*, CallCount> Names{{
#define APS5_V3E_RECORD_NAME(name, kind) #name,
    APS5_V3E_RECORD_CALLS(APS5_V3E_RECORD_NAME)
#undef APS5_V3E_RECORD_NAME
}};

struct State {
    std::mutex producer;
    Block* write = nullptr;
    std::size_t writeOffset = 0;
    std::uint64_t pushed = 0;
    std::atomic<std::uint64_t> published{0};
    Block* read = nullptr;
    std::size_t readOffset = 0;
    std::atomic<std::uint64_t> executed{0};
    std::mutex freeMutex;
    std::vector<Block*> freeBlocks;
    std::mutex sleepMutex;
    std::condition_variable workerWake;
    std::condition_variable drained;
    std::atomic<bool> workerSleeping{false};
    std::atomic<int> drainSleepers{0};
    std::thread worker;
    std::atomic<bool> started{false};
    std::atomic<VkResult> failure{VK_SUCCESS};
    std::atomic<const char*> failedCall{nullptr};
    std::atomic<std::uint64_t> queued{0}, queuedBytes{0}, direct{0}, drains{0}, drainWaits{0}, drainWaitNs{0}, workerNs{0}, workerWakes{0};
    std::array<std::uint64_t, 8> reported{};
};

State& state() {
    static auto* created = new State();
    return *created;
}

thread_local bool onWorker = false;

void noteFailure(VkResult result, const char* call) {
    if (result == VK_SUCCESS) return;
    auto& s = state();
    VkResult expected = VK_SUCCESS;
    if (s.failure.compare_exchange_strong(expected, result)) s.failedCall.store(call, std::memory_order_release);
}

Block* takeBlock() {
    auto& s = state();
    {
        std::lock_guard lock(s.freeMutex);
        if (!s.freeBlocks.empty()) {
            auto* block = s.freeBlocks.back();
            s.freeBlocks.pop_back();
            block->next = nullptr;
            return block;
        }
    }
    return new Block();
}

void recycle(Block* block) {
    auto& s = state();
    std::lock_guard lock(s.freeMutex);
    s.freeBlocks.push_back(block);
}

void workerLoop() {
    onWorker = true;
    VkCallProfile::NameThread("record worker");
    auto& s = state();
    std::uint64_t done = s.executed.load(std::memory_order_relaxed);
    for (;;) {
        SpinUntil(HandoffSpinWindow(), [&] { return s.published.load(std::memory_order_acquire) != done; });
        if (s.published.load(std::memory_order_acquire) == done) {
            std::unique_lock lock(s.sleepMutex);
            s.workerSleeping.store(true, std::memory_order_seq_cst);
            s.workerWake.wait(lock, [&] { return s.published.load(std::memory_order_seq_cst) != done; });
            s.workerSleeping.store(false, std::memory_order_relaxed);
            s.workerWakes.fetch_add(1, std::memory_order_relaxed);
        }
        const auto start = std::chrono::steady_clock::now();
        for (;;) {
            const auto target = s.published.load(std::memory_order_acquire);
            if (target == done) break;
            while (done != target) {
                auto* header = reinterpret_cast<Header*>(s.read->data + s.readOffset);
                if (header->jump != 0) {
                    auto* old = s.read;
                    s.read = old->next;
                    s.readOffset = 0;
                    recycle(old);
                    continue;
                }
                header->run(s.read->data + s.readOffset + sizeof(Header));
                s.readOffset += sizeof(Header) + header->bytes;
                ++done;
                s.executed.store(done, std::memory_order_seq_cst);
                if (s.drainSleepers.load(std::memory_order_seq_cst) != 0) {
                    std::lock_guard lock(s.sleepMutex);
                    s.drained.notify_all();
                }
            }
        }
        s.workerNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
    }
}

void startLocked() {
    auto& s = state();
    if (s.started.load(std::memory_order_relaxed)) return;
    s.write = s.read = takeBlock();
    s.writeOffset = s.readOffset = 0;
    s.worker = std::thread(workerLoop);
    s.worker.detach();
    s.started.store(true, std::memory_order_release);
}

void drainTo(std::uint64_t target) {
    auto& s = state();
    if (onWorker) return;
    s.drains.fetch_add(1, std::memory_order_relaxed);
    if (s.executed.load(std::memory_order_acquire) >= target) return;
    const auto start = std::chrono::steady_clock::now();
    s.drainWaits.fetch_add(1, std::memory_order_relaxed);
    SpinUntil(HandoffSpinWindow() * 2, [&] { return s.executed.load(std::memory_order_acquire) >= target; });
    if (s.executed.load(std::memory_order_acquire) < target) {
        std::unique_lock lock(s.sleepMutex);
        s.drainSleepers.fetch_add(1, std::memory_order_seq_cst);
        s.drained.wait(lock, [&] { return s.executed.load(std::memory_order_seq_cst) >= target; });
        s.drainSleepers.fetch_sub(1, std::memory_order_seq_cst);
    }
    s.drainWaitNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()), std::memory_order_relaxed);
}

template<typename TCall>
auto direct(TCall&& call) {
    auto& s = state();
    std::lock_guard lock(s.producer);
    drainTo(s.pushed);
    s.direct.fetch_add(1, std::memory_order_relaxed);
    return call();
}

class Packet {
public:
    Packet(std::size_t bytes, RunFunction run) : lock(state().producer) {
        auto& s = state();
        startLocked();
        const auto total = sizeof(Header) + bytes;
        if (s.writeOffset + total + sizeof(Header) > BlockBytes) {
            auto* next = takeBlock();
            auto* jump = reinterpret_cast<Header*>(s.write->data + s.writeOffset);
            jump->run = nullptr;
            jump->bytes = 0;
            jump->jump = 1;
            s.write->next = next;
            s.write = next;
            s.writeOffset = 0;
        }
        auto* header = reinterpret_cast<Header*>(s.write->data + s.writeOffset);
        header->run = run;
        header->bytes = static_cast<std::uint32_t>(bytes);
        header->jump = 0;
        this->header = header;
        base = s.write->data + s.writeOffset + sizeof(Header);
        s.writeOffset += total;
        capacity = bytes;
    }
    Packet(const Packet&) = delete;
    Packet& operator=(const Packet&) = delete;
    ~Packet() {
        auto& s = state();
        if (std::uncaught_exceptions() != 0) header->run = &skip;
        ++s.pushed;
        s.queued.fetch_add(1, std::memory_order_relaxed);
        s.queuedBytes.fetch_add(capacity, std::memory_order_relaxed);
        s.published.store(s.pushed, std::memory_order_seq_cst);
        if (s.workerSleeping.load(std::memory_order_seq_cst)) {
            std::lock_guard sleep(s.sleepMutex);
            s.workerWake.notify_one();
        }
    }
    template<typename T, typename... TArgs>
    T* Make(TArgs&&... args) {
        auto* at = take(Bytes<T>());
        return new (at) T{std::forward<TArgs>(args)...};
    }
    template<typename T>
    const T* Copy(const T* source, std::size_t count) {
        if (source == nullptr || count == 0) return source;
        auto* at = take(Bytes<T>(count));
        std::memcpy(at, source, sizeof(T) * count);
        return reinterpret_cast<const T*>(at);
    }
    const void* CopyBytes(const void* source, std::size_t bytes) {
        if (source == nullptr || bytes == 0) return source;
        auto* at = take(Rounded(bytes));
        std::memcpy(at, source, bytes);
        return at;
    }

private:
    static void skip(std::byte*) {}
    std::byte* take(std::size_t bytes) {
        if (used + bytes > capacity) throw std::logic_error("AGC record worker: packet layout exceeds its size");
        auto* at = base + used;
        used += bytes;
        return at;
    }
    std::unique_lock<std::mutex> lock;
    Header* header = nullptr;
    std::byte* base = nullptr;
    std::size_t used = 0;
    std::size_t capacity = 0;
};

constexpr std::size_t PacketLimit = BlockBytes / 2;

template<typename T>
bool Chained(const T* items, std::size_t count) {
    for (std::size_t i = 0; i < count && items != nullptr; ++i) {
        if (items[i].pNext != nullptr) return true;
    }
    return false;
}

template<CallId Id, typename TFunction>
struct Real {
    static inline std::atomic<TFunction> function{nullptr};
    static TFunction Get() { return function.load(std::memory_order_relaxed); }
};

template<CallId Id, typename TFunction>
struct Queued;

template<CallId Id, typename TResult, typename... TArgs>
struct Queued<Id, TResult (VKAPI_PTR*)(TArgs...)> {
    using Function = TResult (VKAPI_PTR*)(TArgs...);
    using Arguments = std::tuple<TArgs...>;
    static_assert(std::is_trivially_destructible_v<Arguments> && (std::is_trivially_copyable_v<TArgs> && ...));
    static void Run(std::byte* payload) {
        auto& arguments = *reinterpret_cast<Arguments*>(payload);
        if constexpr (std::is_same_v<TResult, VkResult>) noteFailure(std::apply(Real<Id, Function>::Get(), arguments), Names[Id]);
        else std::apply(Real<Id, Function>::Get(), arguments);
    }
    static TResult VKAPI_CALL Call(TArgs... args) {
        {
            Packet packet(Bytes<Arguments>(), &Run);
            packet.Make<Arguments>(args...);
        }
        if constexpr (std::is_same_v<TResult, VkResult>) return VK_SUCCESS;
    }
};

template<CallId Id, typename TFunction, bool ReportsFailure>
struct Drained;

template<CallId Id, typename TResult, typename... TArgs, bool ReportsFailure>
struct Drained<Id, TResult (VKAPI_PTR*)(TArgs...), ReportsFailure> {
    using Function = TResult (VKAPI_PTR*)(TArgs...);
    static TResult VKAPI_CALL Call(TArgs... args) {
        return direct([&]() -> TResult {
            if constexpr (ReportsFailure && std::is_same_v<TResult, VkResult>) {
                const auto failed = state().failure.load(std::memory_order_acquire);
                if (failed != VK_SUCCESS) return failed;
            }
            return Real<Id, Function>::Get()(args...);
        });
    }
};

template<CallId Id>
struct Custom;

#define APS5_V3E_REAL(name) Real<Id_##name, PFN_##name>::Get()

template<>
struct Custom<Id_vkCmdPipelineBarrier> {
    struct Args {
        VkCommandBuffer commands;
        VkPipelineStageFlags source, destination;
        VkDependencyFlags dependency;
        std::uint32_t memoryCount;
        const VkMemoryBarrier* memory;
        std::uint32_t bufferCount;
        const VkBufferMemoryBarrier* buffers;
        std::uint32_t imageCount;
        const VkImageMemoryBarrier* images;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdPipelineBarrier)(a.commands, a.source, a.destination, a.dependency, a.memoryCount, a.memory, a.bufferCount, a.buffers, a.imageCount, a.images);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkDependencyFlags dependency, std::uint32_t memoryCount, const VkMemoryBarrier* memory, std::uint32_t bufferCount, const VkBufferMemoryBarrier* buffers, std::uint32_t imageCount, const VkImageMemoryBarrier* images) {
        const auto bytes = Bytes<Args>() + Bytes<VkMemoryBarrier>(memoryCount) + Bytes<VkBufferMemoryBarrier>(bufferCount) + Bytes<VkImageMemoryBarrier>(imageCount);
        if (bytes > PacketLimit || Chained(memory, memoryCount) || Chained(buffers, bufferCount) || Chained(images, imageCount)) {
            return direct([&] { APS5_V3E_REAL(vkCmdPipelineBarrier)(commands, source, destination, dependency, memoryCount, memory, bufferCount, buffers, imageCount, images); });
        }
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, source, destination, dependency, memoryCount, nullptr, bufferCount, nullptr, imageCount, nullptr});
        a->memory = packet.Copy(memory, memoryCount);
        a->buffers = packet.Copy(buffers, bufferCount);
        a->images = packet.Copy(images, imageCount);
    }
};

template<>
struct Custom<Id_vkCmdCopyBuffer> {
    struct Args {
        VkCommandBuffer commands;
        VkBuffer source, destination;
        std::uint32_t count;
        const VkBufferCopy* regions;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdCopyBuffer)(a.commands, a.source, a.destination, a.count, a.regions);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t count, const VkBufferCopy* regions) {
        const auto bytes = Bytes<Args>() + Bytes<VkBufferCopy>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdCopyBuffer)(commands, source, destination, count, regions); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, source, destination, count, nullptr});
        a->regions = packet.Copy(regions, count);
    }
};

template<>
struct Custom<Id_vkCmdPushConstants> {
    struct Args {
        VkCommandBuffer commands;
        VkPipelineLayout layout;
        VkShaderStageFlags stages;
        std::uint32_t offset, size;
        const void* values;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdPushConstants)(a.commands, a.layout, a.stages, a.offset, a.size, a.values);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset, std::uint32_t size, const void* values) {
        const auto bytes = Bytes<Args>() + Rounded(size);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdPushConstants)(commands, layout, stages, offset, size, values); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, layout, stages, offset, size, nullptr});
        a->values = packet.CopyBytes(values, size);
    }
};

template<>
struct Custom<Id_vkCmdBindDescriptorSets> {
    struct Args {
        VkCommandBuffer commands;
        VkPipelineBindPoint point;
        VkPipelineLayout layout;
        std::uint32_t first, count;
        const VkDescriptorSet* sets;
        std::uint32_t dynamicCount;
        const std::uint32_t* dynamicOffsets;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdBindDescriptorSets)(a.commands, a.point, a.layout, a.first, a.count, a.sets, a.dynamicCount, a.dynamicOffsets);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkPipelineBindPoint point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets, std::uint32_t dynamicCount, const std::uint32_t* dynamicOffsets) {
        const auto bytes = Bytes<Args>() + Bytes<VkDescriptorSet>(count) + Bytes<std::uint32_t>(dynamicCount);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdBindDescriptorSets)(commands, point, layout, first, count, sets, dynamicCount, dynamicOffsets); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, point, layout, first, count, nullptr, dynamicCount, nullptr});
        a->sets = packet.Copy(sets, count);
        a->dynamicOffsets = packet.Copy(dynamicOffsets, dynamicCount);
    }
};

template<>
struct Custom<Id_vkCmdCopyImageToBuffer> {
    struct Args {
        VkCommandBuffer commands;
        VkImage image;
        VkImageLayout layout;
        VkBuffer buffer;
        std::uint32_t count;
        const VkBufferImageCopy* regions;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdCopyImageToBuffer)(a.commands, a.image, a.layout, a.buffer, a.count, a.regions);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkImage image, VkImageLayout layout, VkBuffer buffer, std::uint32_t count, const VkBufferImageCopy* regions) {
        const auto bytes = Bytes<Args>() + Bytes<VkBufferImageCopy>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdCopyImageToBuffer)(commands, image, layout, buffer, count, regions); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, image, layout, buffer, count, nullptr});
        a->regions = packet.Copy(regions, count);
    }
};

template<>
struct Custom<Id_vkCmdUpdateBuffer> {
    struct Args {
        VkCommandBuffer commands;
        VkBuffer buffer;
        VkDeviceSize offset, size;
        const void* data;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdUpdateBuffer)(a.commands, a.buffer, a.offset, a.size, a.data);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void* data) {
        const auto bytes = Bytes<Args>() + Rounded(static_cast<std::size_t>(size));
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdUpdateBuffer)(commands, buffer, offset, size, data); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, buffer, offset, size, nullptr});
        a->data = packet.CopyBytes(data, static_cast<std::size_t>(size));
    }
};

template<>
struct Custom<Id_vkCmdCopyBufferToImage> {
    struct Args {
        VkCommandBuffer commands;
        VkBuffer buffer;
        VkImage image;
        VkImageLayout layout;
        std::uint32_t count;
        const VkBufferImageCopy* regions;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdCopyBufferToImage)(a.commands, a.buffer, a.image, a.layout, a.count, a.regions);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkBuffer buffer, VkImage image, VkImageLayout layout, std::uint32_t count, const VkBufferImageCopy* regions) {
        const auto bytes = Bytes<Args>() + Bytes<VkBufferImageCopy>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdCopyBufferToImage)(commands, buffer, image, layout, count, regions); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, buffer, image, layout, count, nullptr});
        a->regions = packet.Copy(regions, count);
    }
};

template<>
struct Custom<Id_vkCmdClearColorImage> {
    struct Args {
        VkCommandBuffer commands;
        VkImage image;
        VkImageLayout layout;
        const VkClearColorValue* color;
        std::uint32_t count;
        const VkImageSubresourceRange* ranges;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdClearColorImage)(a.commands, a.image, a.layout, a.color, a.count, a.ranges);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearColorValue* color, std::uint32_t count, const VkImageSubresourceRange* ranges) {
        const auto bytes = Bytes<Args>() + Bytes<VkClearColorValue>() + Bytes<VkImageSubresourceRange>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdClearColorImage)(commands, image, layout, color, count, ranges); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, image, layout, nullptr, count, nullptr});
        a->color = packet.Copy(color, 1);
        a->ranges = packet.Copy(ranges, count);
    }
};

template<>
struct Custom<Id_vkCmdCopyImage> {
    struct Args {
        VkCommandBuffer commands;
        VkImage source;
        VkImageLayout sourceLayout;
        VkImage destination;
        VkImageLayout destinationLayout;
        std::uint32_t count;
        const VkImageCopy* regions;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdCopyImage)(a.commands, a.source, a.sourceLayout, a.destination, a.destinationLayout, a.count, a.regions);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageCopy* regions) {
        const auto bytes = Bytes<Args>() + Bytes<VkImageCopy>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdCopyImage)(commands, source, sourceLayout, destination, destinationLayout, count, regions); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, source, sourceLayout, destination, destinationLayout, count, nullptr});
        a->regions = packet.Copy(regions, count);
    }
};

template<>
struct Custom<Id_vkCmdBeginRenderPass> {
    struct Args {
        VkCommandBuffer commands;
        VkRenderPassBeginInfo info;
        VkSubpassContents contents;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdBeginRenderPass)(a.commands, &a.info, a.contents);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, const VkRenderPassBeginInfo* info, VkSubpassContents contents) {
        const auto bytes = Bytes<Args>() + Bytes<VkClearValue>(info->clearValueCount);
        if (bytes > PacketLimit || info->pNext != nullptr) return direct([&] { APS5_V3E_REAL(vkCmdBeginRenderPass)(commands, info, contents); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, *info, contents});
        a->info.pClearValues = packet.Copy(info->pClearValues, info->clearValueCount);
    }
};

template<>
struct Custom<Id_vkCmdSetViewport> {
    struct Args {
        VkCommandBuffer commands;
        std::uint32_t first, count;
        const VkViewport* viewports;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdSetViewport)(a.commands, a.first, a.count, a.viewports);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkViewport* viewports) {
        const auto bytes = Bytes<Args>() + Bytes<VkViewport>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdSetViewport)(commands, first, count, viewports); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, first, count, nullptr});
        a->viewports = packet.Copy(viewports, count);
    }
};

template<>
struct Custom<Id_vkCmdSetScissor> {
    struct Args {
        VkCommandBuffer commands;
        std::uint32_t first, count;
        const VkRect2D* scissors;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdSetScissor)(a.commands, a.first, a.count, a.scissors);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkRect2D* scissors) {
        const auto bytes = Bytes<Args>() + Bytes<VkRect2D>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdSetScissor)(commands, first, count, scissors); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, first, count, nullptr});
        a->scissors = packet.Copy(scissors, count);
    }
};

template<>
struct Custom<Id_vkCmdBlitImage> {
    struct Args {
        VkCommandBuffer commands;
        VkImage source;
        VkImageLayout sourceLayout;
        VkImage destination;
        VkImageLayout destinationLayout;
        std::uint32_t count;
        const VkImageBlit* regions;
        VkFilter filter;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdBlitImage)(a.commands, a.source, a.sourceLayout, a.destination, a.destinationLayout, a.count, a.regions, a.filter);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkImage source, VkImageLayout sourceLayout, VkImage destination, VkImageLayout destinationLayout, std::uint32_t count, const VkImageBlit* regions, VkFilter filter) {
        const auto bytes = Bytes<Args>() + Bytes<VkImageBlit>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdBlitImage)(commands, source, sourceLayout, destination, destinationLayout, count, regions, filter); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, source, sourceLayout, destination, destinationLayout, count, nullptr, filter});
        a->regions = packet.Copy(regions, count);
    }
};

template<>
struct Custom<Id_vkCmdClearDepthStencilImage> {
    struct Args {
        VkCommandBuffer commands;
        VkImage image;
        VkImageLayout layout;
        const VkClearDepthStencilValue* value;
        std::uint32_t count;
        const VkImageSubresourceRange* ranges;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdClearDepthStencilImage)(a.commands, a.image, a.layout, a.value, a.count, a.ranges);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, VkImage image, VkImageLayout layout, const VkClearDepthStencilValue* value, std::uint32_t count, const VkImageSubresourceRange* ranges) {
        const auto bytes = Bytes<Args>() + Bytes<VkClearDepthStencilValue>() + Bytes<VkImageSubresourceRange>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdClearDepthStencilImage)(commands, image, layout, value, count, ranges); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, image, layout, nullptr, count, nullptr});
        a->value = packet.Copy(value, 1);
        a->ranges = packet.Copy(ranges, count);
    }
};

template<>
struct Custom<Id_vkCmdBindVertexBuffers> {
    struct Args {
        VkCommandBuffer commands;
        std::uint32_t first, count;
        const VkBuffer* buffers;
        const VkDeviceSize* offsets;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdBindVertexBuffers)(a.commands, a.first, a.count, a.buffers, a.offsets);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets) {
        const auto bytes = Bytes<Args>() + Bytes<VkBuffer>(count) + Bytes<VkDeviceSize>(count);
        if (bytes > PacketLimit) return direct([&] { APS5_V3E_REAL(vkCmdBindVertexBuffers)(commands, first, count, buffers, offsets); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, first, count, nullptr, nullptr});
        a->buffers = packet.Copy(buffers, count);
        a->offsets = packet.Copy(offsets, count);
    }
};

template<>
struct Custom<Id_vkCmdBeginRenderingKHR> {
    struct Args {
        VkCommandBuffer commands;
        VkRenderingInfoKHR info;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        APS5_V3E_REAL(vkCmdBeginRenderingKHR)(a.commands, &a.info);
    }
    static void VKAPI_CALL Call(VkCommandBuffer commands, const VkRenderingInfoKHR* info) {
        const auto bytes = Bytes<Args>() + Bytes<VkRenderingAttachmentInfoKHR>(info->colorAttachmentCount) + 2 * Bytes<VkRenderingAttachmentInfoKHR>();
        const bool chained = info->pNext != nullptr || Chained(info->pColorAttachments, info->colorAttachmentCount) || Chained(info->pDepthAttachment, 1) || Chained(info->pStencilAttachment, 1);
        if (bytes > PacketLimit || chained) return direct([&] { APS5_V3E_REAL(vkCmdBeginRenderingKHR)(commands, info); });
        Packet packet(bytes, &Run);
        auto* a = packet.Make<Args>(Args{commands, *info});
        a->info.pColorAttachments = packet.Copy(info->pColorAttachments, info->colorAttachmentCount);
        if (info->pDepthAttachment != nullptr && info->pDepthAttachment == info->pStencilAttachment) {
            a->info.pDepthAttachment = a->info.pStencilAttachment = packet.Copy(info->pDepthAttachment, 1);
        } else {
            a->info.pDepthAttachment = packet.Copy(info->pDepthAttachment, 1);
            a->info.pStencilAttachment = packet.Copy(info->pStencilAttachment, 1);
        }
    }
};

template<>
struct Custom<Id_vkBeginCommandBuffer> {
    struct Args {
        VkCommandBuffer commands;
        VkCommandBufferBeginInfo info;
    };
    static void Run(std::byte* payload) {
        const auto& a = *reinterpret_cast<const Args*>(payload);
        noteFailure(APS5_V3E_REAL(vkBeginCommandBuffer)(a.commands, &a.info), "vkBeginCommandBuffer");
    }
    static VkResult VKAPI_CALL Call(VkCommandBuffer commands, const VkCommandBufferBeginInfo* info) {
        if (info->pNext != nullptr || info->pInheritanceInfo != nullptr) return direct([&] { return APS5_V3E_REAL(vkBeginCommandBuffer)(commands, info); });
        Packet packet(Bytes<Args>(), &Run);
        packet.Make<Args>(Args{commands, *info});
        return VK_SUCCESS;
    }
};

#undef APS5_V3E_REAL

using Installer = PFN_vkVoidFunction (*)(PFN_vkVoidFunction);

template<CallId Id, typename TFunction, Kind K>
PFN_vkVoidFunction Install(PFN_vkVoidFunction resolved) {
    auto& slot = Real<Id, TFunction>::function;
    const auto function = reinterpret_cast<TFunction>(resolved);
    TFunction expected = nullptr;
    if (!slot.compare_exchange_strong(expected, function) && expected != function) throw std::runtime_error("AGC record worker: a Vulkan entry point resolved to two different functions");
    if constexpr (K == QueuedCall) {
        return reinterpret_cast<PFN_vkVoidFunction>(&Queued<Id, TFunction>::Call);
    } else if constexpr (K == CustomCall) {
        return reinterpret_cast<PFN_vkVoidFunction>(&Custom<Id>::Call);
    } else {
        return reinterpret_cast<PFN_vkVoidFunction>(&Drained<Id, TFunction, K == DrainedReporting>::Call);
    }
}

struct Entry {
    const char* name;
    Installer install;
};

constexpr std::array<Entry, CallCount> Entries{{
#define APS5_V3E_RECORD_ENTRY(name, kind) {#name, &Install<Id_##name, PFN_##name, kind>},
    APS5_V3E_RECORD_CALLS(APS5_V3E_RECORD_ENTRY)
#undef APS5_V3E_RECORD_ENTRY
}};

std::atomic<PFN_vkGetDeviceProcAddr> inner{nullptr};

PFN_vkVoidFunction VKAPI_CALL resolve(VkDevice device, const char* name) {
    const auto resolved = inner.load(std::memory_order_acquire)(device, name);
    if (resolved == nullptr || name == nullptr || name[0] != 'v' || name[1] != 'k') return resolved;
    for (const auto& entry : Entries) {
        if (std::strcmp(entry.name, name) == 0) return entry.install(resolved);
    }
    if (std::strncmp(name, "vkCmd", 5) == 0 || std::strncmp(name, "vkQueue", 7) == 0) throw std::runtime_error(std::string("AGC record worker: no rule for ") + name);
    return resolved;
}

}

bool Enabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("APS5_V3E_RECORD_WORKER");
        return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0;
    }();
    return enabled;
}

PFN_vkGetDeviceProcAddr Wrap(PFN_vkGetDeviceProcAddr resolver) {
    if (!Enabled() || resolver == nullptr) return resolver;
    if (resolver == &resolve) return resolver;
    inner.store(resolver, std::memory_order_release);
    return &resolve;
}

void Drain() {
    if (!state().started.load(std::memory_order_acquire)) return;
    {
        auto& s = state();
        std::lock_guard lock(s.producer);
        drainTo(s.pushed);
    }
    const auto failed = state().failure.load(std::memory_order_acquire);
    if (failed != VK_SUCCESS) {
        const char* call = state().failedCall.load(std::memory_order_acquire);
        throw std::runtime_error(std::string("AGC record worker: ") + (call != nullptr ? call : "a queued call") + " failed with VkResult " + std::to_string(static_cast<int>(failed)));
    }
}

Counters Totals() {
    const auto& s = state();
    constexpr auto relaxed = std::memory_order_relaxed;
    return {s.queued.load(relaxed), s.queuedBytes.load(relaxed), s.direct.load(relaxed), s.drains.load(relaxed), s.drainWaits.load(relaxed), s.drainWaitNs.load(relaxed), s.workerNs.load(relaxed), s.workerWakes.load(relaxed)};
}

void Close(const char* label) {
    if (!Enabled()) return;
    auto& s = state();
    const auto now = Totals();
    const std::array<std::uint64_t, 8> values{now.queued, now.queuedBytes, now.direct, now.drains, now.drainWaits, now.drainWaitNs, now.workerNs, now.workerWakes};
    std::array<std::uint64_t, 8> delta{};
    for (std::size_t i = 0; i < values.size(); ++i) {
        delta[i] = values[i] - s.reported[i];
        s.reported[i] = values[i];
    }
    AgcDriver::ProfilePrint_nid_no_patch("[recworker] %s: %llu calls queued (%.1f MiB), %llu made on the caller after a drain; %llu drains, %llu waited %.1f ms; worker busy %.1f ms, %llu wakes\n", label, static_cast<unsigned long long>(delta[0]), static_cast<double>(delta[1]) / 1048576.0, static_cast<unsigned long long>(delta[2]), static_cast<unsigned long long>(delta[3]), static_cast<unsigned long long>(delta[4]), static_cast<double>(delta[5]) / 1e6, static_cast<double>(delta[6]) / 1e6, static_cast<unsigned long long>(delta[7]));
}

void ForTestsReset(PFN_vkGetDeviceProcAddr resolver) {
    Drain();
#define APS5_V3E_RECORD_CLEAR(name, kind) Real<Id_##name, PFN_##name>::function.store(nullptr, std::memory_order_relaxed);
    APS5_V3E_RECORD_CALLS(APS5_V3E_RECORD_CLEAR)
#undef APS5_V3E_RECORD_CLEAR
    state().failure.store(VK_SUCCESS, std::memory_order_relaxed);
    inner.store(resolver, std::memory_order_release);
}

}
