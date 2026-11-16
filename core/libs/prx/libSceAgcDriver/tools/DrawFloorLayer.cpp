#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <windows.h>
#include <intrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

using Tick = std::uint64_t;

inline Tick Now() { return __rdtsc(); }

enum class Group : std::uint8_t { Draw, Dynamic, Other, Sets, Buffers, Submit, Objects, Poll, Wait, None };

#define APS5_FLOOR_SLOTS(X) \
    X(CmdBindPipeline, "vkCmdBindPipeline", Draw) \
    X(CmdBindDescriptorSets, "vkCmdBindDescriptorSets", Draw) \
    X(CmdPushConstants, "vkCmdPushConstants", Draw) \
    X(CmdBindVertexBuffers, "vkCmdBindVertexBuffers", Draw) \
    X(CmdBindIndexBuffer, "vkCmdBindIndexBuffer", Draw) \
    X(CmdDraw, "vkCmdDraw", Draw) \
    X(CmdDrawIndexed, "vkCmdDrawIndexed", Draw) \
    X(CmdDrawIndirect, "vkCmdDrawIndirect", Draw) \
    X(CmdDrawIndexedIndirect, "vkCmdDrawIndexedIndirect", Draw) \
    X(CmdDrawIndirectCountKHR, "vkCmdDrawIndirectCountKHR", Draw) \
    X(CmdDrawIndirectCount, "vkCmdDrawIndirectCount", Draw) \
    X(CmdDrawIndexedIndirectCountKHR, "vkCmdDrawIndexedIndirectCountKHR", Draw) \
    X(CmdDrawIndexedIndirectCount, "vkCmdDrawIndexedIndirectCount", Draw) \
    X(CmdDrawMeshTasksEXT, "vkCmdDrawMeshTasksEXT", Draw) \
    X(CmdDrawMeshTasksIndirectEXT, "vkCmdDrawMeshTasksIndirectEXT", Draw) \
    X(CmdBeginRenderingKHR, "vkCmdBeginRenderingKHR", Draw) \
    X(CmdBeginRendering, "vkCmdBeginRendering", Draw) \
    X(CmdEndRenderingKHR, "vkCmdEndRenderingKHR", Draw) \
    X(CmdEndRendering, "vkCmdEndRendering", Draw) \
    X(CmdBeginRenderPass, "vkCmdBeginRenderPass", Draw) \
    X(CmdEndRenderPass, "vkCmdEndRenderPass", Draw) \
    X(CmdExecuteCommands, "vkCmdExecuteCommands", Draw) \
    X(CmdSetViewport, "vkCmdSetViewport", Dynamic) \
    X(CmdSetScissor, "vkCmdSetScissor", Dynamic) \
    X(CmdSetViewportWithCountEXT, "vkCmdSetViewportWithCountEXT", Dynamic) \
    X(CmdSetViewportWithCount, "vkCmdSetViewportWithCount", Dynamic) \
    X(CmdSetScissorWithCountEXT, "vkCmdSetScissorWithCountEXT", Dynamic) \
    X(CmdSetScissorWithCount, "vkCmdSetScissorWithCount", Dynamic) \
    X(CmdSetColorWriteMaskEXT, "vkCmdSetColorWriteMaskEXT", Dynamic) \
    X(CmdSetDepthBounds, "vkCmdSetDepthBounds", Dynamic) \
    X(CmdSetDepthBias, "vkCmdSetDepthBias", Dynamic) \
    X(CmdSetCullModeEXT, "vkCmdSetCullModeEXT", Dynamic) \
    X(CmdSetCullMode, "vkCmdSetCullMode", Dynamic) \
    X(CmdSetFrontFaceEXT, "vkCmdSetFrontFaceEXT", Dynamic) \
    X(CmdSetFrontFace, "vkCmdSetFrontFace", Dynamic) \
    X(CmdSetDepthTestEnableEXT, "vkCmdSetDepthTestEnableEXT", Dynamic) \
    X(CmdSetDepthTestEnable, "vkCmdSetDepthTestEnable", Dynamic) \
    X(CmdSetDepthWriteEnableEXT, "vkCmdSetDepthWriteEnableEXT", Dynamic) \
    X(CmdSetDepthWriteEnable, "vkCmdSetDepthWriteEnable", Dynamic) \
    X(CmdSetDepthCompareOpEXT, "vkCmdSetDepthCompareOpEXT", Dynamic) \
    X(CmdSetDepthCompareOp, "vkCmdSetDepthCompareOp", Dynamic) \
    X(CmdSetDepthBoundsTestEnableEXT, "vkCmdSetDepthBoundsTestEnableEXT", Dynamic) \
    X(CmdSetDepthBoundsTestEnable, "vkCmdSetDepthBoundsTestEnable", Dynamic) \
    X(CmdSetStencilTestEnableEXT, "vkCmdSetStencilTestEnableEXT", Dynamic) \
    X(CmdSetStencilTestEnable, "vkCmdSetStencilTestEnable", Dynamic) \
    X(CmdSetStencilOpEXT, "vkCmdSetStencilOpEXT", Dynamic) \
    X(CmdSetStencilOp, "vkCmdSetStencilOp", Dynamic) \
    X(CmdSetStencilCompareMask, "vkCmdSetStencilCompareMask", Dynamic) \
    X(CmdSetStencilWriteMask, "vkCmdSetStencilWriteMask", Dynamic) \
    X(CmdSetStencilReference, "vkCmdSetStencilReference", Dynamic) \
    X(CmdSetDepthBias2EXT, "vkCmdSetDepthBias2EXT", Dynamic) \
    X(CmdPipelineBarrier, "vkCmdPipelineBarrier", Other) \
    X(CmdCopyBuffer, "vkCmdCopyBuffer", Other) \
    X(CmdCopyBufferToImage, "vkCmdCopyBufferToImage", Other) \
    X(CmdCopyImageToBuffer, "vkCmdCopyImageToBuffer", Other) \
    X(CmdCopyImage, "vkCmdCopyImage", Other) \
    X(CmdBlitImage, "vkCmdBlitImage", Other) \
    X(CmdFillBuffer, "vkCmdFillBuffer", Other) \
    X(CmdUpdateBuffer, "vkCmdUpdateBuffer", Other) \
    X(CmdClearColorImage, "vkCmdClearColorImage", Other) \
    X(CmdClearDepthStencilImage, "vkCmdClearDepthStencilImage", Other) \
    X(CmdDispatch, "vkCmdDispatch", Other) \
    X(CmdDispatchIndirect, "vkCmdDispatchIndirect", Other) \
    X(CmdPushDescriptorSetKHR, "vkCmdPushDescriptorSetKHR", Other) \
    X(CmdWriteTimestamp, "vkCmdWriteTimestamp", Other) \
    X(CmdResetQueryPool, "vkCmdResetQueryPool", Other) \
    X(CmdBeginQuery, "vkCmdBeginQuery", Other) \
    X(CmdEndQuery, "vkCmdEndQuery", Other) \
    X(CmdCopyQueryPoolResults, "vkCmdCopyQueryPoolResults", Other) \
    X(UpdateDescriptorSets, "vkUpdateDescriptorSets", Sets) \
    X(AllocateDescriptorSets, "vkAllocateDescriptorSets", Sets) \
    X(FreeDescriptorSets, "vkFreeDescriptorSets", Sets) \
    X(ResetDescriptorPool, "vkResetDescriptorPool", Sets) \
    X(CreateDescriptorPool, "vkCreateDescriptorPool", Sets) \
    X(DestroyDescriptorPool, "vkDestroyDescriptorPool", Sets) \
    X(BeginCommandBuffer, "vkBeginCommandBuffer", Buffers) \
    X(EndCommandBuffer, "vkEndCommandBuffer", Buffers) \
    X(ResetCommandBuffer, "vkResetCommandBuffer", Buffers) \
    X(ResetCommandPool, "vkResetCommandPool", Buffers) \
    X(AllocateCommandBuffers, "vkAllocateCommandBuffers", Buffers) \
    X(FreeCommandBuffers, "vkFreeCommandBuffers", Buffers) \
    X(CreateCommandPool, "vkCreateCommandPool", Buffers) \
    X(DestroyCommandPool, "vkDestroyCommandPool", Buffers) \
    X(QueueSubmit, "vkQueueSubmit", Submit) \
    X(CreateBuffer, "vkCreateBuffer", Objects) \
    X(DestroyBuffer, "vkDestroyBuffer", Objects) \
    X(GetBufferMemoryRequirements, "vkGetBufferMemoryRequirements", Objects) \
    X(AllocateMemory, "vkAllocateMemory", Objects) \
    X(FreeMemory, "vkFreeMemory", Objects) \
    X(BindBufferMemory, "vkBindBufferMemory", Objects) \
    X(MapMemory, "vkMapMemory", Objects) \
    X(UnmapMemory, "vkUnmapMemory", Objects) \
    X(FlushMappedMemoryRanges, "vkFlushMappedMemoryRanges", Objects) \
    X(InvalidateMappedMemoryRanges, "vkInvalidateMappedMemoryRanges", Objects) \
    X(CreateImage, "vkCreateImage", Objects) \
    X(DestroyImage, "vkDestroyImage", Objects) \
    X(CreateImageView, "vkCreateImageView", Objects) \
    X(DestroyImageView, "vkDestroyImageView", Objects) \
    X(CreateSampler, "vkCreateSampler", Objects) \
    X(CreateGraphicsPipelines, "vkCreateGraphicsPipelines", Objects) \
    X(CreateComputePipelines, "vkCreateComputePipelines", Objects) \
    X(DestroyPipeline, "vkDestroyPipeline", Objects) \
    X(CreateFramebuffer, "vkCreateFramebuffer", Objects) \
    X(DestroyFramebuffer, "vkDestroyFramebuffer", Objects) \
    X(GetFenceStatus, "vkGetFenceStatus", Poll) \
    X(ResetFences, "vkResetFences", Poll) \
    X(GetSemaphoreCounterValueKHR, "vkGetSemaphoreCounterValueKHR", Poll) \
    X(GetSemaphoreCounterValue, "vkGetSemaphoreCounterValue", Poll) \
    X(GetQueryPoolResults, "vkGetQueryPoolResults", Poll) \
    X(WaitForFences, "vkWaitForFences", Wait) \
    X(WaitSemaphoresKHR, "vkWaitSemaphoresKHR", Wait) \
    X(WaitSemaphores, "vkWaitSemaphores", Wait) \
    X(QueueWaitIdle, "vkQueueWaitIdle", Wait) \
    X(DeviceWaitIdle, "vkDeviceWaitIdle", Wait) \
    X(GetDeviceQueue, "vkGetDeviceQueue", None) \
    X(GetDeviceQueue2, "vkGetDeviceQueue2", None) \
    X(DestroyDevice, "vkDestroyDevice", None)

enum class Slot : std::uint16_t {
#define APS5_FLOOR_ENUM(id, name, group) id,
    APS5_FLOOR_SLOTS(APS5_FLOOR_ENUM)
#undef APS5_FLOOR_ENUM
    Count
};

constexpr std::size_t SlotCount = static_cast<std::size_t>(Slot::Count);

constexpr std::array<const char*, SlotCount> SlotNames{
#define APS5_FLOOR_NAME(id, name, group) name,
    APS5_FLOOR_SLOTS(APS5_FLOOR_NAME)
#undef APS5_FLOOR_NAME
};

constexpr std::array<Group, SlotCount> SlotGroups{
#define APS5_FLOOR_GROUP(id, name, group) Group::group,
    APS5_FLOOR_SLOTS(APS5_FLOOR_GROUP)
#undef APS5_FLOOR_GROUP
};

constexpr std::size_t Index(Slot slot) { return static_cast<std::size_t>(slot); }

enum class Kind : std::uint8_t {
    BindPipeline,
    BindSets,
    Push,
    VertexBuffers,
    IndexBuffer,
    Draw,
    DrawIndexed,
    DrawIndirect,
    DrawIndexedIndirect,
    DrawIndirectCount,
    DrawIndexedIndirectCount,
    DrawMesh,
    DrawMeshIndirect,
    BeginRendering,
    EndRendering,
    BeginRenderPass,
    EndRenderPass,
    Viewport,
    Scissor,
    ViewportWithCount,
    ScissorWithCount,
    ColorWriteMask,
    DepthBounds,
    DepthBias,
    CullMode,
    FrontFace,
    DepthTest,
    DepthWrite,
    DepthCompare,
    DepthBoundsTest,
    StencilTest,
    StencilOp,
    StencilCompareMask,
    StencilWriteMask,
    StencilReference,
    Execute,
    Count
};

constexpr std::size_t KindCount = static_cast<std::size_t>(Kind::Count);

constexpr bool IsDraw(Kind kind) { return kind >= Kind::Draw && kind <= Kind::DrawMeshIndirect; }
constexpr bool IsDynamic(Kind kind) { return kind >= Kind::Viewport && kind <= Kind::StencilReference; }
constexpr bool IsBegin(Kind kind) { return kind == Kind::BeginRendering || kind == Kind::BeginRenderPass; }
constexpr bool IsEnd(Kind kind) { return kind == Kind::EndRendering || kind == Kind::EndRenderPass; }

struct Command {
    Kind kind;
    std::uint32_t u[5];
    std::uint64_t h[4];
    std::uint32_t at[2];
    std::uint32_t bytes[2];
};

struct RenderingRecord {
    VkRect2D area;
    std::uint32_t flags;
    std::uint32_t layers;
    std::uint32_t viewMask;
    std::uint32_t colors;
    std::uint32_t depth;
    std::uint32_t stencil;
};

struct RenderPassRecord {
    std::uint64_t renderPass;
    std::uint64_t framebuffer;
    VkRect2D area;
    std::uint32_t clears;
    std::uint32_t padding;
};

struct Capture {
    std::vector<Command> commands;
    std::vector<std::uint64_t> arena;
    std::uint32_t draws = 0;
    bool secondary = false;

    void Clear() {
        commands.clear();
        arena.clear();
        draws = 0;
    }

    Command& Add(Kind kind) {
        commands.emplace_back();
        auto& command = commands.back();
        command.kind = kind;
        return command;
    }

    void Store(Command& command, int which, const void* data, std::size_t bytes) {
        const auto at = static_cast<std::uint32_t>(arena.size());
        arena.resize(arena.size() + (bytes + 7u) / 8u);
        if (bytes != 0 && data != nullptr) std::memcpy(arena.data() + at, data, bytes);
        command.at[which] = at;
        command.bytes[which] = static_cast<std::uint32_t>(bytes);
    }

    template<typename T>
    const T* At(const Command& command, int which) const {
        return command.bytes[which] == 0 ? nullptr : reinterpret_cast<const T*>(arena.data() + command.at[which]);
    }
};

bool SamePayload(const Capture& capture, const Command& left, const Command& right) {
    if (left.kind != right.kind) return false;
    if (std::memcmp(left.u, right.u, sizeof left.u) != 0 || std::memcmp(left.h, right.h, sizeof left.h) != 0) return false;
    for (int which = 0; which < 2; ++which) {
        if (left.bytes[which] != right.bytes[which]) return false;
        if (left.bytes[which] != 0 && std::memcmp(capture.arena.data() + left.at[which], capture.arena.data() + right.at[which], left.bytes[which]) != 0) return false;
    }
    return true;
}

template<typename T>
std::uint64_t Bits(T handle) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<std::uint64_t>(handle);
    else return static_cast<std::uint64_t>(handle);
}

template<typename T>
T Handle(std::uint64_t bits) {
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(bits);
    else return static_cast<T>(bits);
}

std::uint32_t FloatBits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

float BitsFloat(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

struct Functions {
    PFN_vkCmdBindPipeline bindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets bindSets = nullptr;
    PFN_vkCmdPushConstants push = nullptr;
    PFN_vkCmdBindVertexBuffers vertexBuffers = nullptr;
    PFN_vkCmdBindIndexBuffer indexBuffer = nullptr;
    PFN_vkCmdDraw draw = nullptr;
    PFN_vkCmdDrawIndexed drawIndexed = nullptr;
    PFN_vkCmdDrawIndirect drawIndirect = nullptr;
    PFN_vkCmdDrawIndexedIndirect drawIndexedIndirect = nullptr;
    PFN_vkCmdDrawIndirectCount drawIndirectCount = nullptr;
    PFN_vkCmdDrawIndexedIndirectCount drawIndexedIndirectCount = nullptr;
    PFN_vkCmdDrawMeshTasksEXT drawMesh = nullptr;
    PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshIndirect = nullptr;
    PFN_vkCmdBeginRenderingKHR beginRendering = nullptr;
    PFN_vkCmdEndRenderingKHR endRendering = nullptr;
    PFN_vkCmdBeginRenderPass beginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass endRenderPass = nullptr;
    PFN_vkCmdSetViewport viewport = nullptr;
    PFN_vkCmdSetScissor scissor = nullptr;
    PFN_vkCmdSetViewportWithCount viewportWithCount = nullptr;
    PFN_vkCmdSetScissorWithCount scissorWithCount = nullptr;
    PFN_vkCmdSetColorWriteMaskEXT colorWriteMask = nullptr;
    PFN_vkCmdSetDepthBounds depthBounds = nullptr;
    PFN_vkCmdSetDepthBias depthBias = nullptr;
    PFN_vkCmdSetCullMode cullMode = nullptr;
    PFN_vkCmdSetFrontFace frontFace = nullptr;
    PFN_vkCmdSetDepthTestEnable depthTest = nullptr;
    PFN_vkCmdSetDepthWriteEnable depthWrite = nullptr;
    PFN_vkCmdSetDepthCompareOp depthCompare = nullptr;
    PFN_vkCmdSetDepthBoundsTestEnable depthBoundsTest = nullptr;
    PFN_vkCmdSetStencilTestEnable stencilTest = nullptr;
    PFN_vkCmdSetStencilOp stencilOp = nullptr;
    PFN_vkCmdSetStencilCompareMask stencilCompareMask = nullptr;
    PFN_vkCmdSetStencilWriteMask stencilWriteMask = nullptr;
    PFN_vkCmdSetStencilReference stencilReference = nullptr;
    PFN_vkCmdExecuteCommands execute = nullptr;
    PFN_vkCreateCommandPool createPool = nullptr;
    PFN_vkDestroyCommandPool destroyPool = nullptr;
    PFN_vkResetCommandPool resetPool = nullptr;
    PFN_vkAllocateCommandBuffers allocate = nullptr;
    PFN_vkBeginCommandBuffer begin = nullptr;
    PFN_vkEndCommandBuffer end = nullptr;
};

struct Pool {
    VkCommandPool pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> buffers;
};

struct Scratch {
    Pool primary;
    Pool redone;
    Pool kept;
    Pool shared;
};

struct ViewInfo {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImage image = VK_NULL_HANDLE;
};

struct Device {
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr nextProc = nullptr;
    PFN_vkSetDeviceLoaderData setLoaderData = nullptr;
    std::array<PFN_vkVoidFunction, SlotCount> next{};
    Functions fn;
    std::mutex replayMutex;
    std::unordered_map<std::uint32_t, Scratch> scratch;
    std::mutex objectMutex;
    std::unordered_map<VkQueue, std::uint32_t> queueFamilies;
    std::unordered_map<VkImageView, ViewInfo> views;
    std::unordered_map<VkImage, VkSampleCountFlagBits> images;
    std::atomic<std::uint64_t> submitted{0};
};

struct Instance {
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr nextProc = nullptr;
};

template<typename T>
struct Registry {
    std::array<std::atomic<void*>, 16> keys{};
    std::array<std::atomic<T*>, 16> values{};
    std::mutex mutex;

    void Add(void* key, T* value) {
        std::lock_guard lock(mutex);
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].load(std::memory_order_relaxed) == nullptr) {
                values[i].store(value, std::memory_order_relaxed);
                keys[i].store(key, std::memory_order_release);
                return;
            }
        }
        std::fprintf(stderr, "[draw-floor] too many dispatchable objects\n");
        std::abort();
    }

    void Remove(void* key) {
        std::lock_guard lock(mutex);
        for (auto& entry : keys) {
            if (entry.load(std::memory_order_relaxed) == key) entry.store(nullptr, std::memory_order_release);
        }
    }

    T* Find(void* key) const {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].load(std::memory_order_acquire) == key) return values[i].load(std::memory_order_relaxed);
        }
        return nullptr;
    }
};

Registry<Device>& Devices() {
    static Registry<Device> registry;
    return registry;
}

Registry<Instance>& Instances() {
    static Registry<Instance> registry;
    return registry;
}

inline void* Key(const void* handle) { return *static_cast<void* const*>(handle); }

inline Device& DeviceOf(const void* handle) {
    auto* device = Devices().Find(Key(handle));
    if (device == nullptr) {
        std::fprintf(stderr, "[draw-floor] call on an unknown device\n");
        std::abort();
    }
    return *device;
}

struct SlotTotals {
    std::atomic<std::uint64_t> ticks{0};
    std::atomic<std::uint64_t> calls{0};
};

struct SituTotals {
    std::array<SlotTotals, SlotCount> committer;
    std::array<SlotTotals, SlotCount> others;
};

SituTotals& InSitu() {
    static SituTotals totals;
    return totals;
}

using DescriptionFunction = HRESULT(WINAPI*)(HANDLE, PWSTR*);

DescriptionFunction ThreadDescription() {
    static const auto function = reinterpret_cast<DescriptionFunction>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription")));
    return function;
}

thread_local int threadRole = -1;
thread_local std::uint32_t roleChecks = 0;

bool OnCommitter() {
    if (threadRole >= 0) return threadRole == 1;
    if ((roleChecks++ & 1023u) != 0) return false;
    const auto describe = ThreadDescription();
    if (describe == nullptr) {
        threadRole = 0;
        return false;
    }
    PWSTR name = nullptr;
    if (FAILED(describe(GetCurrentThread(), &name)) || name == nullptr) return false;
    const std::wstring_view named(name);
    if (!named.empty()) threadRole = named.rfind(L"draw-commit", 0) == 0 ? 1 : 0;
    LocalFree(name);
    return threadRole == 1;
}

inline void Account(Slot slot, Tick ticks) {
    auto& totals = OnCommitter() ? InSitu().committer : InSitu().others;
    totals[Index(slot)].ticks.fetch_add(ticks, std::memory_order_relaxed);
    totals[Index(slot)].calls.fetch_add(1, std::memory_order_relaxed);
}

template<typename F>
inline F Next(const Device& device, Slot slot) {
    return reinterpret_cast<F>(device.next[Index(slot)]);
}

template<Slot S, typename F>
struct Timed;

template<Slot S, typename R, typename H, typename... A>
struct Timed<S, R(VKAPI_PTR*)(H, A...)> {
    static R VKAPI_CALL Call(H handle, A... args) {
        const Device& device = DeviceOf(handle);
        const auto next = Next<R(VKAPI_PTR*)(H, A...)>(device, S);
        const Tick start = Now();
        if constexpr (std::is_void_v<R>) {
            next(handle, args...);
            Account(S, Now() - start);
        } else {
            R result = next(handle, args...);
            Account(S, Now() - start);
            return result;
        }
    }
};

std::mutex& CaptureMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<VkCommandBuffer, std::unique_ptr<Capture>>& Captures() {
    static std::unordered_map<VkCommandBuffer, std::unique_ptr<Capture>> captures;
    return captures;
}

thread_local VkCommandBuffer cachedBuffer = VK_NULL_HANDLE;
thread_local Capture* cachedCapture = nullptr;

Capture& CaptureOf(VkCommandBuffer buffer) {
    if (buffer == cachedBuffer) return *cachedCapture;
    std::lock_guard lock(CaptureMutex());
    auto& slot = Captures()[buffer];
    if (!slot) slot = std::make_unique<Capture>();
    cachedBuffer = buffer;
    cachedCapture = slot.get();
    return *slot;
}

template<Slot S, typename F, typename... A>
inline Capture& Forward(VkCommandBuffer buffer, A... args) {
    const Device& device = DeviceOf(buffer);
    const auto next = Next<F>(device, S);
    const Tick start = Now();
    next(buffer, args...);
    Account(S, Now() - start);
    return CaptureOf(buffer);
}

namespace field {
enum : std::size_t {
    Submits,
    BuffersWithDraws,
    Replayed,
    Draws,
    PassesRaw,
    PassesKept,
    PipelinesRaw,
    PipelinesKept,
    SetsRaw,
    SetsKept,
    Pushes,
    VertexRaw,
    VertexKept,
    IndexRaw,
    IndexKept,
    DynamicRaw,
    DynamicKept,
    Executes,
    RawTotal,
    RawMin,
    StrictTotal,
    StrictRecord,
    StrictMin,
    StateTotal,
    StateRecord,
    StateMin,
    RedoneTotal,
    RedoneMin,
    KeptTotal,
    KeptMin,
    SharedTotal,
    SharedMin,
    Count
};
}

constexpr std::size_t FloorFields = field::Count;

std::array<std::atomic<std::uint64_t>, FloorFields>& FloorTotals() {
    static std::array<std::atomic<std::uint64_t>, FloorFields> totals{};
    return totals;
}

inline void AddFloor(std::size_t which, std::uint64_t value) {
    FloorTotals()[which].fetch_add(value, std::memory_order_relaxed);
}

constexpr int Repeats = 3;

std::uint64_t ReplayEvery() {
    static const std::uint64_t every = [] {
        const char* value = std::getenv("APS5_MEASURE_DRAW_FLOOR");
        if (value == nullptr || *value == '\0') return std::uint64_t{1};
        return static_cast<std::uint64_t>(std::strtoull(value, nullptr, 10));
    }();
    return every;
}

struct Pass {
    std::uint32_t begin = 0;
    std::vector<std::uint32_t> prefix;
    std::vector<std::uint32_t> body;
    std::vector<VkFormat> colors;
    VkFormat depth = VK_FORMAT_UNDEFINED;
    VkFormat stencil = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

struct Counts {
    std::uint64_t pipelinesRaw = 0, pipelinesKept = 0, setsRaw = 0, setsKept = 0, pushes = 0, vertexRaw = 0, vertexKept = 0, indexRaw = 0, indexKept = 0, dynamicRaw = 0, dynamicKept = 0, passesRaw = 0, passesKept = 0, executes = 0;
};

constexpr VkShaderStageFlags GraphicsStages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;

void Build(const Capture& capture, bool dynamic, std::vector<std::uint32_t>& out, Counts& counts) {
    std::uint64_t pipeline = 0;
    std::array<std::uint64_t, 32> sets{};
    std::array<std::uint64_t, 32> setLayouts{};
    std::array<bool, 32> setKnown{};
    std::array<std::uint64_t, 32> vertexBuffers{};
    std::array<std::uint64_t, 32> vertexOffsets{};
    std::array<bool, 32> vertexKnown{};
    std::uint64_t indexBuffer = 0, indexOffset = 0;
    std::uint32_t indexType = ~0u;
    std::array<int, KindCount> lastDynamic;
    lastDynamic.fill(-1);
    int openBegin = -1;
    int pendingEnd = -1;
    for (std::uint32_t i = 0; i < capture.commands.size(); ++i) {
        const auto& command = capture.commands[i];
        switch (command.kind) {
        case Kind::BindPipeline: {
            if (command.u[0] != VK_PIPELINE_BIND_POINT_GRAPHICS) break;
            ++counts.pipelinesRaw;
            if (command.h[0] == pipeline) break;
            pipeline = command.h[0];
            ++counts.pipelinesKept;
            out.push_back(i);
            break;
        }
        case Kind::BindSets: {
            if (command.u[0] != VK_PIPELINE_BIND_POINT_GRAPHICS) break;
            ++counts.setsRaw;
            bool changed = command.u[3] != 0;
            const auto* handles = capture.At<std::uint64_t>(command, 0);
            for (std::uint32_t k = 0; k < command.u[2]; ++k) {
                const auto slot = command.u[1] + k;
                if (slot >= sets.size()) {
                    changed = true;
                    continue;
                }
                if (!setKnown[slot] || sets[slot] != handles[k] || setLayouts[slot] != command.h[0]) changed = true;
                setKnown[slot] = true;
                sets[slot] = handles[k];
                setLayouts[slot] = command.h[0];
            }
            if (!changed) break;
            ++counts.setsKept;
            out.push_back(i);
            break;
        }
        case Kind::Push: {
            if ((command.u[0] & GraphicsStages) == 0) break;
            ++counts.pushes;
            out.push_back(i);
            break;
        }
        case Kind::VertexBuffers: {
            ++counts.vertexRaw;
            bool changed = false;
            const auto* buffers = capture.At<std::uint64_t>(command, 0);
            const auto* offsets = capture.At<std::uint64_t>(command, 1);
            for (std::uint32_t k = 0; k < command.u[1]; ++k) {
                const auto slot = command.u[0] + k;
                if (slot >= vertexBuffers.size()) {
                    changed = true;
                    continue;
                }
                if (!vertexKnown[slot] || vertexBuffers[slot] != buffers[k] || vertexOffsets[slot] != offsets[k]) changed = true;
                vertexKnown[slot] = true;
                vertexBuffers[slot] = buffers[k];
                vertexOffsets[slot] = offsets[k];
            }
            if (!changed) break;
            ++counts.vertexKept;
            out.push_back(i);
            break;
        }
        case Kind::IndexBuffer: {
            ++counts.indexRaw;
            if (indexBuffer == command.h[0] && indexOffset == command.h[1] && indexType == command.u[0]) break;
            indexBuffer = command.h[0];
            indexOffset = command.h[1];
            indexType = command.u[0];
            ++counts.indexKept;
            out.push_back(i);
            break;
        }
        case Kind::BeginRendering:
        case Kind::BeginRenderPass: {
            ++counts.passesRaw;
            if (pendingEnd >= 0 && openBegin >= 0 && SamePayload(capture, capture.commands[openBegin], command)) {
                pendingEnd = -1;
                break;
            }
            if (pendingEnd >= 0) out.push_back(static_cast<std::uint32_t>(pendingEnd));
            pendingEnd = -1;
            openBegin = static_cast<int>(i);
            ++counts.passesKept;
            out.push_back(i);
            break;
        }
        case Kind::EndRendering:
        case Kind::EndRenderPass:
            pendingEnd = static_cast<int>(i);
            break;
        case Kind::Execute:
            ++counts.executes;
            break;
        default:
            if (IsDraw(command.kind)) {
                out.push_back(i);
                break;
            }
            if (IsDynamic(command.kind)) {
                ++counts.dynamicRaw;
                if (!dynamic) break;
                const auto kind = static_cast<std::size_t>(command.kind);
                if (lastDynamic[kind] >= 0 && SamePayload(capture, capture.commands[lastDynamic[kind]], command)) break;
                lastDynamic[kind] = static_cast<int>(i);
                ++counts.dynamicKept;
                out.push_back(i);
            }
            break;
        }
    }
    if (pendingEnd >= 0) out.push_back(static_cast<std::uint32_t>(pendingEnd));
}

void BuildRaw(const Capture& capture, std::vector<std::uint32_t>& out) {
    for (std::uint32_t i = 0; i < capture.commands.size(); ++i) {
        if (capture.commands[i].kind != Kind::Execute) out.push_back(i);
    }
}

void BuildPasses(Device& device, const Capture& capture, const std::vector<std::uint32_t>& list, std::vector<Pass>& passes) {
    int pipelineAt = -1, vertexAt = -1, indexAt = -1;
    std::array<int, 32> setAt;
    setAt.fill(-1);
    std::array<int, KindCount> dynamicAt;
    dynamicAt.fill(-1);
    std::vector<std::pair<std::uint64_t, int>> pushAt;
    int open = -1;
    for (const auto index : list) {
        const auto& command = capture.commands[index];
        if (IsBegin(command.kind)) {
            passes.emplace_back();
            open = static_cast<int>(passes.size() - 1);
            auto& pass = passes.back();
            pass.begin = index;
            auto add = [&](int at) {
                if (at >= 0) pass.prefix.push_back(static_cast<std::uint32_t>(at));
            };
            add(pipelineAt);
            for (const auto at : setAt) add(at);
            for (const auto& [key, at] : pushAt) add(at);
            add(vertexAt);
            add(indexAt);
            for (const auto at : dynamicAt) add(at);
            std::sort(pass.prefix.begin(), pass.prefix.end());
            pass.prefix.erase(std::unique(pass.prefix.begin(), pass.prefix.end()), pass.prefix.end());
            if (command.kind == Kind::BeginRendering) {
                const auto* record = capture.At<RenderingRecord>(command, 0);
                const auto* attachments = capture.At<VkRenderingAttachmentInfoKHR>(command, 1);
                std::lock_guard lock(device.objectMutex);
                auto formatOf = [&](VkImageView view) {
                    if (view == VK_NULL_HANDLE) return VK_FORMAT_UNDEFINED;
                    const auto found = device.views.find(view);
                    if (found == device.views.end()) return VK_FORMAT_UNDEFINED;
                    if (pass.samples == VK_SAMPLE_COUNT_1_BIT) {
                        const auto image = device.images.find(found->second.image);
                        if (image != device.images.end()) pass.samples = image->second;
                    }
                    return found->second.format;
                };
                for (std::uint32_t k = 0; k < record->colors; ++k) pass.colors.push_back(formatOf(attachments[k].imageView));
                if (record->depth != 0) pass.depth = formatOf(attachments[record->colors].imageView);
                if (record->stencil != 0) pass.stencil = formatOf(attachments[record->colors + record->depth].imageView);
            }
            continue;
        }
        if (IsEnd(command.kind)) {
            open = -1;
            continue;
        }
        switch (command.kind) {
        case Kind::BindPipeline:
            pipelineAt = static_cast<int>(index);
            break;
        case Kind::BindSets:
            for (std::uint32_t k = 0; k < command.u[2]; ++k) {
                const auto slot = command.u[1] + k;
                if (slot < setAt.size()) setAt[slot] = static_cast<int>(index);
            }
            break;
        case Kind::Push: {
            const std::uint64_t key = (static_cast<std::uint64_t>(command.u[0]) << 40u) | (static_cast<std::uint64_t>(command.u[1]) << 20u) | command.u[2];
            auto found = std::find_if(pushAt.begin(), pushAt.end(), [&](const auto& entry) { return entry.first == key; });
            if (found == pushAt.end()) pushAt.emplace_back(key, static_cast<int>(index));
            else found->second = static_cast<int>(index);
            break;
        }
        case Kind::VertexBuffers:
            vertexAt = static_cast<int>(index);
            break;
        case Kind::IndexBuffer:
            indexAt = static_cast<int>(index);
            break;
        default:
            if (IsDynamic(command.kind)) dynamicAt[static_cast<std::size_t>(command.kind)] = static_cast<int>(index);
            break;
        }
        if (open >= 0) passes[static_cast<std::size_t>(open)].body.push_back(index);
    }
}

inline void IssueBegin(const Device& device, VkCommandBuffer buffer, const Capture& capture, const Command& command, bool secondary) {
    if (command.kind == Kind::BeginRendering) {
        const auto* record = capture.At<RenderingRecord>(command, 0);
        const auto* attachments = capture.At<VkRenderingAttachmentInfoKHR>(command, 1);
        VkRenderingInfoKHR info{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
        info.flags = record->flags | (secondary ? static_cast<VkRenderingFlags>(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR) : 0u);
        info.renderArea = record->area;
        info.layerCount = record->layers;
        info.viewMask = record->viewMask;
        info.colorAttachmentCount = record->colors;
        info.pColorAttachments = record->colors != 0 ? attachments : nullptr;
        info.pDepthAttachment = record->depth != 0 ? attachments + record->colors : nullptr;
        info.pStencilAttachment = record->stencil != 0 ? attachments + record->colors + record->depth : nullptr;
        device.fn.beginRendering(buffer, &info);
        return;
    }
    const auto* record = capture.At<RenderPassRecord>(command, 0);
    VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    info.renderPass = Handle<VkRenderPass>(record->renderPass);
    info.framebuffer = Handle<VkFramebuffer>(record->framebuffer);
    info.renderArea = record->area;
    info.clearValueCount = record->clears;
    info.pClearValues = capture.At<VkClearValue>(command, 1);
    device.fn.beginRenderPass(buffer, &info, secondary ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : static_cast<VkSubpassContents>(command.u[0]));
}

inline void IssueEnd(const Device& device, VkCommandBuffer buffer, Kind begin) {
    if (begin == Kind::BeginRendering || begin == Kind::EndRendering) device.fn.endRendering(buffer);
    else device.fn.endRenderPass(buffer);
}

inline void Issue(const Device& device, VkCommandBuffer buffer, const Capture& capture, const Command& command) {
    const auto& fn = device.fn;
    const auto* u = command.u;
    const auto* h = command.h;
    switch (command.kind) {
    case Kind::BindPipeline:
        fn.bindPipeline(buffer, static_cast<VkPipelineBindPoint>(u[0]), Handle<VkPipeline>(h[0]));
        break;
    case Kind::BindSets:
        fn.bindSets(buffer, static_cast<VkPipelineBindPoint>(u[0]), Handle<VkPipelineLayout>(h[0]), u[1], u[2], capture.At<VkDescriptorSet>(command, 0), u[3], capture.At<std::uint32_t>(command, 1));
        break;
    case Kind::Push:
        fn.push(buffer, Handle<VkPipelineLayout>(h[0]), u[0], u[1], u[2], capture.At<void>(command, 0));
        break;
    case Kind::VertexBuffers:
        fn.vertexBuffers(buffer, u[0], u[1], capture.At<VkBuffer>(command, 0), capture.At<VkDeviceSize>(command, 1));
        break;
    case Kind::IndexBuffer:
        fn.indexBuffer(buffer, Handle<VkBuffer>(h[0]), h[1], static_cast<VkIndexType>(u[0]));
        break;
    case Kind::Draw:
        fn.draw(buffer, u[0], u[1], u[2], u[3]);
        break;
    case Kind::DrawIndexed:
        fn.drawIndexed(buffer, u[0], u[1], u[2], static_cast<std::int32_t>(u[3]), u[4]);
        break;
    case Kind::DrawIndirect:
        fn.drawIndirect(buffer, Handle<VkBuffer>(h[0]), h[1], u[0], u[1]);
        break;
    case Kind::DrawIndexedIndirect:
        fn.drawIndexedIndirect(buffer, Handle<VkBuffer>(h[0]), h[1], u[0], u[1]);
        break;
    case Kind::DrawIndirectCount:
        fn.drawIndirectCount(buffer, Handle<VkBuffer>(h[0]), h[1], Handle<VkBuffer>(h[2]), h[3], u[0], u[1]);
        break;
    case Kind::DrawIndexedIndirectCount:
        fn.drawIndexedIndirectCount(buffer, Handle<VkBuffer>(h[0]), h[1], Handle<VkBuffer>(h[2]), h[3], u[0], u[1]);
        break;
    case Kind::DrawMesh:
        fn.drawMesh(buffer, u[0], u[1], u[2]);
        break;
    case Kind::DrawMeshIndirect:
        fn.drawMeshIndirect(buffer, Handle<VkBuffer>(h[0]), h[1], u[0], u[1]);
        break;
    case Kind::BeginRendering:
    case Kind::BeginRenderPass:
        IssueBegin(device, buffer, capture, command, false);
        break;
    case Kind::EndRendering:
    case Kind::EndRenderPass:
        IssueEnd(device, buffer, command.kind);
        break;
    case Kind::Viewport:
        fn.viewport(buffer, u[0], u[1], capture.At<VkViewport>(command, 0));
        break;
    case Kind::Scissor:
        fn.scissor(buffer, u[0], u[1], capture.At<VkRect2D>(command, 0));
        break;
    case Kind::ViewportWithCount:
        fn.viewportWithCount(buffer, u[0], capture.At<VkViewport>(command, 0));
        break;
    case Kind::ScissorWithCount:
        fn.scissorWithCount(buffer, u[0], capture.At<VkRect2D>(command, 0));
        break;
    case Kind::ColorWriteMask:
        fn.colorWriteMask(buffer, u[0], u[1], capture.At<VkColorComponentFlags>(command, 0));
        break;
    case Kind::DepthBounds:
        fn.depthBounds(buffer, BitsFloat(u[0]), BitsFloat(u[1]));
        break;
    case Kind::DepthBias:
        fn.depthBias(buffer, BitsFloat(u[0]), BitsFloat(u[1]), BitsFloat(u[2]));
        break;
    case Kind::CullMode:
        fn.cullMode(buffer, u[0]);
        break;
    case Kind::FrontFace:
        fn.frontFace(buffer, static_cast<VkFrontFace>(u[0]));
        break;
    case Kind::DepthTest:
        fn.depthTest(buffer, u[0]);
        break;
    case Kind::DepthWrite:
        fn.depthWrite(buffer, u[0]);
        break;
    case Kind::DepthCompare:
        fn.depthCompare(buffer, static_cast<VkCompareOp>(u[0]));
        break;
    case Kind::DepthBoundsTest:
        fn.depthBoundsTest(buffer, u[0]);
        break;
    case Kind::StencilTest:
        fn.stencilTest(buffer, u[0]);
        break;
    case Kind::StencilOp:
        fn.stencilOp(buffer, u[0], static_cast<VkStencilOp>(u[1]), static_cast<VkStencilOp>(u[2]), static_cast<VkStencilOp>(u[3]), static_cast<VkCompareOp>(u[4]));
        break;
    case Kind::StencilCompareMask:
        fn.stencilCompareMask(buffer, u[0], u[1]);
        break;
    case Kind::StencilWriteMask:
        fn.stencilWriteMask(buffer, u[0], u[1]);
        break;
    case Kind::StencilReference:
        fn.stencilReference(buffer, u[0], u[1]);
        break;
    case Kind::Execute:
    case Kind::Count:
        break;
    }
}

void Ensure(Device& device, Pool& pool, std::uint32_t family, VkCommandBufferLevel level, std::size_t count) {
    if (pool.pool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        info.queueFamilyIndex = family;
        if (device.fn.createPool(device.device, &info, nullptr, &pool.pool) != VK_SUCCESS) {
            std::fprintf(stderr, "[draw-floor] vkCreateCommandPool failed\n");
            std::abort();
        }
    }
    if (pool.buffers.size() >= count) return;
    const auto missing = static_cast<std::uint32_t>(count - pool.buffers.size());
    std::vector<VkCommandBuffer> added(missing);
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = pool.pool;
    info.level = level;
    info.commandBufferCount = missing;
    if (device.fn.allocate(device.device, &info, added.data()) != VK_SUCCESS) {
        std::fprintf(stderr, "[draw-floor] vkAllocateCommandBuffers failed\n");
        std::abort();
    }
    for (auto buffer : added) {
        if (device.setLoaderData != nullptr) device.setLoaderData(device.device, buffer);
        pool.buffers.push_back(buffer);
    }
}

struct Timing {
    Tick total = 0;
    Tick record = 0;
};

Timing RecordFlat(const Device& device, Scratch& scratch, const Capture& capture, const std::vector<std::uint32_t>& list) {
    const auto buffer = scratch.primary.buffers[0];
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const Tick start = Now();
    device.fn.resetPool(device.device, scratch.primary.pool, 0);
    device.fn.begin(buffer, &begin);
    const Tick recordStart = Now();
    for (const auto index : list) Issue(device, buffer, capture, capture.commands[index]);
    const Tick recordEnd = Now();
    device.fn.end(buffer);
    const Tick end = Now();
    return {end - start, recordEnd - recordStart};
}

void RecordSecondaries(const Device& device, Pool& pool, VkCommandBufferUsageFlags flags, const Capture& capture, const std::vector<Pass>& passes) {
    device.fn.resetPool(device.device, pool.pool, 0);
    for (std::size_t i = 0; i < passes.size(); ++i) {
        const auto& pass = passes[i];
        const auto& opening = capture.commands[pass.begin];
        VkCommandBufferInheritanceRenderingInfoKHR rendering{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR};
        VkCommandBufferInheritanceInfo inherit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
        if (opening.kind == Kind::BeginRendering) {
            const auto* record = capture.At<RenderingRecord>(opening, 0);
            rendering.flags = record->flags & ~static_cast<VkRenderingFlags>(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR);
            rendering.viewMask = record->viewMask;
            rendering.colorAttachmentCount = static_cast<std::uint32_t>(pass.colors.size());
            rendering.pColorAttachmentFormats = pass.colors.empty() ? nullptr : pass.colors.data();
            rendering.depthAttachmentFormat = pass.depth;
            rendering.stencilAttachmentFormat = pass.stencil;
            rendering.rasterizationSamples = pass.samples;
            inherit.pNext = &rendering;
        } else {
            const auto* record = capture.At<RenderPassRecord>(opening, 0);
            inherit.renderPass = Handle<VkRenderPass>(record->renderPass);
            inherit.subpass = 0;
            inherit.framebuffer = Handle<VkFramebuffer>(record->framebuffer);
        }
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = flags | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        begin.pInheritanceInfo = &inherit;
        const auto buffer = pool.buffers[i];
        device.fn.begin(buffer, &begin);
        for (const auto index : pass.prefix) Issue(device, buffer, capture, capture.commands[index]);
        for (const auto index : pass.body) Issue(device, buffer, capture, capture.commands[index]);
        device.fn.end(buffer);
    }
}

void RecordExecuting(const Device& device, Scratch& scratch, const Pool& secondaries, const Capture& capture, const std::vector<Pass>& passes) {
    const auto buffer = scratch.primary.buffers[0];
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    device.fn.resetPool(device.device, scratch.primary.pool, 0);
    device.fn.begin(buffer, &begin);
    for (std::size_t i = 0; i < passes.size(); ++i) {
        const auto& opening = capture.commands[passes[i].begin];
        IssueBegin(device, buffer, capture, opening, true);
        device.fn.execute(buffer, 1, &secondaries.buffers[i]);
        IssueEnd(device, buffer, opening.kind);
    }
    device.fn.end(buffer);
}

struct Series {
    Tick total = 0;
    Tick record = 0;
    Tick best = ~Tick{0};

    void Add(Timing timing) {
        total += timing.total;
        record += timing.record;
        best = std::min(best, timing.total);
    }
};

void Replay(Device& device, std::uint32_t family, const Capture& capture) {
    Counts counts;
    std::vector<std::uint32_t> raw, strict, state;
    BuildRaw(capture, raw);
    Build(capture, false, strict, counts);
    Counts stateCounts;
    Build(capture, true, state, stateCounts);
    counts.dynamicKept = stateCounts.dynamicKept;
    std::vector<Pass> passes;
    BuildPasses(device, capture, state, passes);
    std::lock_guard lock(device.replayMutex);
    auto& scratch = device.scratch[family];
    Ensure(device, scratch.primary, family, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
    Ensure(device, scratch.redone, family, VK_COMMAND_BUFFER_LEVEL_SECONDARY, std::max<std::size_t>(passes.size(), 1));
    Ensure(device, scratch.kept, family, VK_COMMAND_BUFFER_LEVEL_SECONDARY, std::max<std::size_t>(passes.size(), 1));
    Ensure(device, scratch.shared, family, VK_COMMAND_BUFFER_LEVEL_SECONDARY, std::max<std::size_t>(passes.size(), 1));
    Series rawSeries, strictSeries, stateSeries, redoneSeries, keptSeries, sharedSeries;
    for (int r = 0; r < Repeats; ++r) rawSeries.Add(RecordFlat(device, scratch, capture, raw));
    for (int r = 0; r < Repeats; ++r) strictSeries.Add(RecordFlat(device, scratch, capture, strict));
    for (int r = 0; r < Repeats; ++r) stateSeries.Add(RecordFlat(device, scratch, capture, state));
    for (int r = 0; r < Repeats; ++r) {
        const Tick start = Now();
        RecordSecondaries(device, scratch.redone, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, capture, passes);
        RecordExecuting(device, scratch, scratch.redone, capture, passes);
        redoneSeries.Add({Now() - start, 0});
    }
    RecordSecondaries(device, scratch.kept, 0, capture, passes);
    for (int r = 0; r < Repeats; ++r) {
        const Tick start = Now();
        RecordExecuting(device, scratch, scratch.kept, capture, passes);
        keptSeries.Add({Now() - start, 0});
    }
    RecordSecondaries(device, scratch.shared, VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT, capture, passes);
    for (int r = 0; r < Repeats; ++r) {
        const Tick start = Now();
        RecordExecuting(device, scratch, scratch.shared, capture, passes);
        sharedSeries.Add({Now() - start, 0});
    }
    AddFloor(field::Replayed, 1);
    AddFloor(field::Draws, capture.draws);
    AddFloor(field::PassesRaw, counts.passesRaw);
    AddFloor(field::PassesKept, counts.passesKept);
    AddFloor(field::PipelinesRaw, counts.pipelinesRaw);
    AddFloor(field::PipelinesKept, counts.pipelinesKept);
    AddFloor(field::SetsRaw, counts.setsRaw);
    AddFloor(field::SetsKept, counts.setsKept);
    AddFloor(field::Pushes, counts.pushes);
    AddFloor(field::VertexRaw, counts.vertexRaw);
    AddFloor(field::VertexKept, counts.vertexKept);
    AddFloor(field::IndexRaw, counts.indexRaw);
    AddFloor(field::IndexKept, counts.indexKept);
    AddFloor(field::DynamicRaw, counts.dynamicRaw);
    AddFloor(field::DynamicKept, counts.dynamicKept);
    AddFloor(field::Executes, counts.executes);
    AddFloor(field::RawTotal, rawSeries.total);
    AddFloor(field::RawMin, rawSeries.best);
    AddFloor(field::StrictTotal, strictSeries.total);
    AddFloor(field::StrictRecord, strictSeries.record);
    AddFloor(field::StrictMin, strictSeries.best);
    AddFloor(field::StateTotal, stateSeries.total);
    AddFloor(field::StateRecord, stateSeries.record);
    AddFloor(field::StateMin, stateSeries.best);
    AddFloor(field::RedoneTotal, redoneSeries.total);
    AddFloor(field::RedoneMin, redoneSeries.best);
    AddFloor(field::KeptTotal, keptSeries.total);
    AddFloor(field::KeptMin, keptSeries.best);
    AddFloor(field::SharedTotal, sharedSeries.total);
    AddFloor(field::SharedMin, sharedSeries.best);
}

struct Clock {
    LARGE_INTEGER frequency{};
    LARGE_INTEGER startCounter{};
    Tick startTick = 0;
    Clock() {
        QueryPerformanceFrequency(&frequency);
        QueryPerformanceCounter(&startCounter);
        startTick = Now();
    }
    double Seconds() const {
        LARGE_INTEGER counter;
        QueryPerformanceCounter(&counter);
        return static_cast<double>(counter.QuadPart - startCounter.QuadPart) / static_cast<double>(frequency.QuadPart);
    }
    double TicksPerSecond() const {
        const double seconds = Seconds();
        if (seconds < 0.5) return 4.2e9;
        return static_cast<double>(Now() - startTick) / seconds;
    }
};

Clock& LayerClock() {
    static Clock clock;
    return clock;
}

struct Snapshot {
    double seconds = 0.0;
    std::array<std::uint64_t, FloorFields> floor{};
    std::array<std::uint64_t, SlotCount> committerTicks{}, committerCalls{}, otherTicks{}, otherCalls{};
};

Snapshot Take() {
    Snapshot snapshot;
    snapshot.seconds = LayerClock().Seconds();
    for (std::size_t i = 0; i < FloorFields; ++i) snapshot.floor[i] = FloorTotals()[i].load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < SlotCount; ++i) {
        snapshot.committerTicks[i] = InSitu().committer[i].ticks.load(std::memory_order_relaxed);
        snapshot.committerCalls[i] = InSitu().committer[i].calls.load(std::memory_order_relaxed);
        snapshot.otherTicks[i] = InSitu().others[i].ticks.load(std::memory_order_relaxed);
        snapshot.otherCalls[i] = InSitu().others[i].calls.load(std::memory_order_relaxed);
    }
    return snapshot;
}

std::mutex& ReportMutex() {
    static std::mutex mutex;
    return mutex;
}

Snapshot& Previous() {
    static Snapshot previous;
    return previous;
}

constexpr double ReportSeconds = 5.0;

std::uint64_t DrawCalls(const std::array<std::uint64_t, SlotCount>& calls) {
    std::uint64_t draws = 0;
    for (const auto slot : {Slot::CmdDraw, Slot::CmdDrawIndexed, Slot::CmdDrawIndirect, Slot::CmdDrawIndexedIndirect, Slot::CmdDrawIndirectCountKHR, Slot::CmdDrawIndirectCount, Slot::CmdDrawIndexedIndirectCountKHR, Slot::CmdDrawIndexedIndirectCount, Slot::CmdDrawMeshTasksEXT, Slot::CmdDrawMeshTasksIndirectEXT}) draws += calls[Index(slot)];
    return draws;
}

void Report(bool force) {
    std::unique_lock lock(ReportMutex(), std::try_to_lock);
    if (!lock.owns_lock()) return;
    const double seconds = LayerClock().Seconds();
    auto& previous = Previous();
    if (!force && seconds - previous.seconds < ReportSeconds) return;
    const auto now = Take();
    const double tps = LayerClock().TicksPerSecond();
    std::array<std::uint64_t, FloorFields> d{};
    for (std::size_t i = 0; i < FloorFields; ++i) d[i] = now.floor[i] - previous.floor[i];
    const double draws = static_cast<double>(d[field::Draws]);
    const double replayed = static_cast<double>(d[field::Replayed]);
    const double reps = static_cast<double>(Repeats);
    auto perDraw = [&](std::uint64_t ticks, double divisor) { return draws > 0 ? static_cast<double>(ticks) / divisor / tps * 1e6 / draws : 0.0; };
    auto perBuffer = [&](std::uint64_t ticks) { return replayed > 0 ? static_cast<double>(ticks) / reps / tps * 1e6 / replayed : 0.0; };
    auto ratio = [&](std::size_t which) { return draws > 0 ? static_cast<double>(d[which]) / draws : 0.0; };
    auto count = [&](std::size_t which) { return static_cast<unsigned long long>(d[which]); };
    std::fprintf(stderr,
        "[draw-floor] t=%.1f period=%.1f submits=%llu buffers=%llu replayed=%llu draws=%llu passes=%llu>%llu tps=%.0f | raw %.3f min %.3f | strict %.3f record %.3f min %.3f | state %.3f record %.3f min %.3f | redone %.3f min %.3f | kept %.3f min %.3f | shared %.3f min %.3f us/draw | buffer overhead %.3f us | per draw pipeline %.3f>%.3f sets %.3f>%.3f push %.3f vertex %.3f>%.3f index %.3f>%.3f dynamic %.3f>%.3f executes %llu\n",
        seconds, seconds - previous.seconds, count(field::Submits), count(field::BuffersWithDraws), count(field::Replayed), count(field::Draws), count(field::PassesRaw), count(field::PassesKept), tps,
        perDraw(d[field::RawTotal], reps), perDraw(d[field::RawMin], 1.0),
        perDraw(d[field::StrictTotal], reps), perDraw(d[field::StrictRecord], reps), perDraw(d[field::StrictMin], 1.0),
        perDraw(d[field::StateTotal], reps), perDraw(d[field::StateRecord], reps), perDraw(d[field::StateMin], 1.0),
        perDraw(d[field::RedoneTotal], reps), perDraw(d[field::RedoneMin], 1.0),
        perDraw(d[field::KeptTotal], reps), perDraw(d[field::KeptMin], 1.0),
        perDraw(d[field::SharedTotal], reps), perDraw(d[field::SharedMin], 1.0),
        perBuffer(d[field::StrictTotal] - d[field::StrictRecord]),
        ratio(field::PipelinesRaw), ratio(field::PipelinesKept), ratio(field::SetsRaw), ratio(field::SetsKept), ratio(field::Pushes), ratio(field::VertexRaw), ratio(field::VertexKept), ratio(field::IndexRaw), ratio(field::IndexKept), ratio(field::DynamicRaw), ratio(field::DynamicKept), count(field::Executes));
    for (int role = 0; role < 2; ++role) {
        const auto& ticks = role == 0 ? now.committerTicks : now.otherTicks;
        const auto& calls = role == 0 ? now.committerCalls : now.otherCalls;
        const auto& previousTicks = role == 0 ? previous.committerTicks : previous.otherTicks;
        const auto& previousCalls = role == 0 ? previous.committerCalls : previous.otherCalls;
        std::array<std::uint64_t, SlotCount> dt{}, dc{};
        for (std::size_t i = 0; i < SlotCount; ++i) {
            dt[i] = ticks[i] - previousTicks[i];
            dc[i] = calls[i] - previousCalls[i];
        }
        const auto threadDraws = DrawCalls(dc);
        std::array<std::uint64_t, 10> groups{};
        std::uint64_t allTicks = 0;
        for (std::size_t i = 0; i < SlotCount; ++i) {
            groups[static_cast<std::size_t>(SlotGroups[i])] += dt[i];
            if (SlotGroups[i] != Group::Wait) allTicks += dt[i];
        }
        const double divisor = threadDraws > 0 ? static_cast<double>(threadDraws) : 1.0;
        auto us = [&](std::uint64_t value) { return static_cast<double>(value) / tps * 1e6 / divisor; };
        std::fprintf(stderr, "[draw-floor-insitu] thread=%s draws=%llu | all %.3f draw %.3f dynamic %.3f other %.3f sets %.3f buffers %.3f submit %.3f objects %.3f poll %.3f wait %.3f us/draw | submits %llu %.1f us each\n",
            role == 0 ? "draw-commit" : "others", static_cast<unsigned long long>(threadDraws), us(allTicks),
            us(groups[static_cast<std::size_t>(Group::Draw)]), us(groups[static_cast<std::size_t>(Group::Dynamic)]), us(groups[static_cast<std::size_t>(Group::Other)]), us(groups[static_cast<std::size_t>(Group::Sets)]), us(groups[static_cast<std::size_t>(Group::Buffers)]), us(groups[static_cast<std::size_t>(Group::Submit)]), us(groups[static_cast<std::size_t>(Group::Objects)]), us(groups[static_cast<std::size_t>(Group::Poll)]), us(groups[static_cast<std::size_t>(Group::Wait)]),
            static_cast<unsigned long long>(dc[Index(Slot::QueueSubmit)]), dc[Index(Slot::QueueSubmit)] > 0 ? static_cast<double>(dt[Index(Slot::QueueSubmit)]) / tps * 1e6 / static_cast<double>(dc[Index(Slot::QueueSubmit)]) : 0.0);
        std::vector<std::size_t> order(SlotCount);
        for (std::size_t i = 0; i < SlotCount; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) { return dt[left] > dt[right]; });
        std::string line = std::string("[draw-floor-slots] thread=") + (role == 0 ? "draw-commit" : "others");
        for (std::size_t k = 0; k < 18 && k < order.size(); ++k) {
            const auto i = order[k];
            if (dc[i] == 0) break;
            char item[160];
            std::snprintf(item, sizeof item, " %s=%llu/%.3f", SlotNames[i] + 2, static_cast<unsigned long long>(dc[i]), static_cast<double>(dt[i]) / tps * 1e6 / static_cast<double>(dc[i]));
            line += item;
        }
        line += "\n";
        std::fputs(line.c_str(), stderr);
    }
    std::fflush(stderr);
    previous = now;
}

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info) {
    const Device& device = DeviceOf(buffer);
    const Tick start = Now();
    const auto result = Next<PFN_vkBeginCommandBuffer>(device, Slot::BeginCommandBuffer)(buffer, info);
    Account(Slot::BeginCommandBuffer, Now() - start);
    CaptureOf(buffer).Clear();
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice handle, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* buffers) {
    const Device& device = DeviceOf(handle);
    const Tick start = Now();
    const auto result = Next<PFN_vkAllocateCommandBuffers>(device, Slot::AllocateCommandBuffers)(handle, info, buffers);
    Account(Slot::AllocateCommandBuffers, Now() - start);
    if (result == VK_SUCCESS) {
        for (std::uint32_t i = 0; i < info->commandBufferCount; ++i) {
            auto& capture = CaptureOf(buffers[i]);
            capture.Clear();
            capture.secondary = info->level == VK_COMMAND_BUFFER_LEVEL_SECONDARY;
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice handle, VkCommandPool pool, std::uint32_t count, const VkCommandBuffer* buffers) {
    const Device& device = DeviceOf(handle);
    const Tick start = Now();
    Next<PFN_vkFreeCommandBuffers>(device, Slot::FreeCommandBuffers)(handle, pool, count, buffers);
    Account(Slot::FreeCommandBuffers, Now() - start);
    std::lock_guard lock(CaptureMutex());
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto found = Captures().find(buffers[i]);
        if (found == Captures().end()) continue;
        found->second->commands = {};
        found->second->arena = {};
        found->second->draws = 0;
    }
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue(VkDevice handle, std::uint32_t family, std::uint32_t index, VkQueue* queue) {
    Device& device = DeviceOf(handle);
    Next<PFN_vkGetDeviceQueue>(device, Slot::GetDeviceQueue)(handle, family, index, queue);
    std::lock_guard lock(device.objectMutex);
    device.queueFamilies[*queue] = family;
}

VKAPI_ATTR void VKAPI_CALL GetDeviceQueue2(VkDevice handle, const VkDeviceQueueInfo2* info, VkQueue* queue) {
    Device& device = DeviceOf(handle);
    Next<PFN_vkGetDeviceQueue2>(device, Slot::GetDeviceQueue2)(handle, info, queue);
    std::lock_guard lock(device.objectMutex);
    device.queueFamilies[*queue] = info->queueFamilyIndex;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImage(VkDevice handle, const VkImageCreateInfo* info, const VkAllocationCallbacks* allocator, VkImage* image) {
    Device& device = DeviceOf(handle);
    const Tick start = Now();
    const auto result = Next<PFN_vkCreateImage>(device, Slot::CreateImage)(handle, info, allocator, image);
    Account(Slot::CreateImage, Now() - start);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(device.objectMutex);
        device.images[*image] = info->samples;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImage(VkDevice handle, VkImage image, const VkAllocationCallbacks* allocator) {
    Device& device = DeviceOf(handle);
    {
        std::lock_guard lock(device.objectMutex);
        device.images.erase(image);
    }
    const Tick start = Now();
    Next<PFN_vkDestroyImage>(device, Slot::DestroyImage)(handle, image, allocator);
    Account(Slot::DestroyImage, Now() - start);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateImageView(VkDevice handle, const VkImageViewCreateInfo* info, const VkAllocationCallbacks* allocator, VkImageView* view) {
    Device& device = DeviceOf(handle);
    const Tick start = Now();
    const auto result = Next<PFN_vkCreateImageView>(device, Slot::CreateImageView)(handle, info, allocator, view);
    Account(Slot::CreateImageView, Now() - start);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(device.objectMutex);
        device.views[*view] = ViewInfo{info->format, info->image};
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL DestroyImageView(VkDevice handle, VkImageView view, const VkAllocationCallbacks* allocator) {
    Device& device = DeviceOf(handle);
    {
        std::lock_guard lock(device.objectMutex);
        device.views.erase(view);
    }
    const Tick start = Now();
    Next<PFN_vkDestroyImageView>(device, Slot::DestroyImageView)(handle, view, allocator);
    Account(Slot::DestroyImageView, Now() - start);
}

VKAPI_ATTR VkResult VKAPI_CALL QueueSubmit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    Device& device = DeviceOf(queue);
    AddFloor(field::Submits, 1);
    const auto every = ReplayEvery();
    std::uint32_t family = 0;
    {
        std::lock_guard lock(device.objectMutex);
        const auto found = device.queueFamilies.find(queue);
        if (found != device.queueFamilies.end()) family = found->second;
    }
    for (std::uint32_t s = 0; s < count; ++s) {
        for (std::uint32_t b = 0; b < submits[s].commandBufferCount; ++b) {
            const auto& capture = CaptureOf(submits[s].pCommandBuffers[b]);
            if (capture.draws == 0) continue;
            AddFloor(field::BuffersWithDraws, 1);
            if (every == 0) continue;
            if (device.submitted.fetch_add(1, std::memory_order_relaxed) % every != 0) continue;
            Replay(device, family, capture);
        }
    }
    const Tick start = Now();
    const auto result = Next<PFN_vkQueueSubmit>(device, Slot::QueueSubmit)(queue, count, submits, fence);
    Account(Slot::QueueSubmit, Now() - start);
    Report(false);
    return result;
}

VKAPI_ATTR void VKAPI_CALL CmdBindPipeline(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipeline pipeline) {
    auto& capture = Forward<Slot::CmdBindPipeline, PFN_vkCmdBindPipeline>(buffer, point, pipeline);
    auto& command = capture.Add(Kind::BindPipeline);
    command.u[0] = point;
    command.h[0] = Bits(pipeline);
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets, std::uint32_t dynamicCount, const std::uint32_t* dynamicOffsets) {
    auto& capture = Forward<Slot::CmdBindDescriptorSets, PFN_vkCmdBindDescriptorSets>(buffer, point, layout, first, count, sets, dynamicCount, dynamicOffsets);
    auto& command = capture.Add(Kind::BindSets);
    command.u[0] = point;
    command.u[1] = first;
    command.u[2] = count;
    command.u[3] = dynamicCount;
    command.h[0] = Bits(layout);
    capture.Store(command, 0, sets, sizeof(VkDescriptorSet) * count);
    capture.Store(command, 1, dynamicOffsets, sizeof(std::uint32_t) * dynamicCount);
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer buffer, VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset, std::uint32_t size, const void* values) {
    auto& capture = Forward<Slot::CmdPushConstants, PFN_vkCmdPushConstants>(buffer, layout, stages, offset, size, values);
    auto& command = capture.Add(Kind::Push);
    command.u[0] = stages;
    command.u[1] = offset;
    command.u[2] = size;
    command.h[0] = Bits(layout);
    capture.Store(command, 0, values, size);
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers(VkCommandBuffer buffer, std::uint32_t first, std::uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets) {
    auto& capture = Forward<Slot::CmdBindVertexBuffers, PFN_vkCmdBindVertexBuffers>(buffer, first, count, buffers, offsets);
    auto& command = capture.Add(Kind::VertexBuffers);
    command.u[0] = first;
    command.u[1] = count;
    capture.Store(command, 0, buffers, sizeof(VkBuffer) * count);
    capture.Store(command, 1, offsets, sizeof(VkDeviceSize) * count);
}

VKAPI_ATTR void VKAPI_CALL CmdBindIndexBuffer(VkCommandBuffer buffer, VkBuffer indices, VkDeviceSize offset, VkIndexType type) {
    auto& capture = Forward<Slot::CmdBindIndexBuffer, PFN_vkCmdBindIndexBuffer>(buffer, indices, offset, type);
    auto& command = capture.Add(Kind::IndexBuffer);
    command.h[0] = Bits(indices);
    command.h[1] = offset;
    command.u[0] = type;
}

VKAPI_ATTR void VKAPI_CALL CmdDraw(VkCommandBuffer buffer, std::uint32_t vertices, std::uint32_t instances, std::uint32_t firstVertex, std::uint32_t firstInstance) {
    auto& capture = Forward<Slot::CmdDraw, PFN_vkCmdDraw>(buffer, vertices, instances, firstVertex, firstInstance);
    auto& command = capture.Add(Kind::Draw);
    command.u[0] = vertices;
    command.u[1] = instances;
    command.u[2] = firstVertex;
    command.u[3] = firstInstance;
    ++capture.draws;
}

VKAPI_ATTR void VKAPI_CALL CmdDrawIndexed(VkCommandBuffer buffer, std::uint32_t indices, std::uint32_t instances, std::uint32_t firstIndex, std::int32_t vertexOffset, std::uint32_t firstInstance) {
    auto& capture = Forward<Slot::CmdDrawIndexed, PFN_vkCmdDrawIndexed>(buffer, indices, instances, firstIndex, vertexOffset, firstInstance);
    auto& command = capture.Add(Kind::DrawIndexed);
    command.u[0] = indices;
    command.u[1] = instances;
    command.u[2] = firstIndex;
    command.u[3] = static_cast<std::uint32_t>(vertexOffset);
    command.u[4] = firstInstance;
    ++capture.draws;
}

template<Slot S, Kind K>
VKAPI_ATTR void VKAPI_CALL CmdDrawIndirectLike(VkCommandBuffer buffer, VkBuffer records, VkDeviceSize offset, std::uint32_t count, std::uint32_t stride) {
    auto& capture = Forward<S, PFN_vkCmdDrawIndirect>(buffer, records, offset, count, stride);
    auto& command = capture.Add(K);
    command.h[0] = Bits(records);
    command.h[1] = offset;
    command.u[0] = count;
    command.u[1] = stride;
    ++capture.draws;
}

template<Slot S, Kind K>
VKAPI_ATTR void VKAPI_CALL CmdDrawCountLike(VkCommandBuffer buffer, VkBuffer records, VkDeviceSize offset, VkBuffer counter, VkDeviceSize counterOffset, std::uint32_t maximum, std::uint32_t stride) {
    auto& capture = Forward<S, PFN_vkCmdDrawIndirectCount>(buffer, records, offset, counter, counterOffset, maximum, stride);
    auto& command = capture.Add(K);
    command.h[0] = Bits(records);
    command.h[1] = offset;
    command.h[2] = Bits(counter);
    command.h[3] = counterOffset;
    command.u[0] = maximum;
    command.u[1] = stride;
    ++capture.draws;
}

VKAPI_ATTR void VKAPI_CALL CmdDrawMeshTasks(VkCommandBuffer buffer, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    auto& capture = Forward<Slot::CmdDrawMeshTasksEXT, PFN_vkCmdDrawMeshTasksEXT>(buffer, x, y, z);
    auto& command = capture.Add(Kind::DrawMesh);
    command.u[0] = x;
    command.u[1] = y;
    command.u[2] = z;
    ++capture.draws;
}

VkRenderingAttachmentInfoKHR Normalized(const VkRenderingAttachmentInfoKHR& source) {
    VkRenderingAttachmentInfoKHR copy;
    std::memset(&copy, 0, sizeof copy);
    copy.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    copy.imageView = source.imageView;
    copy.imageLayout = source.imageLayout;
    copy.resolveMode = source.resolveMode;
    copy.resolveImageView = source.resolveImageView;
    copy.resolveImageLayout = source.resolveImageLayout;
    copy.loadOp = source.loadOp;
    copy.storeOp = source.storeOp;
    if (source.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) copy.clearValue = source.clearValue;
    return copy;
}

template<Slot S>
VKAPI_ATTR void VKAPI_CALL CmdBeginRenderingLike(VkCommandBuffer buffer, const VkRenderingInfoKHR* info) {
    auto& capture = Forward<S, PFN_vkCmdBeginRenderingKHR>(buffer, info);
    auto& command = capture.Add(Kind::BeginRendering);
    RenderingRecord record;
    std::memset(&record, 0, sizeof record);
    record.area = info->renderArea;
    record.flags = info->flags;
    record.layers = info->layerCount;
    record.viewMask = info->viewMask;
    record.colors = info->colorAttachmentCount;
    record.depth = info->pDepthAttachment != nullptr ? 1u : 0u;
    record.stencil = info->pStencilAttachment != nullptr ? 1u : 0u;
    std::vector<VkRenderingAttachmentInfoKHR> attachments;
    attachments.reserve(record.colors + 2u);
    for (std::uint32_t k = 0; k < record.colors; ++k) attachments.push_back(Normalized(info->pColorAttachments[k]));
    if (record.depth != 0) attachments.push_back(Normalized(*info->pDepthAttachment));
    if (record.stencil != 0) attachments.push_back(Normalized(*info->pStencilAttachment));
    capture.Store(command, 0, &record, sizeof record);
    capture.Store(command, 1, attachments.data(), attachments.size() * sizeof(VkRenderingAttachmentInfoKHR));
}

template<Slot S>
VKAPI_ATTR void VKAPI_CALL CmdEndRenderingLike(VkCommandBuffer buffer) {
    auto& capture = Forward<S, PFN_vkCmdEndRenderingKHR>(buffer);
    capture.Add(Kind::EndRendering);
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRenderPass(VkCommandBuffer buffer, const VkRenderPassBeginInfo* info, VkSubpassContents contents) {
    auto& capture = Forward<Slot::CmdBeginRenderPass, PFN_vkCmdBeginRenderPass>(buffer, info, contents);
    auto& command = capture.Add(Kind::BeginRenderPass);
    RenderPassRecord record;
    std::memset(&record, 0, sizeof record);
    record.renderPass = Bits(info->renderPass);
    record.framebuffer = Bits(info->framebuffer);
    record.area = info->renderArea;
    record.clears = info->clearValueCount;
    command.u[0] = contents;
    capture.Store(command, 0, &record, sizeof record);
    capture.Store(command, 1, info->pClearValues, sizeof(VkClearValue) * info->clearValueCount);
}

VKAPI_ATTR void VKAPI_CALL CmdEndRenderPass(VkCommandBuffer buffer) {
    auto& capture = Forward<Slot::CmdEndRenderPass, PFN_vkCmdEndRenderPass>(buffer);
    capture.Add(Kind::EndRenderPass);
}

VKAPI_ATTR void VKAPI_CALL CmdExecuteCommands(VkCommandBuffer buffer, std::uint32_t count, const VkCommandBuffer* buffers) {
    auto& capture = Forward<Slot::CmdExecuteCommands, PFN_vkCmdExecuteCommands>(buffer, count, buffers);
    auto& command = capture.Add(Kind::Execute);
    command.u[0] = count;
}

template<Slot S, Kind K, typename T>
VKAPI_ATTR void VKAPI_CALL CmdArrayState(VkCommandBuffer buffer, std::uint32_t first, std::uint32_t count, const T* values) {
    auto& capture = Forward<S, void(VKAPI_PTR*)(VkCommandBuffer, std::uint32_t, std::uint32_t, const T*)>(buffer, first, count, values);
    auto& command = capture.Add(K);
    command.u[0] = first;
    command.u[1] = count;
    capture.Store(command, 0, values, sizeof(T) * count);
}

template<Slot S, Kind K, typename T>
VKAPI_ATTR void VKAPI_CALL CmdCountedState(VkCommandBuffer buffer, std::uint32_t count, const T* values) {
    auto& capture = Forward<S, void(VKAPI_PTR*)(VkCommandBuffer, std::uint32_t, const T*)>(buffer, count, values);
    auto& command = capture.Add(K);
    command.u[0] = count;
    capture.Store(command, 0, values, sizeof(T) * count);
}

template<Slot S, Kind K, typename T>
VKAPI_ATTR void VKAPI_CALL CmdValueState(VkCommandBuffer buffer, T value) {
    auto& capture = Forward<S, void(VKAPI_PTR*)(VkCommandBuffer, T)>(buffer, value);
    auto& command = capture.Add(K);
    command.u[0] = static_cast<std::uint32_t>(value);
}

template<Slot S, Kind K, typename T>
VKAPI_ATTR void VKAPI_CALL CmdFaceState(VkCommandBuffer buffer, VkStencilFaceFlags face, T value) {
    auto& capture = Forward<S, void(VKAPI_PTR*)(VkCommandBuffer, VkStencilFaceFlags, T)>(buffer, face, value);
    auto& command = capture.Add(K);
    command.u[0] = face;
    command.u[1] = static_cast<std::uint32_t>(value);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBounds(VkCommandBuffer buffer, float minimum, float maximum) {
    auto& capture = Forward<Slot::CmdSetDepthBounds, PFN_vkCmdSetDepthBounds>(buffer, minimum, maximum);
    auto& command = capture.Add(Kind::DepthBounds);
    command.u[0] = FloatBits(minimum);
    command.u[1] = FloatBits(maximum);
}

VKAPI_ATTR void VKAPI_CALL CmdSetDepthBias(VkCommandBuffer buffer, float constant, float clamp, float slope) {
    auto& capture = Forward<Slot::CmdSetDepthBias, PFN_vkCmdSetDepthBias>(buffer, constant, clamp, slope);
    auto& command = capture.Add(Kind::DepthBias);
    command.u[0] = FloatBits(constant);
    command.u[1] = FloatBits(clamp);
    command.u[2] = FloatBits(slope);
}

template<Slot S>
VKAPI_ATTR void VKAPI_CALL CmdSetStencilOpLike(VkCommandBuffer buffer, VkStencilFaceFlags face, VkStencilOp fail, VkStencilOp pass, VkStencilOp depthFail, VkCompareOp compare) {
    auto& capture = Forward<S, PFN_vkCmdSetStencilOp>(buffer, face, fail, pass, depthFail, compare);
    auto& command = capture.Add(Kind::StencilOp);
    command.u[0] = face;
    command.u[1] = fail;
    command.u[2] = pass;
    command.u[3] = depthFail;
    command.u[4] = compare;
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info, const VkAllocationCallbacks* allocator, VkDevice* out);
VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice handle, const VkAllocationCallbacks* allocator);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name);

struct Hook {
    Slot slot;
    PFN_vkVoidFunction proc;
};

template<typename F>
PFN_vkVoidFunction Proc(F function) {
    return reinterpret_cast<PFN_vkVoidFunction>(function);
}

template<Slot S, typename F>
PFN_vkVoidFunction TimedProc() {
    return Proc(&Timed<S, F>::Call);
}

const std::vector<Hook>& Hooks() {
    static const std::vector<Hook> hooks{
        {Slot::CmdBindPipeline, Proc(&CmdBindPipeline)},
        {Slot::CmdBindDescriptorSets, Proc(&CmdBindDescriptorSets)},
        {Slot::CmdPushConstants, Proc(&CmdPushConstants)},
        {Slot::CmdBindVertexBuffers, Proc(&CmdBindVertexBuffers)},
        {Slot::CmdBindIndexBuffer, Proc(&CmdBindIndexBuffer)},
        {Slot::CmdDraw, Proc(&CmdDraw)},
        {Slot::CmdDrawIndexed, Proc(&CmdDrawIndexed)},
        {Slot::CmdDrawIndirect, Proc(&CmdDrawIndirectLike<Slot::CmdDrawIndirect, Kind::DrawIndirect>)},
        {Slot::CmdDrawIndexedIndirect, Proc(&CmdDrawIndirectLike<Slot::CmdDrawIndexedIndirect, Kind::DrawIndexedIndirect>)},
        {Slot::CmdDrawIndirectCountKHR, Proc(&CmdDrawCountLike<Slot::CmdDrawIndirectCountKHR, Kind::DrawIndirectCount>)},
        {Slot::CmdDrawIndirectCount, Proc(&CmdDrawCountLike<Slot::CmdDrawIndirectCount, Kind::DrawIndirectCount>)},
        {Slot::CmdDrawIndexedIndirectCountKHR, Proc(&CmdDrawCountLike<Slot::CmdDrawIndexedIndirectCountKHR, Kind::DrawIndexedIndirectCount>)},
        {Slot::CmdDrawIndexedIndirectCount, Proc(&CmdDrawCountLike<Slot::CmdDrawIndexedIndirectCount, Kind::DrawIndexedIndirectCount>)},
        {Slot::CmdDrawMeshTasksEXT, Proc(&CmdDrawMeshTasks)},
        {Slot::CmdDrawMeshTasksIndirectEXT, Proc(&CmdDrawIndirectLike<Slot::CmdDrawMeshTasksIndirectEXT, Kind::DrawMeshIndirect>)},
        {Slot::CmdBeginRenderingKHR, Proc(&CmdBeginRenderingLike<Slot::CmdBeginRenderingKHR>)},
        {Slot::CmdBeginRendering, Proc(&CmdBeginRenderingLike<Slot::CmdBeginRendering>)},
        {Slot::CmdEndRenderingKHR, Proc(&CmdEndRenderingLike<Slot::CmdEndRenderingKHR>)},
        {Slot::CmdEndRendering, Proc(&CmdEndRenderingLike<Slot::CmdEndRendering>)},
        {Slot::CmdBeginRenderPass, Proc(&CmdBeginRenderPass)},
        {Slot::CmdEndRenderPass, Proc(&CmdEndRenderPass)},
        {Slot::CmdExecuteCommands, Proc(&CmdExecuteCommands)},
        {Slot::CmdSetViewport, Proc(&CmdArrayState<Slot::CmdSetViewport, Kind::Viewport, VkViewport>)},
        {Slot::CmdSetScissor, Proc(&CmdArrayState<Slot::CmdSetScissor, Kind::Scissor, VkRect2D>)},
        {Slot::CmdSetViewportWithCountEXT, Proc(&CmdCountedState<Slot::CmdSetViewportWithCountEXT, Kind::ViewportWithCount, VkViewport>)},
        {Slot::CmdSetViewportWithCount, Proc(&CmdCountedState<Slot::CmdSetViewportWithCount, Kind::ViewportWithCount, VkViewport>)},
        {Slot::CmdSetScissorWithCountEXT, Proc(&CmdCountedState<Slot::CmdSetScissorWithCountEXT, Kind::ScissorWithCount, VkRect2D>)},
        {Slot::CmdSetScissorWithCount, Proc(&CmdCountedState<Slot::CmdSetScissorWithCount, Kind::ScissorWithCount, VkRect2D>)},
        {Slot::CmdSetColorWriteMaskEXT, Proc(&CmdArrayState<Slot::CmdSetColorWriteMaskEXT, Kind::ColorWriteMask, VkColorComponentFlags>)},
        {Slot::CmdSetDepthBounds, Proc(&CmdSetDepthBounds)},
        {Slot::CmdSetDepthBias, Proc(&CmdSetDepthBias)},
        {Slot::CmdSetCullModeEXT, Proc(&CmdValueState<Slot::CmdSetCullModeEXT, Kind::CullMode, VkCullModeFlags>)},
        {Slot::CmdSetCullMode, Proc(&CmdValueState<Slot::CmdSetCullMode, Kind::CullMode, VkCullModeFlags>)},
        {Slot::CmdSetFrontFaceEXT, Proc(&CmdValueState<Slot::CmdSetFrontFaceEXT, Kind::FrontFace, VkFrontFace>)},
        {Slot::CmdSetFrontFace, Proc(&CmdValueState<Slot::CmdSetFrontFace, Kind::FrontFace, VkFrontFace>)},
        {Slot::CmdSetDepthTestEnableEXT, Proc(&CmdValueState<Slot::CmdSetDepthTestEnableEXT, Kind::DepthTest, VkBool32>)},
        {Slot::CmdSetDepthTestEnable, Proc(&CmdValueState<Slot::CmdSetDepthTestEnable, Kind::DepthTest, VkBool32>)},
        {Slot::CmdSetDepthWriteEnableEXT, Proc(&CmdValueState<Slot::CmdSetDepthWriteEnableEXT, Kind::DepthWrite, VkBool32>)},
        {Slot::CmdSetDepthWriteEnable, Proc(&CmdValueState<Slot::CmdSetDepthWriteEnable, Kind::DepthWrite, VkBool32>)},
        {Slot::CmdSetDepthCompareOpEXT, Proc(&CmdValueState<Slot::CmdSetDepthCompareOpEXT, Kind::DepthCompare, VkCompareOp>)},
        {Slot::CmdSetDepthCompareOp, Proc(&CmdValueState<Slot::CmdSetDepthCompareOp, Kind::DepthCompare, VkCompareOp>)},
        {Slot::CmdSetDepthBoundsTestEnableEXT, Proc(&CmdValueState<Slot::CmdSetDepthBoundsTestEnableEXT, Kind::DepthBoundsTest, VkBool32>)},
        {Slot::CmdSetDepthBoundsTestEnable, Proc(&CmdValueState<Slot::CmdSetDepthBoundsTestEnable, Kind::DepthBoundsTest, VkBool32>)},
        {Slot::CmdSetStencilTestEnableEXT, Proc(&CmdValueState<Slot::CmdSetStencilTestEnableEXT, Kind::StencilTest, VkBool32>)},
        {Slot::CmdSetStencilTestEnable, Proc(&CmdValueState<Slot::CmdSetStencilTestEnable, Kind::StencilTest, VkBool32>)},
        {Slot::CmdSetStencilOpEXT, Proc(&CmdSetStencilOpLike<Slot::CmdSetStencilOpEXT>)},
        {Slot::CmdSetStencilOp, Proc(&CmdSetStencilOpLike<Slot::CmdSetStencilOp>)},
        {Slot::CmdSetStencilCompareMask, Proc(&CmdFaceState<Slot::CmdSetStencilCompareMask, Kind::StencilCompareMask, std::uint32_t>)},
        {Slot::CmdSetStencilWriteMask, Proc(&CmdFaceState<Slot::CmdSetStencilWriteMask, Kind::StencilWriteMask, std::uint32_t>)},
        {Slot::CmdSetStencilReference, Proc(&CmdFaceState<Slot::CmdSetStencilReference, Kind::StencilReference, std::uint32_t>)},
        {Slot::CmdSetDepthBias2EXT, TimedProc<Slot::CmdSetDepthBias2EXT, PFN_vkCmdSetDepthBias2EXT>()},
        {Slot::CmdPipelineBarrier, TimedProc<Slot::CmdPipelineBarrier, PFN_vkCmdPipelineBarrier>()},
        {Slot::CmdCopyBuffer, TimedProc<Slot::CmdCopyBuffer, PFN_vkCmdCopyBuffer>()},
        {Slot::CmdCopyBufferToImage, TimedProc<Slot::CmdCopyBufferToImage, PFN_vkCmdCopyBufferToImage>()},
        {Slot::CmdCopyImageToBuffer, TimedProc<Slot::CmdCopyImageToBuffer, PFN_vkCmdCopyImageToBuffer>()},
        {Slot::CmdCopyImage, TimedProc<Slot::CmdCopyImage, PFN_vkCmdCopyImage>()},
        {Slot::CmdBlitImage, TimedProc<Slot::CmdBlitImage, PFN_vkCmdBlitImage>()},
        {Slot::CmdFillBuffer, TimedProc<Slot::CmdFillBuffer, PFN_vkCmdFillBuffer>()},
        {Slot::CmdUpdateBuffer, TimedProc<Slot::CmdUpdateBuffer, PFN_vkCmdUpdateBuffer>()},
        {Slot::CmdClearColorImage, TimedProc<Slot::CmdClearColorImage, PFN_vkCmdClearColorImage>()},
        {Slot::CmdClearDepthStencilImage, TimedProc<Slot::CmdClearDepthStencilImage, PFN_vkCmdClearDepthStencilImage>()},
        {Slot::CmdDispatch, TimedProc<Slot::CmdDispatch, PFN_vkCmdDispatch>()},
        {Slot::CmdDispatchIndirect, TimedProc<Slot::CmdDispatchIndirect, PFN_vkCmdDispatchIndirect>()},
        {Slot::CmdPushDescriptorSetKHR, TimedProc<Slot::CmdPushDescriptorSetKHR, PFN_vkCmdPushDescriptorSetKHR>()},
        {Slot::CmdWriteTimestamp, TimedProc<Slot::CmdWriteTimestamp, PFN_vkCmdWriteTimestamp>()},
        {Slot::CmdResetQueryPool, TimedProc<Slot::CmdResetQueryPool, PFN_vkCmdResetQueryPool>()},
        {Slot::CmdBeginQuery, TimedProc<Slot::CmdBeginQuery, PFN_vkCmdBeginQuery>()},
        {Slot::CmdEndQuery, TimedProc<Slot::CmdEndQuery, PFN_vkCmdEndQuery>()},
        {Slot::CmdCopyQueryPoolResults, TimedProc<Slot::CmdCopyQueryPoolResults, PFN_vkCmdCopyQueryPoolResults>()},
        {Slot::UpdateDescriptorSets, TimedProc<Slot::UpdateDescriptorSets, PFN_vkUpdateDescriptorSets>()},
        {Slot::AllocateDescriptorSets, TimedProc<Slot::AllocateDescriptorSets, PFN_vkAllocateDescriptorSets>()},
        {Slot::FreeDescriptorSets, TimedProc<Slot::FreeDescriptorSets, PFN_vkFreeDescriptorSets>()},
        {Slot::ResetDescriptorPool, TimedProc<Slot::ResetDescriptorPool, PFN_vkResetDescriptorPool>()},
        {Slot::CreateDescriptorPool, TimedProc<Slot::CreateDescriptorPool, PFN_vkCreateDescriptorPool>()},
        {Slot::DestroyDescriptorPool, TimedProc<Slot::DestroyDescriptorPool, PFN_vkDestroyDescriptorPool>()},
        {Slot::BeginCommandBuffer, Proc(&BeginCommandBuffer)},
        {Slot::EndCommandBuffer, TimedProc<Slot::EndCommandBuffer, PFN_vkEndCommandBuffer>()},
        {Slot::ResetCommandBuffer, TimedProc<Slot::ResetCommandBuffer, PFN_vkResetCommandBuffer>()},
        {Slot::ResetCommandPool, TimedProc<Slot::ResetCommandPool, PFN_vkResetCommandPool>()},
        {Slot::AllocateCommandBuffers, Proc(&AllocateCommandBuffers)},
        {Slot::FreeCommandBuffers, Proc(&FreeCommandBuffers)},
        {Slot::CreateCommandPool, TimedProc<Slot::CreateCommandPool, PFN_vkCreateCommandPool>()},
        {Slot::DestroyCommandPool, TimedProc<Slot::DestroyCommandPool, PFN_vkDestroyCommandPool>()},
        {Slot::QueueSubmit, Proc(&QueueSubmit)},
        {Slot::CreateBuffer, TimedProc<Slot::CreateBuffer, PFN_vkCreateBuffer>()},
        {Slot::DestroyBuffer, TimedProc<Slot::DestroyBuffer, PFN_vkDestroyBuffer>()},
        {Slot::GetBufferMemoryRequirements, TimedProc<Slot::GetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements>()},
        {Slot::AllocateMemory, TimedProc<Slot::AllocateMemory, PFN_vkAllocateMemory>()},
        {Slot::FreeMemory, TimedProc<Slot::FreeMemory, PFN_vkFreeMemory>()},
        {Slot::BindBufferMemory, TimedProc<Slot::BindBufferMemory, PFN_vkBindBufferMemory>()},
        {Slot::MapMemory, TimedProc<Slot::MapMemory, PFN_vkMapMemory>()},
        {Slot::UnmapMemory, TimedProc<Slot::UnmapMemory, PFN_vkUnmapMemory>()},
        {Slot::FlushMappedMemoryRanges, TimedProc<Slot::FlushMappedMemoryRanges, PFN_vkFlushMappedMemoryRanges>()},
        {Slot::InvalidateMappedMemoryRanges, TimedProc<Slot::InvalidateMappedMemoryRanges, PFN_vkInvalidateMappedMemoryRanges>()},
        {Slot::CreateImage, Proc(&CreateImage)},
        {Slot::DestroyImage, Proc(&DestroyImage)},
        {Slot::CreateImageView, Proc(&CreateImageView)},
        {Slot::DestroyImageView, Proc(&DestroyImageView)},
        {Slot::CreateSampler, TimedProc<Slot::CreateSampler, PFN_vkCreateSampler>()},
        {Slot::CreateGraphicsPipelines, TimedProc<Slot::CreateGraphicsPipelines, PFN_vkCreateGraphicsPipelines>()},
        {Slot::CreateComputePipelines, TimedProc<Slot::CreateComputePipelines, PFN_vkCreateComputePipelines>()},
        {Slot::DestroyPipeline, TimedProc<Slot::DestroyPipeline, PFN_vkDestroyPipeline>()},
        {Slot::CreateFramebuffer, TimedProc<Slot::CreateFramebuffer, PFN_vkCreateFramebuffer>()},
        {Slot::DestroyFramebuffer, TimedProc<Slot::DestroyFramebuffer, PFN_vkDestroyFramebuffer>()},
        {Slot::GetFenceStatus, TimedProc<Slot::GetFenceStatus, PFN_vkGetFenceStatus>()},
        {Slot::ResetFences, TimedProc<Slot::ResetFences, PFN_vkResetFences>()},
        {Slot::GetSemaphoreCounterValueKHR, TimedProc<Slot::GetSemaphoreCounterValueKHR, PFN_vkGetSemaphoreCounterValueKHR>()},
        {Slot::GetSemaphoreCounterValue, TimedProc<Slot::GetSemaphoreCounterValue, PFN_vkGetSemaphoreCounterValue>()},
        {Slot::GetQueryPoolResults, TimedProc<Slot::GetQueryPoolResults, PFN_vkGetQueryPoolResults>()},
        {Slot::WaitForFences, TimedProc<Slot::WaitForFences, PFN_vkWaitForFences>()},
        {Slot::WaitSemaphoresKHR, TimedProc<Slot::WaitSemaphoresKHR, PFN_vkWaitSemaphoresKHR>()},
        {Slot::WaitSemaphores, TimedProc<Slot::WaitSemaphores, PFN_vkWaitSemaphores>()},
        {Slot::QueueWaitIdle, TimedProc<Slot::QueueWaitIdle, PFN_vkQueueWaitIdle>()},
        {Slot::DeviceWaitIdle, TimedProc<Slot::DeviceWaitIdle, PFN_vkDeviceWaitIdle>()},
        {Slot::GetDeviceQueue, Proc(&GetDeviceQueue)},
        {Slot::GetDeviceQueue2, Proc(&GetDeviceQueue2)},
        {Slot::DestroyDevice, Proc(&DestroyDevice)},
    };
    return hooks;
}

const Hook* FindHook(const char* name) {
    for (const auto& hook : Hooks()) {
        if (std::strcmp(SlotNames[Index(hook.slot)], name) == 0) return &hook;
    }
    return nullptr;
}

template<typename F>
F Pick(const Device& device, Slot first, Slot second) {
    const auto chosen = device.next[Index(first)] != nullptr ? device.next[Index(first)] : device.next[Index(second)];
    return reinterpret_cast<F>(chosen);
}

template<typename F>
F Pick(const Device& device, Slot only) {
    return reinterpret_cast<F>(device.next[Index(only)]);
}

void FillFunctions(Device& device) {
    auto& fn = device.fn;
    fn.bindPipeline = Pick<PFN_vkCmdBindPipeline>(device, Slot::CmdBindPipeline);
    fn.bindSets = Pick<PFN_vkCmdBindDescriptorSets>(device, Slot::CmdBindDescriptorSets);
    fn.push = Pick<PFN_vkCmdPushConstants>(device, Slot::CmdPushConstants);
    fn.vertexBuffers = Pick<PFN_vkCmdBindVertexBuffers>(device, Slot::CmdBindVertexBuffers);
    fn.indexBuffer = Pick<PFN_vkCmdBindIndexBuffer>(device, Slot::CmdBindIndexBuffer);
    fn.draw = Pick<PFN_vkCmdDraw>(device, Slot::CmdDraw);
    fn.drawIndexed = Pick<PFN_vkCmdDrawIndexed>(device, Slot::CmdDrawIndexed);
    fn.drawIndirect = Pick<PFN_vkCmdDrawIndirect>(device, Slot::CmdDrawIndirect);
    fn.drawIndexedIndirect = Pick<PFN_vkCmdDrawIndexedIndirect>(device, Slot::CmdDrawIndexedIndirect);
    fn.drawIndirectCount = Pick<PFN_vkCmdDrawIndirectCount>(device, Slot::CmdDrawIndirectCount, Slot::CmdDrawIndirectCountKHR);
    fn.drawIndexedIndirectCount = Pick<PFN_vkCmdDrawIndexedIndirectCount>(device, Slot::CmdDrawIndexedIndirectCount, Slot::CmdDrawIndexedIndirectCountKHR);
    fn.drawMesh = Pick<PFN_vkCmdDrawMeshTasksEXT>(device, Slot::CmdDrawMeshTasksEXT);
    fn.drawMeshIndirect = Pick<PFN_vkCmdDrawMeshTasksIndirectEXT>(device, Slot::CmdDrawMeshTasksIndirectEXT);
    fn.beginRendering = Pick<PFN_vkCmdBeginRenderingKHR>(device, Slot::CmdBeginRenderingKHR, Slot::CmdBeginRendering);
    fn.endRendering = Pick<PFN_vkCmdEndRenderingKHR>(device, Slot::CmdEndRenderingKHR, Slot::CmdEndRendering);
    fn.beginRenderPass = Pick<PFN_vkCmdBeginRenderPass>(device, Slot::CmdBeginRenderPass);
    fn.endRenderPass = Pick<PFN_vkCmdEndRenderPass>(device, Slot::CmdEndRenderPass);
    fn.viewport = Pick<PFN_vkCmdSetViewport>(device, Slot::CmdSetViewport);
    fn.scissor = Pick<PFN_vkCmdSetScissor>(device, Slot::CmdSetScissor);
    fn.viewportWithCount = Pick<PFN_vkCmdSetViewportWithCount>(device, Slot::CmdSetViewportWithCountEXT, Slot::CmdSetViewportWithCount);
    fn.scissorWithCount = Pick<PFN_vkCmdSetScissorWithCount>(device, Slot::CmdSetScissorWithCountEXT, Slot::CmdSetScissorWithCount);
    fn.colorWriteMask = Pick<PFN_vkCmdSetColorWriteMaskEXT>(device, Slot::CmdSetColorWriteMaskEXT);
    fn.depthBounds = Pick<PFN_vkCmdSetDepthBounds>(device, Slot::CmdSetDepthBounds);
    fn.depthBias = Pick<PFN_vkCmdSetDepthBias>(device, Slot::CmdSetDepthBias);
    fn.cullMode = Pick<PFN_vkCmdSetCullMode>(device, Slot::CmdSetCullModeEXT, Slot::CmdSetCullMode);
    fn.frontFace = Pick<PFN_vkCmdSetFrontFace>(device, Slot::CmdSetFrontFaceEXT, Slot::CmdSetFrontFace);
    fn.depthTest = Pick<PFN_vkCmdSetDepthTestEnable>(device, Slot::CmdSetDepthTestEnableEXT, Slot::CmdSetDepthTestEnable);
    fn.depthWrite = Pick<PFN_vkCmdSetDepthWriteEnable>(device, Slot::CmdSetDepthWriteEnableEXT, Slot::CmdSetDepthWriteEnable);
    fn.depthCompare = Pick<PFN_vkCmdSetDepthCompareOp>(device, Slot::CmdSetDepthCompareOpEXT, Slot::CmdSetDepthCompareOp);
    fn.depthBoundsTest = Pick<PFN_vkCmdSetDepthBoundsTestEnable>(device, Slot::CmdSetDepthBoundsTestEnableEXT, Slot::CmdSetDepthBoundsTestEnable);
    fn.stencilTest = Pick<PFN_vkCmdSetStencilTestEnable>(device, Slot::CmdSetStencilTestEnableEXT, Slot::CmdSetStencilTestEnable);
    fn.stencilOp = Pick<PFN_vkCmdSetStencilOp>(device, Slot::CmdSetStencilOpEXT, Slot::CmdSetStencilOp);
    fn.stencilCompareMask = Pick<PFN_vkCmdSetStencilCompareMask>(device, Slot::CmdSetStencilCompareMask);
    fn.stencilWriteMask = Pick<PFN_vkCmdSetStencilWriteMask>(device, Slot::CmdSetStencilWriteMask);
    fn.stencilReference = Pick<PFN_vkCmdSetStencilReference>(device, Slot::CmdSetStencilReference);
    fn.execute = Pick<PFN_vkCmdExecuteCommands>(device, Slot::CmdExecuteCommands);
    fn.createPool = Pick<PFN_vkCreateCommandPool>(device, Slot::CreateCommandPool);
    fn.destroyPool = Pick<PFN_vkDestroyCommandPool>(device, Slot::DestroyCommandPool);
    fn.resetPool = Pick<PFN_vkResetCommandPool>(device, Slot::ResetCommandPool);
    fn.allocate = Pick<PFN_vkAllocateCommandBuffers>(device, Slot::AllocateCommandBuffers);
    fn.begin = Pick<PFN_vkBeginCommandBuffer>(device, Slot::BeginCommandBuffer);
    fn.end = Pick<PFN_vkEndCommandBuffer>(device, Slot::EndCommandBuffer);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateInstance(const VkInstanceCreateInfo* info, const VkAllocationCallbacks* allocator, VkInstance* out) {
    auto* chain = static_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(info->pNext));
    while (chain != nullptr && !(chain->sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO && chain->function == VK_LAYER_LINK_INFO)) chain = static_cast<VkLayerInstanceCreateInfo*>(const_cast<void*>(chain->pNext));
    if (chain == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    const auto nextProc = chain->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    chain->u.pLayerInfo = chain->u.pLayerInfo->pNext;
    const auto create = reinterpret_cast<PFN_vkCreateInstance>(nextProc(VK_NULL_HANDLE, "vkCreateInstance"));
    if (create == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    const auto result = create(info, allocator, out);
    if (result != VK_SUCCESS) return result;
    auto* instance = new Instance;
    instance->instance = *out;
    instance->nextProc = nextProc;
    Instances().Add(Key(*out), instance);
    LayerClock();
    std::fprintf(stderr, "[draw-floor] layer active, replay every %llu buffer(s) with draws, %d repeats\n", static_cast<unsigned long long>(ReplayEvery()), Repeats);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyInstance(VkInstance handle, const VkAllocationCallbacks* allocator) {
    auto* instance = Instances().Find(Key(handle));
    if (instance == nullptr) return;
    const auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(instance->nextProc(handle, "vkDestroyInstance"));
    Instances().Remove(Key(handle));
    if (destroy != nullptr) destroy(handle, allocator);
}

VKAPI_ATTR VkResult VKAPI_CALL CreateDevice(VkPhysicalDevice physical, const VkDeviceCreateInfo* info, const VkAllocationCallbacks* allocator, VkDevice* out) {
    auto* instance = Instances().Find(Key(physical));
    if (instance == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    VkLayerDeviceCreateInfo* link = nullptr;
    VkLayerDeviceCreateInfo* loaderData = nullptr;
    for (auto* chain = static_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(info->pNext)); chain != nullptr; chain = static_cast<VkLayerDeviceCreateInfo*>(const_cast<void*>(chain->pNext))) {
        if (chain->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO) continue;
        if (chain->function == VK_LAYER_LINK_INFO && link == nullptr) link = chain;
        if (chain->function == VK_LOADER_DATA_CALLBACK && loaderData == nullptr) loaderData = chain;
    }
    if (link == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    const auto nextInstanceProc = link->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    const auto nextDeviceProc = link->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    link->u.pLayerInfo = link->u.pLayerInfo->pNext;
    const auto create = reinterpret_cast<PFN_vkCreateDevice>(nextInstanceProc(instance->instance, "vkCreateDevice"));
    if (create == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    const auto result = create(physical, info, allocator, out);
    if (result != VK_SUCCESS) return result;
    auto* device = new Device;
    device->device = *out;
    device->nextProc = nextDeviceProc;
    device->setLoaderData = loaderData != nullptr ? loaderData->u.pfnSetDeviceLoaderData : nullptr;
    for (std::size_t i = 0; i < SlotCount; ++i) device->next[i] = nextDeviceProc(*out, SlotNames[i]);
    FillFunctions(*device);
    Devices().Add(Key(*out), device);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL DestroyDevice(VkDevice handle, const VkAllocationCallbacks* allocator) {
    Device& device = DeviceOf(handle);
    Report(true);
    {
        std::lock_guard lock(device.replayMutex);
        for (auto& [family, scratch] : device.scratch) {
            for (auto* pool : {&scratch.primary, &scratch.redone, &scratch.kept, &scratch.shared}) {
                if (pool->pool != VK_NULL_HANDLE) device.fn.destroyPool(handle, pool->pool, nullptr);
                pool->pool = VK_NULL_HANDLE;
                pool->buffers.clear();
            }
        }
    }
    const auto destroy = Next<PFN_vkDestroyDevice>(device, Slot::DestroyDevice);
    Devices().Remove(Key(handle));
    destroy(handle, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice handle, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return Proc(&GetDeviceProcAddr);
    auto* device = Devices().Find(Key(handle));
    if (device == nullptr) return nullptr;
    if (const auto* hook = FindHook(name); hook != nullptr) {
        if (device->next[Index(hook->slot)] == nullptr) return nullptr;
        return hook->proc;
    }
    return device->nextProc(handle, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetInstanceProcAddr(VkInstance handle, const char* name) {
    if (std::strcmp(name, "vkGetInstanceProcAddr") == 0) return Proc(&GetInstanceProcAddr);
    if (std::strcmp(name, "vkCreateInstance") == 0) return Proc(&CreateInstance);
    if (std::strcmp(name, "vkDestroyInstance") == 0) return Proc(&DestroyInstance);
    if (std::strcmp(name, "vkCreateDevice") == 0) return Proc(&CreateDevice);
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return Proc(&GetDeviceProcAddr);
    if (handle == VK_NULL_HANDLE) return nullptr;
    auto* instance = Instances().Find(Key(handle));
    if (instance == nullptr) return nullptr;
    if (const auto* hook = FindHook(name); hook != nullptr) {
        if (instance->nextProc(handle, name) == nullptr) return nullptr;
        return hook->proc;
    }
    return instance->nextProc(handle, name);
}

}

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* name) {
    return GetInstanceProcAddr(instance, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* name) {
    return GetDeviceProcAddr(device, name);
}

VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* version) {
    if (version == nullptr || version->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT) return VK_ERROR_INITIALIZATION_FAILED;
    if (version->loaderLayerInterfaceVersion > 2) version->loaderLayerInterfaceVersion = 2;
    version->pfnGetInstanceProcAddr = &GetInstanceProcAddr;
    version->pfnGetDeviceProcAddr = &GetDeviceProcAddr;
    version->pfnGetPhysicalDeviceProcAddr = nullptr;
    return VK_SUCCESS;
}

}
