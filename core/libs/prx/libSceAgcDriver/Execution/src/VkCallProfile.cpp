#include "prx/libSceAgcDriver/Execution/include/VkCallProfile.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::VkCallProfile {

namespace {

enum Category : std::uint8_t { Cmd, Submit, Descriptor, Object, Wait, Pipeline, CategoryCount };
constexpr std::array<const char*, CategoryCount> CategoryNames{"cmd", "submit", "descriptor", "object", "wait", "pipeline"};

#define APS5_V3E_VK_CALLS(X) \
    X(vkCmdPipelineBarrier, Cmd) X(vkCmdCopyBuffer, Cmd) X(vkCmdBindPipeline, Cmd) X(vkCmdPushConstants, Cmd) \
    X(vkCmdBindDescriptorSets, Cmd) X(vkCmdCopyImageToBuffer, Cmd) X(vkCmdUpdateBuffer, Cmd) X(vkCmdDispatch, Cmd) \
    X(vkCmdCopyBufferToImage, Cmd) X(vkCmdFillBuffer, Cmd) X(vkCmdClearColorImage, Cmd) X(vkCmdEndRenderPass, Cmd) \
    X(vkCmdCopyImage, Cmd) X(vkCmdDraw, Cmd) X(vkCmdBeginRenderPass, Cmd) X(vkCmdSetViewport, Cmd) X(vkCmdSetScissor, Cmd) \
    X(vkCmdCopyQueryPoolResults, Cmd) X(vkCmdSetStencilReference, Cmd) X(vkCmdResetQueryPool, Cmd) X(vkCmdDrawIndirect, Cmd) \
    X(vkCmdBlitImage, Cmd) X(vkCmdWriteTimestamp, Cmd) X(vkCmdSetDepthBounds, Cmd) X(vkCmdSetDepthBias, Cmd) \
    X(vkCmdDrawIndexed, Cmd) X(vkCmdSetStencilWriteMask, Cmd) X(vkCmdEndQuery, Cmd) X(vkCmdDrawIndexedIndirect, Cmd) \
    X(vkCmdClearDepthStencilImage, Cmd) X(vkCmdBindVertexBuffers, Cmd) X(vkCmdBindIndexBuffer, Cmd) X(vkCmdBeginQuery, Cmd) \
    X(vkCmdSetStencilTestEnableEXT, Cmd) X(vkCmdSetStencilOpEXT, Cmd) X(vkCmdSetStencilCompareMask, Cmd) \
    X(vkCmdSetFrontFaceEXT, Cmd) X(vkCmdSetDepthWriteEnableEXT, Cmd) X(vkCmdSetDepthTestEnableEXT, Cmd) \
    X(vkCmdSetDepthCompareOpEXT, Cmd) X(vkCmdSetDepthBoundsTestEnableEXT, Cmd) X(vkCmdSetCullModeEXT, Cmd) \
    X(vkCmdEndRenderingKHR, Cmd) X(vkCmdDrawMeshTasksEXT, Cmd) X(vkCmdDispatchIndirect, Cmd) X(vkCmdBeginRenderingKHR, Cmd) \
    X(vkCmdWriteBufferMarkerAMD, Cmd) X(vkCmdSetCheckpointNV, Cmd) X(vkCmdDrawMeshTasksIndirectEXT, Cmd) \
    X(vkCmdDrawIndirectCountKHR, Cmd) X(vkCmdDrawIndexedIndirectCountKHR, Cmd) X(vkCmdDispatchBase, Cmd) \
    X(vkQueueSubmit, Submit) X(vkBeginCommandBuffer, Submit) X(vkEndCommandBuffer, Submit) X(vkResetCommandBuffer, Submit) \
    X(vkAllocateCommandBuffers, Submit) X(vkFreeCommandBuffers, Submit) X(vkQueueBindSparse, Submit) X(vkResetFences, Submit) \
    X(vkUpdateDescriptorSets, Descriptor) X(vkAllocateDescriptorSets, Descriptor) X(vkFreeDescriptorSets, Descriptor) \
    X(vkResetDescriptorPool, Descriptor) X(vkCreateDescriptorPool, Descriptor) X(vkDestroyDescriptorPool, Descriptor) \
    X(vkCreateBuffer, Object) X(vkDestroyBuffer, Object) X(vkCreateImage, Object) X(vkDestroyImage, Object) \
    X(vkCreateImageView, Object) X(vkDestroyImageView, Object) X(vkAllocateMemory, Object) X(vkFreeMemory, Object) \
    X(vkBindImageMemory, Object) X(vkBindBufferMemory, Object) X(vkGetImageMemoryRequirements, Object) \
    X(vkGetBufferMemoryRequirements, Object) X(vkMapMemory, Object) X(vkUnmapMemory, Object) X(vkCreateSampler, Object) \
    X(vkDestroySampler, Object) X(vkCreateFramebuffer, Object) X(vkDestroyFramebuffer, Object) X(vkCreateRenderPass, Object) \
    X(vkDestroyRenderPass, Object) X(vkGetBufferDeviceAddressKHR, Object) X(vkGetBufferDeviceAddress, Object) \
    X(vkCreateFence, Object) X(vkDestroyFence, Object) X(vkCreateSemaphore, Object) X(vkDestroySemaphore, Object) \
    X(vkInvalidateMappedMemoryRanges, Object) \
    X(vkWaitForFences, Wait) X(vkGetFenceStatus, Wait) X(vkWaitSemaphoresKHR, Wait) X(vkGetSemaphoreCounterValueKHR, Wait) \
    X(vkGetQueryPoolResults, Wait) X(vkQueueWaitIdle, Wait) X(vkDeviceWaitIdle, Wait) \
    X(vkCreateComputePipelines, Pipeline) X(vkCreateGraphicsPipelines, Pipeline) X(vkCreateShaderModule, Pipeline) \
    X(vkDestroyShaderModule, Pipeline) X(vkDestroyPipeline, Pipeline) X(vkCreatePipelineLayout, Pipeline) \
    X(vkDestroyPipelineLayout, Pipeline) X(vkCreateDescriptorSetLayout, Pipeline) X(vkDestroyDescriptorSetLayout, Pipeline)

enum CallId : std::uint16_t {
#define APS5_V3E_ID(name, category) Id_##name,
    APS5_V3E_VK_CALLS(APS5_V3E_ID)
#undef APS5_V3E_ID
    CallCount
};

struct CallInfo {
    const char* name;
    Category category;
};

constexpr std::array<CallInfo, CallCount> Calls{{
#define APS5_V3E_INFO(name, category) {#name, category},
    APS5_V3E_VK_CALLS(APS5_V3E_INFO)
#undef APS5_V3E_INFO
}};

struct Counter {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> ns{0};
    std::atomic<std::uint64_t> lockedNs{0};
    void Add(std::uint64_t spent, bool locked) {
        calls.store(calls.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        ns.store(ns.load(std::memory_order_relaxed) + spent, std::memory_order_relaxed);
        if (locked) lockedNs.store(lockedNs.load(std::memory_order_relaxed) + spent, std::memory_order_relaxed);
    }
};

struct Seen {
    std::uint64_t calls = 0, ns = 0, lockedNs = 0;
};

struct ThreadStats {
    std::array<Counter, CallCount> counters{};
    std::array<Seen, CallCount> reported{};
    char role[32] = {};
    std::atomic<bool> named{false};
    std::uint32_t sinceCheck = 0;
};

std::mutex& registryMutex() {
    static std::mutex* mutex = new std::mutex();
    return *mutex;
}

std::vector<ThreadStats*>& registry() {
    static auto* threads = new std::vector<ThreadStats*>();
    return *threads;
}

void nameFromOwnTag(ThreadStats& thread);

thread_local ThreadStats* threadStats = nullptr;
thread_local const char* pendingRole = nullptr;

ThreadStats& stats() {
    if (threadStats != nullptr) return *threadStats;
    auto* created = new ThreadStats();
    if (pendingRole != nullptr) {
        std::snprintf(created->role, sizeof(created->role), "%s", pendingRole);
        created->named.store(true, std::memory_order_release);
    }
    {
        std::lock_guard lock(registryMutex());
        registry().push_back(created);
    }
    threadStats = created;
    nameFromOwnTag(*created);
    return *created;
}

void nameFromOwnTag(ThreadStats& thread) {
    if (thread.named.load(std::memory_order_acquire)) return;
    const auto tag = GuestMemory::GpuLockThreadTag();
    char role[sizeof(thread.role)];
    if (tag == 0xffffffffu) std::snprintf(role, sizeof(role), "other");
    else std::snprintf(role, sizeof(role), "q0x%x", tag);
    std::lock_guard lock(registryMutex());
    std::memcpy(thread.role, role, sizeof(role));
}

std::atomic<PFN_vkGetDeviceProcAddr> inner{nullptr};
std::atomic<std::uint64_t> unwrapped{0};
std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
std::mutex reportMutex;

void report(const char* label);

void maybeReport(ThreadStats& thread) {
    if (++thread.sinceCheck < 1024u) return;
    thread.sinceCheck = 0;
    nameFromOwnTag(thread);
    if (ProfileWindows::FrameWindows()) return;
    std::unique_lock lock(reportMutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    if (!ProfileWindows::ClockDue(lastReport)) return;
    lock.unlock();
    report("10s");
}

struct Scope {
    CallId id;
    std::chrono::steady_clock::time_point start;
    explicit Scope(CallId id) : id(id), start(std::chrono::steady_clock::now()) {}
    ~Scope() {
        const auto spent = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        auto& thread = stats();
        thread.counters[id].Add(spent, GuestMemory::GpuMutex().HeldByThisThread());
        maybeReport(thread);
    }
};

template<CallId Id, typename TFunction>
struct Hook;

template<CallId Id, typename TResult, typename... TArgs>
struct Hook<Id, TResult (VKAPI_PTR*)(TArgs...)> {
    using Function = TResult (VKAPI_PTR*)(TArgs...);
    static inline std::atomic<Function> real{nullptr};
    static TResult VKAPI_CALL Call(TArgs... args) {
        const Scope scope(Id);
        return real.load(std::memory_order_relaxed)(args...);
    }
    static PFN_vkVoidFunction Install(PFN_vkVoidFunction resolved) {
        const auto function = reinterpret_cast<Function>(resolved);
        Function expected = nullptr;
        if (!real.compare_exchange_strong(expected, function) && expected != function) {
            unwrapped.fetch_add(1, std::memory_order_relaxed);
            return resolved;
        }
        return reinterpret_cast<PFN_vkVoidFunction>(&Call);
    }
};

using Installer = PFN_vkVoidFunction (*)(PFN_vkVoidFunction);

constexpr std::array<Installer, CallCount> Installers{{
#define APS5_V3E_INSTALL(name, category) &Hook<Id_##name, PFN_##name>::Install,
    APS5_V3E_VK_CALLS(APS5_V3E_INSTALL)
#undef APS5_V3E_INSTALL
}};

PFN_vkVoidFunction VKAPI_CALL resolve(VkDevice device, const char* name) {
    const auto resolver = inner.load(std::memory_order_acquire);
    const auto resolved = resolver(device, name);
    if (resolved == nullptr || name == nullptr) return resolved;
    for (std::size_t i = 0; i < Calls.size(); ++i) {
        if (std::strcmp(Calls[i].name, name) == 0) return Installers[i](resolved);
    }
    return resolved;
}

void report(const char* label) {
    struct Totals {
        std::array<Seen, CallCount> calls{};
    };
    std::map<std::string, Totals> roles;
    {
        std::lock_guard lock(registryMutex());
        for (auto* thread : registry()) {
            auto& totals = roles[thread->role];
            for (std::size_t i = 0; i < CallCount; ++i) {
                const auto calls = thread->counters[i].calls.load(std::memory_order_relaxed);
                const auto ns = thread->counters[i].ns.load(std::memory_order_relaxed);
                const auto lockedNs = thread->counters[i].lockedNs.load(std::memory_order_relaxed);
                auto& seen = thread->reported[i];
                totals.calls[i].calls += calls - seen.calls;
                totals.calls[i].ns += ns - seen.ns;
                totals.calls[i].lockedNs += lockedNs - seen.lockedNs;
                seen = {calls, ns, lockedNs};
            }
        }
    }
    std::string line = "[vkcalls] ";
    line += label;
    line += ":";
    for (const auto& [role, totals] : roles) {
        std::uint64_t calls = 0, ns = 0, lockedNs = 0;
        std::array<std::uint64_t, CategoryCount> categoryNs{}, categoryCalls{};
        for (std::size_t i = 0; i < CallCount; ++i) {
            calls += totals.calls[i].calls;
            ns += totals.calls[i].ns;
            lockedNs += totals.calls[i].lockedNs;
            categoryNs[Calls[i].category] += totals.calls[i].ns;
            categoryCalls[Calls[i].category] += totals.calls[i].calls;
        }
        if (calls == 0) continue;
        char text[256];
        std::snprintf(text, sizeof(text), " | %s %llu calls %.1f ms (%.1f ms under the GPU mutex);", role.c_str(), static_cast<unsigned long long>(calls), static_cast<double>(ns) / 1e6, static_cast<double>(lockedNs) / 1e6);
        line += text;
        for (std::size_t c = 0; c < CategoryCount; ++c) {
            if (categoryCalls[c] == 0) continue;
            std::snprintf(text, sizeof(text), " %s %llu/%.1f", CategoryNames[c], static_cast<unsigned long long>(categoryCalls[c]), static_cast<double>(categoryNs[c]) / 1e6);
            line += text;
        }
        std::array<std::size_t, CallCount> order{};
        for (std::size_t i = 0; i < CallCount; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return totals.calls[a].ns > totals.calls[b].ns; });
        line += "; top";
        for (std::size_t k = 0; k < 8 && totals.calls[order[k]].calls != 0; ++k) {
            const auto& seen = totals.calls[order[k]];
            std::snprintf(text, sizeof(text), " %s %llu x %.2f us", Calls[order[k]].name + 2, static_cast<unsigned long long>(seen.calls), static_cast<double>(seen.ns) / 1e3 / static_cast<double>(seen.calls));
            line += text;
        }
    }
    char tail[64];
    std::snprintf(tail, sizeof(tail), " | unwrapped %llu\n", static_cast<unsigned long long>(unwrapped.load(std::memory_order_relaxed)));
    line += tail;
    AgcDriver::ProfilePrint_nid_no_patch("%s", line.c_str());
}

}

bool Enabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("APS5_V3E_VK_PROFILE");
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

void NameThread(const char* role) {
    if (!Enabled()) return;
    if (threadStats == nullptr) {
        pendingRole = role;
        return;
    }
    std::lock_guard lock(registryMutex());
    std::snprintf(threadStats->role, sizeof(threadStats->role), "%s", role);
    threadStats->named.store(true, std::memory_order_release);
}

void Close(const char* label) {
    if (!Enabled()) return;
    std::lock_guard lock(reportMutex);
    report(label);
}

}
