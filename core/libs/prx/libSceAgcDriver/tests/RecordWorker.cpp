#include "prx/libSceAgcDriver/Execution/include/RecordWorker.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace AgcDriver;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

struct Event {
    std::string text;
    std::thread::id thread;
};

std::mutex eventsMutex;
std::vector<Event> events;
VkResult endResult = VK_SUCCESS;
std::atomic<int> submits{0};

void Log(std::string text) {
    std::lock_guard lock(eventsMutex);
    events.push_back({std::move(text), std::this_thread::get_id()});
}

std::string Hex(const void* value) {
    char text[32];
    std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(value)));
    return text;
}

template<typename THandle>
THandle Handle(std::uintptr_t value) {
    return reinterpret_cast<THandle>(value);
}

template<typename THandle>
std::string H(THandle handle) {
    if constexpr (std::is_pointer_v<THandle>) return Hex(handle);
    else return Hex(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(handle)));
}

VKAPI_ATTR VkResult VKAPI_CALL mockBegin(VkCommandBuffer commands, const VkCommandBufferBeginInfo* info) {
    Log("begin " + H(commands) + " flags " + std::to_string(info->flags));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL mockEnd(VkCommandBuffer commands) {
    Log("end " + H(commands));
    return endResult;
}

VKAPI_ATTR void VKAPI_CALL mockBarrier(VkCommandBuffer commands, VkPipelineStageFlags source, VkPipelineStageFlags destination, VkDependencyFlags, std::uint32_t memoryCount, const VkMemoryBarrier* memory, std::uint32_t bufferCount, const VkBufferMemoryBarrier*, std::uint32_t imageCount, const VkImageMemoryBarrier* images) {
    std::string text = "barrier " + H(commands) + " " + std::to_string(source) + ">" + std::to_string(destination) + " memory " + std::to_string(memoryCount);
    for (std::uint32_t i = 0; i < memoryCount; ++i) text += " " + std::to_string(memory[i].srcAccessMask) + (memory[i].pNext != nullptr ? "+chain" : "");
    text += " buffers " + std::to_string(bufferCount) + " images " + std::to_string(imageCount);
    for (std::uint32_t i = 0; i < imageCount; ++i) text += " " + H(images[i].image) + "/" + std::to_string(images[i].newLayout);
    Log(text);
}

VKAPI_ATTR void VKAPI_CALL mockBindPipeline(VkCommandBuffer commands, VkPipelineBindPoint point, VkPipeline pipeline) {
    Log("pipeline " + H(commands) + " " + std::to_string(point) + " " + H(pipeline));
}

VKAPI_ATTR void VKAPI_CALL mockPush(VkCommandBuffer commands, VkPipelineLayout layout, VkShaderStageFlags stages, std::uint32_t offset, std::uint32_t size, const void* values) {
    std::string text = "push " + H(commands) + " " + H(layout) + " " + std::to_string(stages) + " " + std::to_string(offset) + ":";
    for (std::uint32_t i = 0; i < size / 4; ++i) text += " " + std::to_string(static_cast<const std::uint32_t*>(values)[i]);
    Log(text);
}

VKAPI_ATTR void VKAPI_CALL mockViewport(VkCommandBuffer commands, std::uint32_t first, std::uint32_t count, const VkViewport* viewports) {
    std::string text = "viewport " + H(commands) + " " + std::to_string(first);
    for (std::uint32_t i = 0; i < count; ++i) text += " " + std::to_string(static_cast<int>(viewports[i].width)) + "x" + std::to_string(static_cast<int>(viewports[i].height));
    Log(text);
}

VKAPI_ATTR void VKAPI_CALL mockDraw(VkCommandBuffer commands, std::uint32_t vertices, std::uint32_t instances, std::uint32_t firstVertex, std::uint32_t firstInstance) {
    Log("draw " + H(commands) + " " + std::to_string(vertices) + " " + std::to_string(instances) + " " + std::to_string(firstVertex) + " " + std::to_string(firstInstance));
}

VKAPI_ATTR void VKAPI_CALL mockCopyBuffer(VkCommandBuffer commands, VkBuffer source, VkBuffer destination, std::uint32_t count, const VkBufferCopy* regions) {
    std::string text = "copy " + H(commands) + " " + H(source) + ">" + H(destination);
    for (std::uint32_t i = 0; i < count; ++i) text += " " + std::to_string(regions[i].srcOffset) + "/" + std::to_string(regions[i].dstOffset) + "/" + std::to_string(regions[i].size);
    Log(text);
}

VKAPI_ATTR void VKAPI_CALL mockUpdate(VkCommandBuffer commands, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, const void* data) {
    std::uint64_t sum = 0;
    for (VkDeviceSize i = 0; i < size; ++i) sum += static_cast<const std::uint8_t*>(data)[i];
    Log("update " + H(commands) + " " + H(buffer) + " " + std::to_string(offset) + " " + std::to_string(size) + " sum " + std::to_string(sum));
}

VKAPI_ATTR void VKAPI_CALL mockBindSets(VkCommandBuffer commands, VkPipelineBindPoint point, VkPipelineLayout layout, std::uint32_t first, std::uint32_t count, const VkDescriptorSet* sets, std::uint32_t dynamicCount, const std::uint32_t* offsets) {
    std::string text = "sets " + H(commands) + " " + std::to_string(point) + " " + H(layout) + " " + std::to_string(first);
    for (std::uint32_t i = 0; i < count; ++i) text += " " + H(sets[i]);
    text += " dynamic";
    for (std::uint32_t i = 0; i < dynamicCount; ++i) text += " " + std::to_string(offsets[i]);
    Log(text);
}

VKAPI_ATTR void VKAPI_CALL mockBeginPass(VkCommandBuffer commands, const VkRenderPassBeginInfo* info, VkSubpassContents) {
    std::string text = "pass " + H(commands) + " " + H(info->renderPass) + " " + std::to_string(info->renderArea.extent.width) + " clears";
    for (std::uint32_t i = 0; i < info->clearValueCount; ++i) text += " " + std::to_string(info->pClearValues[i].color.uint32[0]);
    Log(text);
}

VKAPI_ATTR void VKAPI_CALL mockEndPass(VkCommandBuffer commands) {
    Log("endpass " + H(commands));
}

VKAPI_ATTR void VKAPI_CALL mockBeginRendering(VkCommandBuffer commands, const VkRenderingInfoKHR* info) {
    std::string text = "rendering " + H(commands) + " colors";
    for (std::uint32_t i = 0; i < info->colorAttachmentCount; ++i) text += " " + H(info->pColorAttachments[i].imageView);
    text += " depth " + (info->pDepthAttachment != nullptr ? H(info->pDepthAttachment->imageView) : std::string("none"));
    text += info->pDepthAttachment == info->pStencilAttachment ? " same" : " other";
    Log(text);
}

VKAPI_ATTR VkResult VKAPI_CALL mockSubmit(VkQueue queue, std::uint32_t count, const VkSubmitInfo*, VkFence) {
    Log("submit " + H(queue) + " " + std::to_string(count));
    submits.fetch_add(1);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL mockDestroyBuffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    Log("destroy " + H(buffer));
}

VKAPI_ATTR VkResult VKAPI_CALL mockAllocateSets(VkDevice, const VkDescriptorSetAllocateInfo*, VkDescriptorSet* sets) {
    Log("allocate sets");
    sets[0] = Handle<VkDescriptorSet>(0x77);
    return VK_SUCCESS;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL mockProc(VkDevice, const char* name) {
    struct Named {
        const char* name;
        PFN_vkVoidFunction function;
    };
    static const std::array<Named, 17> table{{
        {"vkBeginCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(&mockBegin)},
        {"vkEndCommandBuffer", reinterpret_cast<PFN_vkVoidFunction>(&mockEnd)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(&mockBarrier)},
        {"vkCmdBindPipeline", reinterpret_cast<PFN_vkVoidFunction>(&mockBindPipeline)},
        {"vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction>(&mockPush)},
        {"vkCmdSetViewport", reinterpret_cast<PFN_vkVoidFunction>(&mockViewport)},
        {"vkCmdDraw", reinterpret_cast<PFN_vkVoidFunction>(&mockDraw)},
        {"vkCmdCopyBuffer", reinterpret_cast<PFN_vkVoidFunction>(&mockCopyBuffer)},
        {"vkCmdUpdateBuffer", reinterpret_cast<PFN_vkVoidFunction>(&mockUpdate)},
        {"vkCmdBindDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(&mockBindSets)},
        {"vkCmdBeginRenderPass", reinterpret_cast<PFN_vkVoidFunction>(&mockBeginPass)},
        {"vkCmdEndRenderPass", reinterpret_cast<PFN_vkVoidFunction>(&mockEndPass)},
        {"vkCmdBeginRenderingKHR", reinterpret_cast<PFN_vkVoidFunction>(&mockBeginRendering)},
        {"vkQueueSubmit", reinterpret_cast<PFN_vkVoidFunction>(&mockSubmit)},
        {"vkDestroyBuffer", reinterpret_cast<PFN_vkVoidFunction>(&mockDestroyBuffer)},
        {"vkAllocateDescriptorSets", reinterpret_cast<PFN_vkVoidFunction>(&mockAllocateSets)},
        {"vkCmdUnknownTESTONLY", reinterpret_cast<PFN_vkVoidFunction>(&mockEndPass)},
    }};
    for (const auto& entry : table) {
        if (std::strcmp(entry.name, name) == 0) return entry.function;
    }
    return nullptr;
}

PFN_vkGetDeviceProcAddr proc = nullptr;
const auto device = Handle<VkDevice>(0x10);

template<typename TFunction>
TFunction Get(const char* name) {
    return reinterpret_cast<TFunction>(proc(device, name));
}

std::vector<Event> TakeEvents() {
    std::lock_guard lock(eventsMutex);
    auto taken = std::move(events);
    events.clear();
    return taken;
}

void Begin(VkCommandBuffer commands) {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    Expect(Get<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin) == VK_SUCCESS, "a queued begin did not return VK_SUCCESS");
}

void Submit(VkQueue queue) {
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    Expect(Get<PFN_vkQueueSubmit>("vkQueueSubmit")(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS, "a submit after queued calls failed");
}

void RecordedInOrderBeforeTheSubmit() {
    TakeEvents();
    const auto commands = Handle<VkCommandBuffer>(0x100);
    Begin(commands);
    std::array<VkImageMemoryBarrier, 2> images{};
    for (std::size_t i = 0; i < images.size(); ++i) {
        images[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        images[i].image = Handle<VkImage>(0x200 + i);
        images[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkMemoryBarrier memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    Get<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, 1, 2, 0, 1, &memory, 0, nullptr, 2, images.data());
    images[0].image = Handle<VkImage>(0xdead);
    memory.srcAccessMask = 0;
    Get<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, Handle<VkPipeline>(0x300));
    std::array<std::uint32_t, 3> words{7, 8, 9};
    Get<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, Handle<VkPipelineLayout>(0x400), VK_SHADER_STAGE_ALL, 4, 12, words.data());
    words = {0, 0, 0};
    VkViewport viewport{0, 0, 1920, 1080, 0, 1};
    Get<PFN_vkCmdSetViewport>("vkCmdSetViewport")(commands, 0, 1, &viewport);
    viewport.width = 1;
    std::array<VkDescriptorSet, 2> sets{Handle<VkDescriptorSet>(0x501), Handle<VkDescriptorSet>(0x502)};
    std::array<std::uint32_t, 1> offsets{64};
    Get<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_GRAPHICS, Handle<VkPipelineLayout>(0x400), 1, 2, sets.data(), 1, offsets.data());
    sets = {};
    offsets = {};
    std::array<VkClearValue, 2> clears{};
    clears[0].color.uint32[0] = 11;
    clears[1].color.uint32[0] = 12;
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = Handle<VkRenderPass>(0x600);
    pass.renderArea.extent = {640, 480};
    pass.clearValueCount = 2;
    pass.pClearValues = clears.data();
    Get<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass")(commands, &pass, VK_SUBPASS_CONTENTS_INLINE);
    clears = {};
    Get<PFN_vkCmdDraw>("vkCmdDraw")(commands, 3, 1, 0, 0);
    Get<PFN_vkCmdEndRenderPass>("vkCmdEndRenderPass")(commands);
    std::array<VkRenderingAttachmentInfoKHR, 2> colors{};
    colors[0].sType = colors[1].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    colors[0].imageView = Handle<VkImageView>(0x701);
    colors[1].imageView = Handle<VkImageView>(0x702);
    VkRenderingAttachmentInfoKHR depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR};
    depth.imageView = Handle<VkImageView>(0x7d0);
    VkRenderingInfoKHR rendering{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
    rendering.colorAttachmentCount = 2;
    rendering.pColorAttachments = colors.data();
    rendering.pDepthAttachment = rendering.pStencilAttachment = &depth;
    Get<PFN_vkCmdBeginRenderingKHR>("vkCmdBeginRenderingKHR")(commands, &rendering);
    colors = {};
    depth.imageView = VK_NULL_HANDLE;
    std::vector<std::uint8_t> data(1000);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i);
    Get<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer")(commands, Handle<VkBuffer>(0x800), 16, data.size(), data.data());
    std::fill(data.begin(), data.end(), 0);
    VkBufferCopy region{1, 2, 3};
    Get<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, Handle<VkBuffer>(0x801), Handle<VkBuffer>(0x802), 1, &region);
    region = {};
    Expect(Get<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands) == VK_SUCCESS, "a queued end did not return VK_SUCCESS");
    Submit(Handle<VkQueue>(0x900));
    const auto got = TakeEvents();
    const std::vector<std::string> expected{
        "begin 100 flags 1",
        "barrier 100 1>2 memory 1 64 buffers 0 images 2 200/1 201/1",
        "pipeline 100 0 300",
        "push 100 400 2147483647 4: 7 8 9",
        "viewport 100 0 1920x1080",
        "sets 100 0 400 1 501 502 dynamic 64",
        "pass 100 600 640 clears 11 12",
        "draw 100 3 1 0 0",
        "endpass 100",
        "rendering 100 colors 701 702 depth 7d0 same",
        "update 100 800 16 1000 sum 124716",
        "copy 100 801>802 1/2/3",
        "end 100",
        "submit 900 1",
    };
    Expect(got.size() == expected.size(), "the worker made " + std::to_string(got.size()) + " calls, not " + std::to_string(expected.size()));
    for (std::size_t i = 0; i < std::min(got.size(), expected.size()); ++i) Expect(got[i].text == expected[i], "call " + std::to_string(i) + ": '" + got[i].text + "' instead of '" + expected[i] + "'");
    for (std::size_t i = 0; i + 1 < got.size(); ++i) Expect(got[i].thread != std::this_thread::get_id(), "the queued call '" + got[i].text + "' ran on the caller thread");
    Expect(!got.empty() && got.back().thread == std::this_thread::get_id(), "the submit did not run on the caller thread");
}

void ChainedAndDrainedCallsKeepTheOrder() {
    TakeEvents();
    const auto commands = Handle<VkCommandBuffer>(0x101);
    Begin(commands);
    Get<PFN_vkCmdDraw>("vkCmdDraw")(commands, 1, 1, 0, 0);
    VkMemoryBarrier chained{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    VkMemoryBarrier next{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    chained.pNext = &next;
    chained.srcAccessMask = 5;
    Get<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, 1, 1, 0, 1, &chained, 0, nullptr, 0, nullptr);
    Get<PFN_vkCmdDraw>("vkCmdDraw")(commands, 2, 1, 0, 0);
    Get<PFN_vkDestroyBuffer>("vkDestroyBuffer")(device, Handle<VkBuffer>(0xb0), nullptr);
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    VkDescriptorSet set = VK_NULL_HANDLE;
    Get<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(device, &allocate, &set);
    Expect(set == Handle<VkDescriptorSet>(0x77), "a passed-through call lost its result");
    Get<PFN_vkCmdDraw>("vkCmdDraw")(commands, 3, 1, 0, 0);
    Get<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands);
    RecordWorker::Drain();
    const auto got = TakeEvents();
    const std::vector<std::string> expected{"begin 101 flags 1", "draw 101 1 1 0 0", "barrier 101 1>1 memory 1 5+chain buffers 0 images 0", "draw 101 2 1 0 0", "destroy b0", "allocate sets", "draw 101 3 1 0 0", "end 101"};
    Expect(got.size() == expected.size(), "the drained sequence made " + std::to_string(got.size()) + " calls");
    for (std::size_t i = 0; i < std::min(got.size(), expected.size()); ++i) Expect(got[i].text == expected[i], "drained call " + std::to_string(i) + ": '" + got[i].text + "' instead of '" + expected[i] + "'");
    if (got.size() == expected.size()) {
        Expect(got[2].thread == std::this_thread::get_id(), "a barrier with a structure chain was queued instead of made on the caller");
        Expect(got[4].thread == std::this_thread::get_id(), "a destroy was not made on the caller");
        Expect(got[1].thread != std::this_thread::get_id() && got[6].thread != std::this_thread::get_id(), "a draw ran on the caller");
    }
}

void FailuresSurfaceAtTheSubmit() {
    TakeEvents();
    const auto commands = Handle<VkCommandBuffer>(0x102);
    endResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    Begin(commands);
    Expect(Get<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands) == VK_SUCCESS, "a queued end returned its result early");
    const auto before = submits.load();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    Expect(Get<PFN_vkQueueSubmit>("vkQueueSubmit")(Handle<VkQueue>(0x900), 1, &submit, VK_NULL_HANDLE) == VK_ERROR_OUT_OF_HOST_MEMORY, "a submit after a failed queued end did not return the failure");
    Expect(submits.load() == before, "a submit after a failed queued end reached the queue");
    bool threw = false;
    try {
        RecordWorker::Drain();
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find("vkEndCommandBuffer") != std::string::npos;
    }
    Expect(threw, "Drain did not throw the failed queued call");
    endResult = VK_SUCCESS;
    RecordWorker::ForTestsReset(&mockProc);
}

void ManyThreadsAcrossBlocks() {
    TakeEvents();
    constexpr int threads = 3;
    constexpr int rounds = 3000;
    std::vector<std::thread> producers;
    std::mutex recording;
    for (int t = 0; t < threads; ++t) {
        producers.emplace_back([t, &recording] {
            const auto commands = Handle<VkCommandBuffer>(0x1000 + t);
            std::vector<std::uint8_t> data(4096, static_cast<std::uint8_t>(t + 1));
            for (int i = 0; i < rounds; ++i) {
                Get<PFN_vkCmdDraw>("vkCmdDraw")(commands, static_cast<std::uint32_t>(i), 1, 0, 0);
                if (i % 7 == 0) Get<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer")(commands, Handle<VkBuffer>(0x2000 + t), 0, data.size(), data.data());
                if (i % 500 == 0) {
                    std::lock_guard lock(recording);
                    Submit(Handle<VkQueue>(0x3000 + t));
                }
            }
        });
    }
    for (auto& producer : producers) producer.join();
    RecordWorker::Drain();
    const auto got = TakeEvents();
    std::array<int, threads> next{};
    std::array<int, threads> updates{};
    bool ordered = true;
    for (const auto& event : got) {
        unsigned commands = 0, vertices = 0;
        if (std::sscanf(event.text.c_str(), "draw %x %u", &commands, &vertices) == 2) {
            const auto t = static_cast<int>(commands - 0x1000);
            if (t < 0 || t >= threads || static_cast<int>(vertices) != next[t]) ordered = false;
            else ++next[t];
        } else if (event.text.rfind("update ", 0) == 0) {
            const auto t = static_cast<int>(std::strtoul(event.text.c_str() + 7, nullptr, 16) - 0x1000);
            if (t >= 0 && t < threads) {
                ++updates[t];
                Expect(event.text.find("sum " + std::to_string(4096 * (t + 1))) != std::string::npos, "an update's bytes were not copied whole: " + event.text);
            }
        }
    }
    Expect(ordered, "draws of one command buffer ran out of order across threads");
    for (int t = 0; t < threads; ++t) {
        Expect(next[t] == rounds, "thread " + std::to_string(t) + " made " + std::to_string(next[t]) + " draws");
        Expect(updates[t] == (rounds + 6) / 7, "thread " + std::to_string(t) + " made " + std::to_string(updates[t]) + " updates");
    }
}

void UnknownRecordingCallsAreRefused() {
    bool threw = false;
    try {
        static_cast<void>(proc(device, "vkCmdUnknownTESTONLY"));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    Expect(threw, "a recording call without a rule was passed through");
}

}

int main() {
#ifdef _WIN32
    _putenv_s("APS5_V3E_RECORD_WORKER", "1");
#else
    setenv("APS5_V3E_RECORD_WORKER", "1", 1);
#endif
    RecordWorker::ForTestsReset(&mockProc);
    proc = RecordWorker::Wrap(&mockProc);
    Expect(proc != &mockProc, "the switch on did not wrap the resolver");
    RecordedInOrderBeforeTheSubmit();
    ChainedAndDrainedCallsKeepTheOrder();
    FailuresSurfaceAtTheSubmit();
    ManyThreadsAcrossBlocks();
    UnknownRecordingCallsAreRefused();
    const auto totals = RecordWorker::Totals();
    std::fprintf(stderr, "record worker: %llu queued, %llu direct, %llu drains\n", static_cast<unsigned long long>(totals.queued), static_cast<unsigned long long>(totals.direct), static_cast<unsigned long long>(totals.drains));
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
