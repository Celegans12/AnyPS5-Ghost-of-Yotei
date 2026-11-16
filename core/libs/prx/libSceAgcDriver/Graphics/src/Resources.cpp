#include "prx/libSceAgcDriver/Graphics/include/MemoryAccounting.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DeferredDestruction.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestWorkMark.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libc/include/RunReport.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <vector>

namespace AgcDriver::Graphics {
void WriteCheckpointSentinel(const Context& context, VkCommandBuffer commands);

namespace {

#ifdef __APPLE__
void ClearPooledTail(void* mapping, std::size_t size, VkDeviceSize mappedBytes, VkMemoryPropertyFlags properties) {
    if (mapping == nullptr || (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0 || mappedBytes <= size) return;
    std::memset(static_cast<std::byte*>(mapping) + size, 0, static_cast<std::size_t>(mappedBytes - size));
}

#endif
VkDeviceSize ClassAllocationBytes(const VkMemoryRequirements& requirements, std::size_t capacity) {
    const auto alignment = std::max<VkDeviceSize>(requirements.alignment, 1);
    return std::max<VkDeviceSize>(requirements.size, (static_cast<VkDeviceSize>(capacity) + alignment - 1) / alignment * alignment);
}

VkBuffer BindExactBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize availableBytes) {
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    Check(context.Resolved(&DeviceFunctions::createBuffer, "vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer pooled");
    try {
        VkMemoryRequirements requirements{};
        context.Resolved(&DeviceFunctions::getBufferMemoryRequirements, "vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        Require(requirements.size <= availableBytes, "pooled buffer memory is smaller than the requested buffer");
        Require(requirements.alignment == 0 || offset % requirements.alignment == 0, "pooled buffer offset breaks the requested buffer's alignment");
        Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, offset), "vkBindBufferMemory pooled");
    } catch (...) {
        context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
        throw;
    }
    return buffer;
}

constexpr VkMemoryPropertyFlags HostDefault = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags HostCard = HostDefault | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

int DrawInputsDevice() {
    static const int mode = [] {
        const char* text = std::getenv("APS5_DRAW_INPUTS_DEVICE");
        if (text == nullptr) return 0;
        if (std::strcmp(text, "1") == 0) return 1;
        if (std::strcmp(text, "2") == 0) return 2;
        return 0;
    }();
    return mode;
}

bool OnCard(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    return DrawInputsDevice() != 0 && properties == HostDefault && (usage == VK_BUFFER_USAGE_VERTEX_BUFFER_BIT || usage == VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

VkBufferUsageFlags StagingUsage(VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) {
    return DrawInputsDevice() == 2 && OnCard(usage, properties) ? usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT : usage;
}

struct CardCounters {
    std::atomic<std::uint64_t> placed{0};
    std::atomic<std::uint64_t> placedBytes{0};
    std::atomic<std::uint64_t> refused{0};
    std::atomic<std::uint64_t> uploads{0};
    std::atomic<std::uint64_t> uploadBytes{0};
    std::atomic<std::uint64_t> rewrites{0};
    std::atomic<std::uint64_t> runs{0};
    std::atomic<std::uint64_t> fromCard{0};
    std::atomic<std::uint64_t> fromHost{0};
    std::atomic<std::int64_t> reportAt{0};
};

CardCounters& cardCounters() {
    static auto* counters = new CardCounters();
    return *counters;
}

void tally(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

void reportCard() {
    auto& counters = cardCounters();
    constexpr std::int64_t interval = 10'000'000'000;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto at = counters.reportAt.load(std::memory_order_relaxed);
    if (at == 0) {
        counters.reportAt.compare_exchange_strong(at, now + interval, std::memory_order_relaxed);
        return;
    }
    if (now < at || !counters.reportAt.compare_exchange_strong(at, now + interval, std::memory_order_relaxed)) return;
    const auto take = [](std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed)); };
    const auto placed = take(counters.placed);
    const auto placedBytes = take(counters.placedBytes);
    const auto refused = take(counters.refused);
    const auto uploads = take(counters.uploads);
    const auto uploadBytes = take(counters.uploadBytes);
    const auto rewrites = take(counters.rewrites);
    const auto runs = take(counters.runs);
    const auto fromCard = take(counters.fromCard);
    const auto fromHost = take(counters.fromHost);
    AgcDriver::ProfilePrint_nid_no_patch("[draw-inputs-device] (10 s) mode %d: %llu inputs placed on the card (%.1f MiB), %llu kept in system memory (no room); %llu uploads (%.1f MiB, %llu of them rewrites) in %llu copy runs; binds %llu from the card, %llu from system memory\n", DrawInputsDevice(), placed, static_cast<double>(placedBytes) / 1048576.0, refused, uploads, static_cast<double>(uploadBytes) / 1048576.0, rewrites, runs, fromCard, fromHost);
}

}

struct CardInput {
    std::shared_ptr<Buffer> target;
    VkBuffer source = VK_NULL_HANDLE;
    std::size_t bytes = 0;
    std::atomic<bool> dirty{true};
    bool uploaded = false;
    bool queued = false;
    std::optional<BufferAllocation> staging;
    std::shared_ptr<BufferPool> pool;
    ~CardInput() {
        if (staging && pool) pool->Put(*staging);
    }
};

namespace {

struct CardQueue {
    std::mutex mutex;
    std::vector<std::weak_ptr<CardInput>> owed;
};

CardQueue& cardQueue() {
    static auto* queue = new CardQueue();
    return *queue;
}

void owe(const std::shared_ptr<CardInput>& input) {
    auto& queue = cardQueue();
    std::lock_guard lock(queue.mutex);
    if (input->queued) return;
    input->queued = true;
    queue.owed.push_back(input);
}

void flushCardInputs(const Context& context) {
    auto* recorder = context.recorder;
    std::vector<std::shared_ptr<CardInput>> items;
    {
        auto& queue = cardQueue();
        std::lock_guard lock(queue.mutex);
        if (queue.owed.empty()) return;
        items.reserve(queue.owed.size());
        for (auto& owed : queue.owed) {
            auto item = owed.lock();
            if (item == nullptr) continue;
            item->queued = false;
            if (item->dirty.load(std::memory_order_relaxed)) items.push_back(std::move(item));
        }
        queue.owed.clear();
    }
    if (items.empty()) return;
    const bool rewrite = std::any_of(items.begin(), items.end(), [](const auto& item) { return item->uploaded; });
    const Recorder::CopyBackReach reach{0, true, false, {}};
    const auto commands = recorder->CommandsKeeping(reach);
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::DrawInputCopy);
    if (rewrite) {
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::DrawInputCopy);
    }
    auto& counters = cardCounters();
    std::uint64_t copied = 0;
    for (auto& item : items) {
        item->dirty.store(false, std::memory_order_relaxed);
        CopyBuffer(context, commands, item->source, 0, item->target->Handle(), 0, item->bytes);
        if (item->uploaded) tally(counters.rewrites);
        item->uploaded = true;
        copied += item->bytes;
        recorder->Keep(item);
    }
    RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT);
    Recorder::CountBarriers(Recorder::CommandClass::DrawInputCopy);
    recorder->EndGpuTiming(timing, copied);
    tally(counters.uploads, items.size());
    tally(counters.uploadBytes, copied);
    tally(counters.runs);
    reportCard();
}

VkBuffer cardHandle(const Context& context, const std::shared_ptr<CardInput>& input, VkBuffer staging, const void* mapping) {
    auto& counters = cardCounters();
    if (DrawInputsDevice() == 1) {
        if (input->dirty.exchange(false, std::memory_order_relaxed)) {
            std::memcpy(input->target->Bytes().data(), mapping, input->bytes);
            if (input->uploaded) tally(counters.rewrites);
            input->uploaded = true;
            tally(counters.uploads);
            tally(counters.uploadBytes, input->bytes);
        }
    } else if (input->dirty.load(std::memory_order_relaxed) || !input->uploaded) {
        owe(input);
        auto* recorder = context.recorder;
        if (recorder != nullptr && GuestMemory::GpuMutex().HeldByThisThread() && !recorder->RenderPassOpen()) flushCardInputs(context);
        if (input->dirty.load(std::memory_order_relaxed) || !input->uploaded) {
            if ((counters.fromHost.fetch_add(1, std::memory_order_relaxed) & 4095u) == 0) reportCard();
            return staging;
        }
    }
    if ((counters.fromCard.fetch_add(1, std::memory_order_relaxed) & 4095u) == 0) reportCard();
    return input->target->Handle();
}

}

Buffer::Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties) : Buffer(context, size, usage, properties, false) {
    if (ready && OnCard(usage, properties)) attachCard(usage);
}

void Buffer::attachCard(VkBufferUsageFlags requested) {
    auto& counters = cardCounters();
    try {
        auto input = std::make_shared<CardInput>();
        input->bytes = size;
        input->source = buffer;
        if (DrawInputsDevice() == 1) {
            std::shared_ptr<Buffer> target(new Buffer(context, size, requested, HostCard, true));
            if (target->ready && target->mapping != nullptr) input->target = std::move(target);
        } else {
            input->target = WithinDeviceLocalHeadroom(context, size, requested | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        }
        if (input->target == nullptr) {
            tally(counters.refused);
            return;
        }
        card = std::move(input);
        tally(counters.placed);
        tally(counters.placedBytes, size);
    } catch (const std::exception&) {
        tally(counters.refused);
    }
    reportCard();
}

std::shared_ptr<Buffer> Buffer::WithinDeviceLocalHeadroom(const Context& context, std::size_t size, VkBufferUsageFlags usage) {
    std::shared_ptr<Buffer> buffer(new Buffer(context, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true));
    return buffer->ready ? buffer : nullptr;
}

Buffer::Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, bool headroom) : context(context), size(size), capacity(BufferPool::Capacity(size)), usage(BufferPool::Usage(StagingUsage(usage, properties), properties)), properties(properties) {
    Require(size != 0, "zero-sized GPU buffer");
    const bool addressable = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    Require(!addressable || context.bufferDeviceAddress, "buffer device address is not enabled");
    cache = GetBufferPool(context);
    if (const auto allocation = cache->Take(size, this->usage, properties, "vkAllocateMemory buffer")) {
        buffer = allocation->buffer;
        memory = allocation->memory;
        mapping = allocation->mapping;
        deviceAddress = allocation->address;
        allocationBytes = allocation->allocationBytes;
        capacity = allocation->bytes;
        offset = allocation->offset;
        slab = allocation->slab;
#ifdef __APPLE__
        ClearPooledTail(mapping, size, std::min<VkDeviceSize>(capacity, allocationBytes), properties);
#endif
        if (allocation->bufferBytes != size) {
            context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            deviceAddress = 0;
            try {
                buffer = BindExactBuffer(context, size, this->usage, memory, offset, allocationBytes);
                initializeAddress(usage);
            } catch (...) {
                release();
                throw;
            }
        }
        ready = true;
        return;
    }
    try {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = this->usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Resolved(&DeviceFunctions::createBuffer, "vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        context.Resolved(&DeviceFunctions::getBufferMemoryRequirements, "vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        const VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        if (addressable) allocation.pNext = &flags;
        allocation.allocationSize = ClassAllocationBytes(requirements, capacity);
        allocationBytes = allocation.allocationSize;
        // The CPU reads most of these buffers back (write-back, diffs), which is very slow from
        // write-combined memory, so the default host properties prefer cached host memory.
        const bool useSlab = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && BufferPool::SlabEligible(capacity, requirements.alignment, requirements.size, context.limits.nonCoherentAtomSize);
        allocation.memoryTypeIndex = context.BufferMemoryType(requirements.memoryTypeBits, properties, useSlab ? BufferPool::SlabBlockBytes(capacity) : allocation.allocationSize);
        if (headroom && !context.DeviceLocalHeadroom(allocation.memoryTypeIndex, useSlab ? BufferPool::SlabBlockBytes(capacity) : allocation.allocationSize)) {
            context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer")(context.device, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
            return;
        }
        if (useSlab) {
            const auto slot = cache->TakeSlot(context, allocation.memoryTypeIndex, capacity, addressable);
            memory = slot.memory;
            offset = slot.offset;
            slab = true;
            allocationBytes = capacity;
            Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, offset), "vkBindBufferMemory slab");
            initializeAddress(usage);
            mapping = slot.mapping;
#ifdef __APPLE__
            ClearPooledTail(mapping, size, capacity, properties);
#endif
            ready = true;
            return;
        }
        memory = cache->AllocateMemory(allocation, "vkAllocateMemory buffer");
        Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory");
        initializeAddress(usage);
        if ((properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) Check(context.Resolved(&DeviceFunctions::mapMemory, "vkMapMemory")(context.device, memory, 0, VK_WHOLE_SIZE, 0, &mapping), "vkMapMemory");
        ready = true;
    } catch (...) {
        release();
        throw;
    }
}

Buffer::~Buffer() {
    release();
}

void Buffer::release() noexcept {
    if (ready && cache) {
        const BufferAllocation allocation{buffer, memory, mapping, deviceAddress, allocationBytes, capacity, usage, properties, offset, slab, size};
        if (card != nullptr) {
            card->pool = cache;
            card->staging = allocation;
            return;
        }
        cache->Put(allocation);
        return;
    }
    if (slab) {
        if (buffer) context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
        cache->PutSlot(memory, offset);
        return;
    }
    if (mapping) context.Resolved(&DeviceFunctions::unmapMemory, "vkUnmapMemory")(context.device, memory);
    if (buffer) context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
    if (memory) MemoryAccounting::Free(context, memory);
}

VkBuffer Buffer::Handle() const {
    if (card == nullptr) return buffer;
    return cardHandle(context, card, buffer, mapping);
}

std::span<std::byte> Buffer::Bytes() {
    Require(mapping != nullptr, "device-local buffer has no host mapping");
    if (card != nullptr) card->dirty.store(true, std::memory_order_relaxed);
    return {static_cast<std::byte*>(mapping), size};
}

void Buffer::Invalidate() {
    Require(mapping != nullptr, "cannot invalidate an unmapped GPU buffer");
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = memory;
    range.offset = offset;
    range.size = slab ? static_cast<VkDeviceSize>(capacity) : VK_WHOLE_SIZE;
    Check(context.Function<PFN_vkInvalidateMappedMemoryRanges>("vkInvalidateMappedMemoryRanges")(context.device, 1, &range), "vkInvalidateMappedMemoryRanges");
}

DeviceBuffer::DeviceBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage) : context(context), size(size), capacity(BufferPool::Capacity(size)), usage(BufferPool::Usage(usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    Require(size != 0, "zero-sized device buffer");
    cache = GetBufferPool(context);
    auto pooled = cache->Take(size, this->usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "vkAllocateMemory device buffer");
    if (!pooled) {
        pooled = cache->Take(size, this->usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "vkAllocateMemory device buffer", true);
        if (pooled && context.DeviceLocalFits(~0u, capacity)) {
            cache->Put(*pooled);
            pooled.reset();
        }
    }
    if (pooled) {
        hostPlaced = pooled->hostPlaced;
        buffer = pooled->buffer;
        memory = pooled->memory;
        allocationBytes = pooled->allocationBytes;
        capacity = pooled->bytes;
        if (pooled->bufferBytes == size) {
            ready = true;
            return;
        }
        context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        try {
            buffer = BindExactBuffer(context, size, this->usage, memory, 0, allocationBytes);
        } catch (...) {
            release();
            throw;
        }
        ready = true;
        return;
    }
    try {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = this->usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Check(context.Resolved(&DeviceFunctions::createBuffer, "vkCreateBuffer")(context.device, &info, nullptr, &buffer), "vkCreateBuffer device");
        VkMemoryRequirements requirements{};
        context.Resolved(&DeviceFunctions::getBufferMemoryRequirements, "vkGetBufferMemoryRequirements")(context.device, buffer, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = ClassAllocationBytes(requirements, capacity);
        allocationBytes = allocation.allocationSize;
        allocation.memoryTypeIndex = context.DeviceMemoryType(requirements.memoryTypeBits, allocation.allocationSize, "device buffer");
        const auto deviceType = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::uint32_t allocatedType = allocation.memoryTypeIndex;
        memory = cache->AllocateMemory(allocation, "vkAllocateMemory device buffer", context.RefusedDeviceMemoryFallback(requirements.memoryTypeBits, allocation.memoryTypeIndex), &allocatedType);
        hostPlaced = allocatedType != deviceType;
        Check(context.Resolved(&DeviceFunctions::bindBufferMemory, "vkBindBufferMemory")(context.device, buffer, memory, 0), "vkBindBufferMemory device");
        ready = true;
    } catch (...) {
        release();
        throw;
    }
}

DeviceBuffer::~DeviceBuffer() {
    release();
}

void DeviceBuffer::release() noexcept {
    if (ready && buffer && memory && cache) {
        BufferAllocation allocation{buffer, memory, nullptr, 0, allocationBytes, capacity, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, size};
        allocation.hostPlaced = hostPlaced;
        cache->Put(allocation);
        return;
    }
    if (buffer) context.Resolved(&DeviceFunctions::destroyBuffer, "vkDestroyBuffer")(context.device, buffer, nullptr);
    if (memory) MemoryAccounting::Free(context, memory);
}

VkBuffer DeviceBuffer::Handle() const {
    return buffer;
}

std::size_t DeviceBuffer::Size() const {
    return size;
}

void CopyBuffer(const Context& context, VkCommandBuffer commands, VkBuffer source, VkDeviceSize sourceOffset, VkBuffer destination, VkDeviceSize destinationOffset, VkDeviceSize bytes) {
    PerformanceTrace::Amount("Graphics.Copy", 1);
    PerformanceTrace::Amount("Graphics.Copy.bytes", bytes);
    const VkBufferCopy region{sourceOffset, destinationOffset, bytes};
    context.Resolved(&DeviceFunctions::cmdCopyBuffer, "vkCmdCopyBuffer")(commands, source, destination, 1, &region);
}

void RecordMemoryBarrier(const Context& context, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess) {
    const VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, sourceAccess, destinationAccess};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, sourceStage, destinationStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

void FillDeviceFunctions(const Context& context, DeviceFunctions& functions) {
    functions.cmdPipelineBarrier = context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
    functions.cmdCopyBuffer = context.Function<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer");
    functions.cmdUpdateBuffer = context.Function<PFN_vkCmdUpdateBuffer>("vkCmdUpdateBuffer");
    functions.cmdFillBuffer = context.Function<PFN_vkCmdFillBuffer>("vkCmdFillBuffer");
    functions.cmdBindPipeline = context.lookup<PFN_vkCmdBindPipeline>("vkCmdBindPipeline");
    functions.cmdBindDescriptorSets = context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets");
    functions.cmdPushConstants = context.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants");
    functions.cmdDispatch = context.Function<PFN_vkCmdDispatch>("vkCmdDispatch");
    functions.cmdDispatchIndirect = context.Function<PFN_vkCmdDispatchIndirect>("vkCmdDispatchIndirect");
    functions.cmdBeginRenderPass = context.Function<PFN_vkCmdBeginRenderPass>("vkCmdBeginRenderPass");
    functions.cmdEndRenderPass = context.Function<PFN_vkCmdEndRenderPass>("vkCmdEndRenderPass");
    functions.cmdSetViewport = context.lookup<PFN_vkCmdSetViewport>("vkCmdSetViewport");
    functions.cmdSetScissor = context.lookup<PFN_vkCmdSetScissor>("vkCmdSetScissor");
    if (context.dynamicColorWriteMask) functions.cmdSetColorWriteMask = context.lookup<PFN_vkCmdSetColorWriteMaskEXT>("vkCmdSetColorWriteMaskEXT");
    functions.cmdSetDepthBounds = context.lookup<PFN_vkCmdSetDepthBounds>("vkCmdSetDepthBounds");
    functions.cmdSetDepthBias = context.lookup<PFN_vkCmdSetDepthBias>("vkCmdSetDepthBias");
    if (context.graphicsPipelineLibrary) {
        functions.cmdBeginRendering = context.Function<PFN_vkCmdBeginRenderingKHR>("vkCmdBeginRenderingKHR");
        functions.cmdEndRendering = context.Function<PFN_vkCmdEndRenderingKHR>("vkCmdEndRenderingKHR");
        functions.cmdSetCullMode = context.lookup<PFN_vkCmdSetCullModeEXT>("vkCmdSetCullModeEXT");
        functions.cmdSetFrontFace = context.lookup<PFN_vkCmdSetFrontFaceEXT>("vkCmdSetFrontFaceEXT");
        functions.cmdSetDepthTestEnable = context.lookup<PFN_vkCmdSetDepthTestEnableEXT>("vkCmdSetDepthTestEnableEXT");
        functions.cmdSetDepthWriteEnable = context.lookup<PFN_vkCmdSetDepthWriteEnableEXT>("vkCmdSetDepthWriteEnableEXT");
        functions.cmdSetDepthCompareOp = context.lookup<PFN_vkCmdSetDepthCompareOpEXT>("vkCmdSetDepthCompareOpEXT");
        functions.cmdSetDepthBoundsTestEnable = context.lookup<PFN_vkCmdSetDepthBoundsTestEnableEXT>("vkCmdSetDepthBoundsTestEnableEXT");
        functions.cmdSetStencilTestEnable = context.lookup<PFN_vkCmdSetStencilTestEnableEXT>("vkCmdSetStencilTestEnableEXT");
        functions.cmdSetStencilOp = context.lookup<PFN_vkCmdSetStencilOpEXT>("vkCmdSetStencilOpEXT");
        functions.cmdSetStencilCompareMask = context.lookup<PFN_vkCmdSetStencilCompareMask>("vkCmdSetStencilCompareMask");
        functions.cmdSetStencilWriteMask = context.lookup<PFN_vkCmdSetStencilWriteMask>("vkCmdSetStencilWriteMask");
        functions.cmdSetStencilReference = context.lookup<PFN_vkCmdSetStencilReference>("vkCmdSetStencilReference");
    }
    functions.cmdBindVertexBuffers = context.Function<PFN_vkCmdBindVertexBuffers>("vkCmdBindVertexBuffers");
    functions.cmdBindIndexBuffer = context.Function<PFN_vkCmdBindIndexBuffer>("vkCmdBindIndexBuffer");
    functions.cmdDraw = context.Function<PFN_vkCmdDraw>("vkCmdDraw");
    functions.cmdDrawIndexed = context.Function<PFN_vkCmdDrawIndexed>("vkCmdDrawIndexed");
    functions.cmdDrawIndirect = context.Function<PFN_vkCmdDrawIndirect>("vkCmdDrawIndirect");
    functions.cmdDrawIndexedIndirect = context.Function<PFN_vkCmdDrawIndexedIndirect>("vkCmdDrawIndexedIndirect");
    functions.cmdCopyBufferToImage = context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage");
    functions.cmdCopyImageToBuffer = context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer");
    functions.cmdClearColorImage = context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage");
    functions.updateDescriptorSets = context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets");
    functions.allocateDescriptorSets = context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets");
    functions.getFenceStatus = context.Function<PFN_vkGetFenceStatus>("vkGetFenceStatus");
    functions.createBuffer = context.Function<PFN_vkCreateBuffer>("vkCreateBuffer");
    functions.destroyBuffer = context.Function<PFN_vkDestroyBuffer>("vkDestroyBuffer");
    functions.getBufferMemoryRequirements = context.Function<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements");
    functions.bindBufferMemory = context.Function<PFN_vkBindBufferMemory>("vkBindBufferMemory");
    functions.allocateMemory = context.Function<PFN_vkAllocateMemory>("vkAllocateMemory");
    functions.freeMemory = context.Function<PFN_vkFreeMemory>("vkFreeMemory");
    functions.mapMemory = context.Function<PFN_vkMapMemory>("vkMapMemory");
    functions.unmapMemory = context.Function<PFN_vkUnmapMemory>("vkUnmapMemory");
}

RenderTarget::RenderTarget(const Context& context, const ColorTarget& target, bool blending) : context(context) {
    VkFormatProperties properties{};
    context.formatProperties(context.physical, target.format, &properties);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT | (blending ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT : 0u);
    Require((properties.optimalTilingFeatures & required) == required, "render-target format does not support required operations");
    constexpr auto usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageFormatProperties supported{};
    Check(context.imageFormatProperties(context.physical, target.format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, 0, &supported), "vkGetPhysicalDeviceImageFormatProperties");
    Require(target.extent.width <= supported.maxExtent.width && target.extent.height <= supported.maxExtent.height && (supported.sampleCounts & VK_SAMPLE_COUNT_1_BIT) != 0 && target.bytes <= supported.maxResourceSize, "render target exceeds device image limits");
    Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "render target exceeds framebuffer limits");
    try {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = target.format;
        info.extent = {target.extent.width, target.extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        memory = AllocateImageMemory(context, allocation, requirements.memoryTypeBits, "vkAllocateMemory render target", &memoryType);
        memoryBytes = allocation.allocationSize;
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = target.format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView");
    } catch (...) {
        release();
        throw;
    }
}

RenderTarget::~RenderTarget() {
    release();
}

void RenderTarget::release() noexcept {
    if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
    if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
    ReleaseImageMemory(context, memory, memoryBytes, memoryType);
}

VkImage RenderTarget::Image() const {
    return image;
}

VkImageView RenderTarget::View() const {
    return view;
}

void NoteGuestWorkOnCommandBuffer(VkCommandBuffer commands) {
    GuestWorkMark::Note(commands);
}

CommandBatch::CommandBatch(const Context& context) : context(context) {
    // Records into the device's one command pool and submits to its queue: device-lock work only.
    GuestMemory::AssertGpuLockHeld("CommandBatch");
    try {
        gpuFlight = GuestMemory::BeginGpuFlight();
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = context.pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        Check(context.Function<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(context.device, &allocation, &commands), "vkAllocateCommandBuffers");
        VkFenceCreateInfo info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence created = VK_NULL_HANDLE;
        Check(context.Function<PFN_vkCreateFence>("vkCreateFence")(context.device, &info, nullptr, &created), "vkCreateFence");
        fence = created;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        Check(context.Function<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin), "vkBeginCommandBuffer");
        WriteCheckpointSentinel(context, commands);
        // Work recorded so far goes first in queue order, so this batch sees its results.
        if (auto* recorder = Recorder::Active(); recorder != nullptr && recorder->Recording()) recorder->Submit();
    } catch (...) {
        release();
        throw;
    }
}

CommandBatch::~CommandBatch() {
    release();
}

void CommandBatch::release() noexcept {
    GuestWorkMark::Forget(commands);
    if (pending) {
        auto result = context.Resolved(&DeviceFunctions::getFenceStatus, "vkGetFenceStatus")(context.device, fence);
        if (result == VK_NOT_READY) result = context.Function<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(context.queue);
        if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) std::terminate();
    }
    GuestMemory::EndGpuFlight(gpuFlight);
    gpuFlight = 0;
    if (commands) context.Function<PFN_vkFreeCommandBuffers>("vkFreeCommandBuffers")(context.device, context.pool, 1, &commands);
    if (fence) context.Function<PFN_vkDestroyFence>("vkDestroyFence")(context.device, fence, nullptr);
}

VkCommandBuffer CommandBatch::Handle() const {
    return commands;
}

void CommandBatch::SubmitAndWait() {
    Submit();
    Wait();
}

void CommandBatch::Submit() {
    PerformanceTimer timing("Graphics.Submit");
    Require(!submitted, "command batch has already been submitted");
    Check(context.Function<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands), "vkEndCommandBuffer");
    VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &commands;
    timing.Mark("command_end");
    DestroyDeferredAtIdle(context);
    Check(context.Function<PFN_vkQueueSubmit>("vkQueueSubmit")(context.queue, 1, &submission, fence), "vkQueueSubmit graphics");
    pending = true;
    submitted = true;
    timing.Mark("queue_submit");
    if (GuestWorkMark::Take(commands)) LibcRunReportMilestone_nid_no_patch(2);
}

void CommandBatch::Wait() {
    Require(submitted, "command batch has not been submitted");
    if (!pending) return;
    PerformanceTimer timing("Graphics.Wait");
    const auto result = context.Function<PFN_vkWaitForFences>("vkWaitForFences")(context.device, 1, &fence, VK_TRUE, 5'000'000'000ULL);
    timing.Mark("fence_wait");
    if (result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST) pending = false;
    Check(result, "vkWaitForFences graphics");
    // Recorded batches preceded this one, so their completions (write-backs) can run now.
    if (auto* recorder = Recorder::Active()) recorder->Reap();
}

}
