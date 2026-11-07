#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace AgcDriver::DriverDetail {

// Diagnostic (APS5_TRACE_EOP=1, v3 exec 11-07): end-of-pipe interrupts deferred behind recorded work, by queue: the lag
// from the guest packet to the delivery (the batch's completion running under a reap), and how many were delivered
// at once; [eop] every 10 s.
namespace {
bool traceEop() {
    static const bool enabled = std::getenv("APS5_TRACE_EOP") != nullptr;
    return enabled;
}
struct EopQueue { std::uint64_t deferred = 0, immediate = 0, lagUs = 0, maxLagUs = 0, piped = 0, submitUs = 0, maxSubmitUs = 0, startUs = 0, maxStartUs = 0, commitUs = 0, maxCommitUs = 0; std::array<std::uint64_t, 4> ahead{}; };
std::array<std::atomic<std::int64_t>, 256> pipeEnqueued{}, pipeStarted{};
std::int64_t pipeNs(std::chrono::steady_clock::time_point at) { return std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count(); }
std::mutex eopMutex;
std::array<EopQueue, 256> eopQueues;
std::chrono::steady_clock::time_point eopReport = std::chrono::steady_clock::now();
void reportEopLocked(std::chrono::steady_clock::time_point now) {
    if (now - eopReport < std::chrono::seconds(10)) return;
    eopReport = now;
    std::string line = "[eop] 10 s, by queue (deferred: count, packet-to-delivery avg / max ms; immediate count):";
    for (std::size_t q = 0; q < eopQueues.size(); ++q) {
        auto& e = eopQueues[q];
        if (e.deferred == 0 && e.immediate == 0 && e.piped == 0) continue;
        char text[128];
        std::snprintf(text, sizeof(text), " 0x%zx %llu %.1f / %.1f, %llu;", q, static_cast<unsigned long long>(e.deferred), e.deferred ? e.lagUs / 1000.0 / static_cast<double>(e.deferred) : 0.0, e.maxLagUs / 1000.0, static_cast<unsigned long long>(e.immediate));
        line += text;
        if (e.piped != 0) {
            char pipe[384];
            const auto n = static_cast<double>(e.piped);
            std::snprintf(pipe, sizeof(pipe), " [EOP packets %llu: guest submit to worker start avg %.1f / max %.1f ms, worker start to packet avg %.1f / max %.1f ms, packet to commit avg %.1f / max %.1f ms; queue-0 flips ahead at the packet 0/1/2/3+: %llu/%llu/%llu/%llu];", static_cast<unsigned long long>(e.piped), e.submitUs / 1000.0 / n, e.maxSubmitUs / 1000.0, e.startUs / 1000.0 / n, e.maxStartUs / 1000.0, e.commitUs / 1000.0 / n, e.maxCommitUs / 1000.0, static_cast<unsigned long long>(e.ahead[0]), static_cast<unsigned long long>(e.ahead[1]), static_cast<unsigned long long>(e.ahead[2]), static_cast<unsigned long long>(e.ahead[3]));
            line += pipe;
        }
        e = {};
    }
    std::fprintf(stderr, "%s\n", line.c_str());
}
}

// Frame marks for [frame-chain] (Acb.cpp): when the queue-0 worker last reached a flip packet and a queue-0 EOP packet.
std::atomic<std::int64_t> frameFlipNs{0}, frameEop0Ns{0};
std::atomic<std::uint64_t> frameFlips{0};

bool TraceWaitDrains() {
    return traceEop();
}

void TraceWaitDrain(std::uint32_t writer, std::chrono::steady_clock::duration drained) {
    static std::mutex mutex;
    static std::array<std::pair<std::uint64_t, std::uint64_t>, 256> byWriter{};
    static auto reported = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex);
    auto& entry = byWriter[writer & 0xffu];
    ++entry.first;
    entry.second += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(drained).count());
    const auto now = std::chrono::steady_clock::now();
    if (now - reported < std::chrono::seconds(10)) return;
    reported = now;
    std::string line = "[wait-drains] 10 s, queue-0 WAIT_REG_MEM drains by the awaited label's last writer queue (0xfe: none seen; count, drain ms):";
    for (std::size_t q = 0; q < byWriter.size(); ++q) {
        if (byWriter[q].first == 0) continue;
        char text[64];
        std::snprintf(text, sizeof(text), " 0x%zx %llu %.1f;", q, static_cast<unsigned long long>(byWriter[q].first), byWriter[q].second / 1000.0);
        line += text;
        byWriter[q] = {};
    }
    std::fprintf(stderr, "%s\n", line.c_str());
}

void TraceFlipReached() {
    if (!traceEop()) return;
    frameFlipNs.store(pipeNs(std::chrono::steady_clock::now()), std::memory_order_relaxed);
    frameFlips.fetch_add(1, std::memory_order_relaxed);
}

void TraceFrameMarks(std::int64_t* flipNs, std::int64_t* eop0Ns, std::uint64_t* flips) {
    *flipNs = frameFlipNs.load(std::memory_order_relaxed);
    *eop0Ns = frameEop0Ns.load(std::memory_order_relaxed);
    *flips = frameFlips.load(std::memory_order_relaxed);
}

void TracePipeStart(std::uint32_t queue, std::chrono::steady_clock::time_point enqueued) {
    if (!traceEop()) return;
    pipeEnqueued[queue % pipeEnqueued.size()].store(pipeNs(enqueued), std::memory_order_relaxed);
    pipeStarted[queue % pipeStarted.size()].store(pipeNs(std::chrono::steady_clock::now()), std::memory_order_relaxed);
}

std::int64_t TracePipeStarted(std::uint32_t queue) {
    return traceEop() ? pipeStarted[queue % pipeStarted.size()].load(std::memory_order_relaxed) : 0;
}

std::int64_t TracePipeSubmitted(std::uint32_t queue) {
    return traceEop() ? pipeEnqueued[queue % pipeEnqueued.size()].load(std::memory_order_relaxed) : 0;
}

void TracePipeEop(std::uint32_t queue, std::int64_t submittedNs, std::int64_t startedNs, std::int64_t seenNs, std::uint32_t flipsAhead) {
    if (!traceEop() || seenNs == 0) return;
    if (queue == 0) frameEop0Ns.store(seenNs, std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const auto nowNs = pipeNs(now);
    std::lock_guard lock(eopMutex);
    auto& e = eopQueues[queue % eopQueues.size()];
    ++e.piped;
    if (submittedNs != 0 && startedNs >= submittedNs) {
        const auto us = static_cast<std::uint64_t>((startedNs - submittedNs) / 1000);
        e.submitUs += us;
        e.maxSubmitUs = std::max(e.maxSubmitUs, us);
    }
    if (startedNs != 0 && seenNs >= startedNs) {
        const auto us = static_cast<std::uint64_t>((seenNs - startedNs) / 1000);
        e.startUs += us;
        e.maxStartUs = std::max(e.maxStartUs, us);
    }
    const auto commitUs = static_cast<std::uint64_t>(std::max<std::int64_t>(0, nowNs - seenNs) / 1000);
    e.commitUs += commitUs;
    e.maxCommitUs = std::max(e.maxCommitUs, commitUs);
    ++e.ahead[std::min<std::uint32_t>(flipsAhead, 3)];
    reportEopLocked(now);
}

void TraceEopDelivered(std::uint32_t queue, std::chrono::steady_clock::time_point registered, bool deferred) {
    if (!traceEop()) return;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(eopMutex);
    auto& e = eopQueues[queue % eopQueues.size()];
    if (deferred) {
        const auto lag = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now - registered).count());
        ++e.deferred;
        e.lagUs += lag;
        e.maxLagUs = std::max(e.maxLagUs, lag);
    } else {
        ++e.immediate;
    }
    reportEopLocked(now);
}

bool Driver::preparePacketMemory(const Submission& submission, QueueState& queue, std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, bool& wroteOnGpu, bool& endOfPipeInterrupt, bool& interruptDeferred, bool& drawPacket, bool& sampleDump) {
    static const bool drainAll = std::getenv("APS5_DRAIN_ALL") != nullptr;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    wroteOnGpu = false;

    bool orderedAlready = false;

    const auto interruptSelect = opcode == 0x49 ? (packet[2] >> 24u) & 7u : 0u;
    endOfPipeInterrupt = interruptSelect != 0 && interruptSelect != 3;
    interruptDeferred = false;
    if (endOfPipeInterrupt && traceEop()) {
        // The worker reached the EOP packet itself (async queues; queue 0 outside the draw pipeline): no commit lag.
        const auto seen = pipeNs(std::chrono::steady_clock::now());
        TracePipeEop(submission.queue, submission.enqueuedAt != std::chrono::steady_clock::time_point{} ? pipeNs(submission.enqueuedAt) : 0, TracePipeStarted(submission.queue), seen, flipsAhead.load(std::memory_order_relaxed));
    }
    if (!drainAll && endOfPipeInterrupt) {
        const auto label = Pm4::DecodeLabelWrite(packet);
        const bool storesNothing = (packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0;
        if (label.has_value() || storesNothing) {
            bumpEpoch(&EpochBumps::drains);
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            const auto localDevice = device.Load();
            recordDeferredLabels(localDevice.get(), submission.queue);
            bool workOpen = Graphics::Recorder::RecordedWorkSinceSubmit() != 0;
            int reason = localDevice != nullptr ? 0 : 1;
            if (label.has_value()) {
                const auto bytes = label->Bytes();
                const auto stamp = ++eventSerial;
                if (localDevice != nullptr && submission.queue == 0 && Graphics::StorageTexture::StoreDrawnLinear("label") != 0) workOpen = true;
                reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(label->address, bytes, stamp, submission.queue) : 4;
                if (reason == 1) GuestMemory::Write(label->address, bytes, 4);
                if (reason == 0 || reason == 1 || reason == 5) {
                    noteLabelStore(label->address, bytes, stamp, submission.queue);
                    Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
                }
                countLabelOutcome(reason);
                ++immediateLabels;
            } else {
                if (localDevice != nullptr && submission.queue == 0 && Graphics::StorageTexture::StoreDrawnLinear("interrupt") != 0) workOpen = true;
                if (localDevice != nullptr) Graphics::StorageTexture::StoreReadbackBeforeLabel("readback before interrupt");
                ++noOpLabels;
            }
            if (reason == 0 || reason == 5) {
                const auto queueId = submission.queue;
                interruptDeferred = localDevice->AfterRecordedWork([this, queueId, at = std::chrono::steady_clock::now()] { TraceEopDelivered(queueId, at, true); deliverGuestEopInterrupt(queueId); }, submission.queue == 0);
                if (interruptDeferred && workOpen) localDevice->SubmitRecorded(submission.queue == 0);
                wroteOnGpu = true;
            } else if (reason == 1) {
                wroteOnGpu = true;
            }
        }
    }
    if (opcode == 0x50 && packet.size() >= 7) {
        if (Pm4::DmaSource(packet) == 0 || Pm4::DmaSource(packet) == 3) Graphics::RequireNoMultisampleTarget(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), packet[6] & 0x3ffffffu, "a DMA copy");
        if (Pm4::DmaSource(packet) == 2 && Pm4::DmaDestination(packet) != 1) {
            const auto address = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
            Graphics::NoteDepthMetadataFill(address, packet[6] & 0x3ffffffu, packet[2]);
            Graphics::NoteColorMetadataFill(address, packet[6] & 0x3ffffffu, packet[2]);
        }
    }
    if (!drainAll && !endOfPipeInterrupt && (opcode == 0x49 || opcode == 0x37)) {
        if (const auto label = Pm4::DecodeLabelWrite(packet)) {
            const auto bytes = label->Bytes();
            if (DeferLabels() && bytes.size() <= DeferredLabel::Capacity && bytes.size() % 4 == 0 && label->address % 4 == 0) {

                auto& deferred = deferredLabels();
                if (deferred.labels.empty()) deferred.since = std::chrono::steady_clock::now();
                auto& entry = deferred.labels.emplace_back();
                entry.address = label->address;
                entry.size = bytes.size();
                std::memcpy(entry.bytes.data(), bytes.data(), bytes.size());
                ++queuedLabels;
                wroteOnGpu = true;
            } else {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();

                recordDeferredLabels(localDevice.get(), submission.queue);
                const auto stamp = ++eventSerial;
                const auto reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(label->address, bytes, stamp, submission.queue) : 4;
                wroteOnGpu = reason == 0 || reason == 5;
                if (reason == 1) {

                    GuestMemory::Write(label->address, bytes, 4);
                    wroteOnGpu = true;
                }
                if (wroteOnGpu) noteLabelStore(label->address, bytes, stamp, submission.queue);
                Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
                countLabelOutcome(reason);
                ++immediateLabels;
            }
        } else if (opcode == 0x49 ? ((packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0) : (packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u)) == 0) {
            orderedAlready = true;
            ++noOpLabels;
        } else {
            ++labelFallbacks[4];
        }
    }

    static const bool cpuStores = std::getenv("APS5_CPU_STORES") != nullptr;
    if (!drainAll && !cpuStores && (opcode == 0x40 || opcode == 0x50 || opcode == 0x83)) {
        constexpr std::size_t gpuStoreLimit = 65536;
        bool drained = true;
        // A DMA from the GDS into memory (an append counter read back for an indirect dispatch) is a copy from the GDS
        // buffer, which is bound like guest memory, so it takes the GPU copy path too (APS5_NO_DMA_COPY_BEHIND_WRITES=1
        // off, below).
        const auto copy = opcode != 0x50 ? std::nullopt : [&]() -> std::optional<Pm4::MemoryCopy> {
            if (auto decoded = Pm4::DecodeMemoryCopy(packet)) return decoded;
            if (packet.size() != 7 || Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDmaCopyBehindWrites)) return std::nullopt;
            if (Pm4::DmaSource(packet) != Pm4::DmaSelectGds || Pm4::DmaDestination(packet) == Pm4::DmaSelectGds || Pm4::DmaDestination(packet) == Pm4::DmaDestinationNowhere || packet[3] != 0) return std::nullopt;
            const std::size_t bytes = Pm4::DmaBytes(packet);
            if (bytes == 0 || packet[2] + bytes > Pm4::GdsBytes) return std::nullopt;
            return Pm4::MemoryCopy{Pm4::GdsAddress() + packet[2], packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u), bytes};
        }();
        const auto copyBehindPendingSource = [&] {
            if (!copy.has_value() || copy->bytes > gpuStoreLimit || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->source), copy->bytes) || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->destination), copy->bytes, true)) return false;
            const auto end = copy->source + copy->bytes;
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{copy->source, end}}};
            const auto& queued = deferredLabels().labels;
            const bool labelQueued = std::any_of(queued.begin(), queued.end(), [&](const DeferredLabel& label) { return label.address < end && copy->source < label.address + label.size; });
            const bool behindWritesEnabled = !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDmaCopyBehindWrites);
            if (!labelQueued && !Graphics::Recorder::SnapshotWriteOverlaps(copy->source, copy->bytes) && !Graphics::StorageTexture::AnyPendingOverlaps(range) && !Graphics::AnyShadowedOverlaps(copy->source, copy->bytes) && !behindWritesEnabled) return false;
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Copy);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            const auto localDevice = device.Load();
            if (localDevice == nullptr) return false;
            recordDeferredLabels(localDevice.get(), submission.queue);
            // A source written by recorded GPU work (dispatches), not only by pending label stores, is copied on the GPU
            // in the queue's order too, instead of the store path reading it on the CPU after the GPU wrote it (a GPU ->
            // CPU -> GPU round trip, ~4 ms per copy on queue 0x30 in the menu); +4.0 % menu presents (best308).
            // APS5_NO_DMA_COPY_BEHIND_WRITES=1 (runtime switch) keeps the store path for those.
            const auto* recorder = Graphics::Recorder::Active();
            const bool behindWrites = behindWritesEnabled && (Graphics::Recorder::SnapshotWriteOverlaps(copy->source, copy->bytes) || (recorder != nullptr && recorder->PendingWriteOverlaps(copy->source, copy->bytes)));
            if (!localDevice->StoresPendingOver(copy->source, copy->bytes) && !behindWrites) return false;
            if (!localDevice->GpuCopyPossible(copy->destination, copy->source, copy->bytes)) return false;
            const auto outcome = localDevice->CopyBuffer(copy->destination, copy->source, copy->bytes, 0, std::numeric_limits<std::size_t>::max(), 0, 0, submission.queue, [](std::span<const std::byte>, std::uint64_t) {});
            if (outcome.path != 1 && outcome.path != 3) throw std::runtime_error("AGC driver: a DMA_DATA copy of up to 64 KiB that GpuCopyPossible accepted took copy path " + std::to_string(outcome.path));
            return true;
        };
        if (copyBehindPendingSource()) {
            ++copiesBehindPendingSources;
            wroteOnGpu = true;
            drained = false;
        } else if (const auto store = Pm4::ResolveStore(packet, queue, gpuStoreLimit)) {
            const auto bytes = store->Bytes();
            if (bytes.empty()) {

                orderedAlready = true;
                drained = false;
            } else {

                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();

                Graphics::StorageTexture::FlushPending(store->address, bytes.size(), nullptr, "packet store", Graphics::PublishScope::PartialUnits);

                recordDeferredLabels(localDevice.get(), submission.queue);
                const auto stamp = ++eventSerial;
                const auto reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(store->address, bytes, stamp, submission.queue) : 4;
                if (reason == 0 || reason == 1 || reason == 5) noteLabelStore(store->address, bytes, stamp, submission.queue);
                if (reason == 0 || reason == 5) {
                    if (reason == 0) ++storesOnGpu;
                    else ++storesBehindCompletions;
                    wroteOnGpu = true;
                    drained = false;
                } else if (reason == 1) {
                    GuestMemory::Write(store->address, bytes, 1);
                    ++storesOnCpu;
                    wroteOnGpu = true;
                    drained = false;
                }
                Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
            }
        }
        if (drained && opcode == 0x50) {
            if (copy.has_value() && copy->bytes > gpuStoreLimit && GuestMemory::Accessible(reinterpret_cast<const void*>(copy->source), copy->bytes) && GuestMemory::Accessible(reinterpret_cast<const void*>(copy->destination), copy->bytes, true)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Copy);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                if (const auto localDevice = device.Load()) {
                    recordDeferredLabels(localDevice.get(), submission.queue);
                    const auto outcome = localDevice->CopyBuffer(copy->destination, copy->source, copy->bytes, 0, std::numeric_limits<std::size_t>::max(), 0, 0, submission.queue, [](std::span<const std::byte>, std::uint64_t) {});
                    if (outcome.path == 1 || outcome.path == 3) {
                        wroteOnGpu = true;
                        drained = false;
                    } else if (outcome.path == 2 && !Graphics::RegisteredReadableCovers(copy->destination, copy->bytes) && !localDevice->StoresPendingOver(copy->destination, copy->bytes)) {
                        Graphics::StorageTexture::FlushPending(copy->source, copy->bytes, nullptr, "buffer copy source", Graphics::PublishScope::Whole);
                        if (auto* recorder = Graphics::Recorder::Active()) recorder->SyncThrough(copy->source, copy->bytes);
                        ++copiesToHostMemory;
                        orderedAlready = true;
                        drained = false;
                    }
                }
            }
        }
        // An immediate-data DMA into the GDS (each queue resets its GDS counters about once a frame) is recorded on the
        // GPU as a fill of the GDS buffer, in the queue's order, instead of a device drain (a wait for all recorded GPU
        // work, ~13 ms on queue 0x30 in the menu) and a CPU store. APS5_NO_GDS_GPU_FILL=1 (runtime switch) drains.
        if (drained && opcode == 0x50 && Pm4::DmaSource(packet) == 2 && Pm4::DmaDestination(packet) == Pm4::DmaSelectGds && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoGdsGpuFill)) {
            const std::uint64_t offset = packet[4];
            const std::size_t bytes = packet[6] & 0x3ffffffu;
            const auto destination = Pm4::GdsAddress() + offset;
            if (packet[5] == 0 && bytes != 0 && offset + bytes <= Pm4::GdsBytes && destination % 16 == 0 && bytes % 16 == 0 && GuestMemory::Accessible(reinterpret_cast<const void*>(destination), bytes, true)) {
                if (const auto localDevice = device.Load()) {
                    storeBufferFill(submission.queue, destination, bytes, {packet[2], packet[2], packet[2], packet[2]}, localDevice, false, false);
                    ++gdsFills;
                    wroteOnGpu = true;
                    drained = false;
                }
            } else if (packet[5] == 0 && bytes != 0 && bytes <= gpuStoreLimit && offset + bytes <= Pm4::GdsBytes && destination % 4 == 0 && bytes % 4 == 0 && GuestMemory::Accessible(reinterpret_cast<const void*>(destination), bytes, true)) {
                // Smaller or 4-byte aligned resets: an immediate store of the repeated value, like a label.
                std::vector<std::byte> value(bytes);
                for (std::size_t at = 0; at < bytes; at += 4) std::memcpy(value.data() + at, &packet[2], 4);
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();
                recordDeferredLabels(localDevice.get(), submission.queue);
                const auto stamp = ++eventSerial;
                const auto reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(destination, value, stamp, submission.queue) : 4;
                if (reason == 0 || reason == 1 || reason == 5) noteLabelStore(destination, value, stamp, submission.queue);
                if (reason == 0 || reason == 5) {
                    ++gdsFills;
                    wroteOnGpu = true;
                    drained = false;
                } else if (reason == 1) {
                    GuestMemory::Write(destination, value, 1);
                    ++gdsFills;
                    wroteOnGpu = true;
                    drained = false;
                }
                Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
            }
        }
        if (drained && opcode == 0x50 && Pm4::DmaSource(packet) == 2 && Pm4::DmaDestination(packet) != 1) {
            const auto destination = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
            const std::size_t bytes = packet[6] & 0x3ffffffu;
            if (bytes != 0 && destination % 16 == 0 && bytes % 16 == 0 && GuestMemory::Accessible(reinterpret_cast<const void*>(destination), bytes, true)) {
                if (const auto localDevice = device.Load()) {
                    storeBufferFill(submission.queue, destination, bytes, {packet[2], packet[2], packet[2], packet[2]}, localDevice, false, false);
                    ++dmaFills;
                    wroteOnGpu = true;
                    drained = false;
                }
            }
        }
        if (drained) ++storesDrained;
    }

    static const bool drawDrain = std::getenv("APS5_DRAW_DRAIN") != nullptr;
    drawPacket = Pm4::DrawOpcode(opcode);
    sampleDump = Pm4::SampleDump(packet);
    if (sampleDump && !drainAll) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (const auto localDevice = device.Load()) {
            recordDeferredLabels(localDevice.get(), submission.queue);
            wroteOnGpu = localDevice->DumpSamplesOnGpu(Pm4::SampleDumpAddress(packet));
        }
    }

    if (header == FlipPacketHeader) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        static_cast<void>(Graphics::StorageTexture::StoreSmallPending("flip"));
    }

    static const bool syncFlip = std::getenv("APS5_SYNC_FLIP") != nullptr;
    const bool drains = drainAll ? ((Pm4::AccessesMemory(header) && opcode != 0x16) || opcode == 0x42 || opcode == 0x46 || opcode == 0x58 || header == FlipPacketHeader)
                                 : (!wroteOnGpu && !orderedAlready && (opcode == 0x49 || opcode == 0x37 || opcode == 0x40 || opcode == 0x45 || opcode == 0x50 || opcode == 0x83 || sampleDump || (drawPacket && drawDrain) || (header == FlipPacketHeader && syncFlip)));
    if (drains) {

        static const bool unlockedDrain = std::getenv("APS5_NO_UNLOCKED_DRAIN") == nullptr && !drainAll;

        bumpEpoch(&EpochBumps::drains);
        std::shared_ptr<VulkanDevice> draining;
        std::uint64_t epoch = 0;
        {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            if (profile) {
                ++drainCounts[header == FlipPacketHeader ? 0xffffu : opcode];
                ++drainTotal;
                if (std::chrono::steady_clock::now() - lastSyncReport > std::chrono::seconds(10)) {
                    lastSyncReport = std::chrono::steady_clock::now();
                    reportSync();
                }
            }
            draining = device.Load();

            recordDeferredLabels(draining.get(), submission.queue);
            if (draining != nullptr) {
                static_cast<void>(Graphics::StorageTexture::StoreDrawnLinear("drain"));
                if (unlockedDrain && draining->CanWaitUnlocked()) epoch = draining->SubmitAndEpoch();
                else draining->WaitIdle();
            }
            if (epoch == 0) draining.reset();
        }
        if (epoch != 0) {
            ++unlockedDrains;
            try {
                draining->WaitRecorded(epoch);
            } catch (...) {

                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                draining.reset();
                throw;
            }
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            draining->ReapRecorded(epoch);
            draining.reset();
        }
    }
    return drains;
}

void Driver::dumpSampleCounters(std::uint64_t address) {
    std::uint64_t samples = 0;
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        auto* recorder = Graphics::Recorder::Active();
        require(recorder != nullptr, "occlusion counters without the command recorder are not implemented");
        recorder->CountSamples();
        samples = recorder->SamplesTotal();
    }
    constexpr std::uint64_t ready = 1ull << 63u;
    for (std::uint64_t db = 0; db < Pm4::SampleDumpBlocks; ++db) {
        const std::uint64_t value = ready | (db == 0 ? samples : 0u);
        GuestMemory::Write(address + db * Pm4::SampleDumpStride, std::as_bytes(std::span(&value, 1)).first(Pm4::SampleDumpBlockBytes), 8);
    }
}

bool Driver::enqueueSampleDumpPacket(std::span<const std::uint32_t> packet, std::uint32_t opcode, std::uint32_t queue) {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_SAMPLE_DUMPS") == nullptr;
    if (!ordered || opcode != 0x46 || queue != 0 || packet.size() < 4 || !Pm4::SampleDump(packet) || !DrawPipeline::Active() || !deferredLabels().labels.empty()) return false;
    const auto address = Pm4::SampleDumpAddress(packet);
    if (address == 0 || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), Pm4::SampleDumpBytes, true)) return false;
    DrawPipeline::Queue0().Enqueue([this, queue, address] {
        GuestMemory::SetCurrentPacket(0x46, queue);
        bool wrote = false;
        {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            if (const auto localDevice = device.Load()) {
                recordDeferredLabels(localDevice.get(), queue);
                wrote = localDevice->DumpSamplesOnGpu(address);
                if (!wrote) localDevice->WaitIdle();
            }
        }
        if (!wrote) dumpSampleCounters(address);
    }, {{address, address + Pm4::SampleDumpBytes}});
    ++sampleDumpsQueued;
    Graphics::Recorder::CountRecordedWork();
    return true;
}

bool Driver::enqueueDmaPacket(std::span<const std::uint32_t> packet, std::uint32_t opcode, std::uint32_t queue, const QueueState& state) {
    const bool enabled = DrawPipeline::DmaItems();
    if (!enabled || opcode != 0x50 || queue != 0 || packet.size() < 7 || !DrawPipeline::Active() || !deferredLabels().labels.empty()) return false;
    const auto source = ((packet[1] >> 29u) & 3u) | ((packet[6] >> 24u) & 4u) | ((packet[6] >> 25u) & 8u);
    constexpr std::size_t immediateLimit = std::size_t{16} << 20u;
    constexpr std::size_t gpuStoreLimit = 65536;
    if (source == 2 && Pm4::DmaBytes(packet) > gpuStoreLimit && Pm4::DmaDestination(packet) != Pm4::DmaSelectGds) {
        const auto fillAddress = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
        const auto fillBytes = Pm4::DmaBytes(packet);
        const auto pattern = packet[2];
        if (fillAddress % 16 != 0 || fillBytes % 16 != 0 || !GuestMemory::Accessible(reinterpret_cast<const void*>(fillAddress), fillBytes, true) || !device.Load()) return false;
        DrawPipeline::Queue0().Enqueue([this, queue, fillAddress, fillBytes, pattern] {
            GuestMemory::SetCurrentPacket(0x50, queue);
            if (const auto localDevice = device.Load()) {
                storeBufferFill(queue, fillAddress, fillBytes, {pattern, pattern, pattern, pattern}, localDevice, false, false);
                ++dmaFills;
            }
        }, {{fillAddress, fillAddress + fillBytes}});
        return true;
    }
    if (source == 2) {
        const auto store = Pm4::ResolveStore(packet, state, immediateLimit);
        if (!store.has_value() || store->Bytes().empty() || !GuestMemory::Accessible(reinterpret_cast<const void*>(store->address), store->Bytes().size(), true)) return false;
        const auto address = store->address;
        std::vector<std::byte> bytes(store->Bytes().begin(), store->Bytes().end());
        const auto size = bytes.size();
        const bool metadataFill = Pm4::DmaDestination(packet) != 1;
        const auto fillAddress = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
        const auto fillBytes = packet[6] & 0x3ffffffu;
        const auto pattern = packet[2];
        DrawPipeline::Queue0().Enqueue([this, queue, address, bytes = std::move(bytes), metadataFill, fillAddress, fillBytes, pattern] {
            GuestMemory::SetCurrentPacket(0x50, queue);
            if (metadataFill) {
                Graphics::NoteDepthMetadataFill(fillAddress, fillBytes, pattern);
                Graphics::NoteColorMetadataFill(fillAddress, fillBytes, pattern);
            }
            commitDmaStore(queue, address, bytes);
        }, {{address, address + size}});
        return true;
    }
    if (Pm4::DmaSource(packet) == 0 || Pm4::DmaSource(packet) == 3) Graphics::RequireNoMultisampleTarget(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), packet[6] & 0x3ffffffu, "a DMA copy");
    const auto copy = Pm4::DecodeMemoryCopy(packet);
    if (!copy.has_value() || copy->destination % 4 != 0 || copy->bytes % 4 != 0 || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->source), copy->bytes) || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->destination), copy->bytes, true)) return false;
    DrawPipeline::Queue0().Enqueue([this, queue, copy = *copy] {
        DrawPipeline::Queue0().ReleaseRunningReads();
        GuestMemory::SetCurrentPacket(0x50, queue);
        constexpr std::size_t gpuStoreLimit = 65536;
        if (copy.bytes > gpuStoreLimit) {
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Copy);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            if (const auto localDevice = device.Load()) {
                recordDeferredLabels(localDevice.get(), queue);
                const auto outcome = localDevice->CopyBuffer(copy.destination, copy.source, copy.bytes, 0, std::numeric_limits<std::size_t>::max(), 0, 0, queue, [](std::span<const std::byte>, std::uint64_t) {});
                if (outcome.path == 1 || outcome.path == 3) return;
                if (outcome.path == 2 && !Graphics::RegisteredReadableCovers(copy.destination, copy.bytes) && !localDevice->StoresPendingOver(copy.destination, copy.bytes)) ++copiesToHostMemory;
            }
        }
        std::vector<std::byte> bytes(copy.bytes);
        GuestMemory::Read(copy.source, bytes, 1);
        commitDmaStore(queue, copy.destination, bytes);
    }, {{copy->destination, copy->destination + copy->bytes}}, 0, {}, {{copy->source, copy->source + copy->bytes}});
    return true;
}

void Driver::commitDmaStore(std::uint32_t queue, std::uint64_t address, std::span<const std::byte> bytes) {
    constexpr std::size_t gpuStoreLimit = 65536;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    const auto localDevice = device.Load();
    Graphics::StorageTexture::FlushPending(address, bytes.size(), nullptr, "packet store", Graphics::PublishScope::PartialUnits);
    recordDeferredLabels(localDevice.get(), queue);
    int reason = 4;
    std::uint64_t stamp = 0;
    if (bytes.size() <= gpuStoreLimit && localDevice != nullptr) {
        stamp = ++eventSerial;
        reason = localDevice->WriteLabelOnGpu(address, bytes, stamp, queue);
    }
    if (reason == 0 || reason == 5 || reason == 6) {
        noteLabelStore(address, bytes, stamp, queue);
        if (reason == 0) ++storesOnGpu;
        else ++storesBehindCompletions;
    } else {
        if (reason != 1 && localDevice != nullptr) localDevice->WaitIdle();
        GuestMemory::Write(address, bytes, 1);
        noteLabelStore(address, bytes, stamp != 0 ? stamp : ++eventSerial, queue);
        ++storesOnCpu;
    }
    Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
}

bool Driver::labelPacketQueues(std::span<const std::uint32_t> packet, std::uint32_t opcode) {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_LABELS") == nullptr;
    if (!ordered || !DeferLabels() || (opcode != 0x49 && opcode != 0x37) || !deferredLabels().labels.empty()) return false;
    if (Pm4::DecodeLabelWrite(packet).has_value()) return true;
    return opcode == 0x49 && ((packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0);
}

bool Driver::enqueueLabelPacket(std::span<const std::uint32_t> packet, std::uint32_t opcode, std::uint32_t queue) {
    if (!labelPacketQueues(packet, opcode)) return false;
    const auto interruptSelect = opcode == 0x49 ? (packet[2] >> 24u) & 7u : 0u;
    const bool endOfPipeInterrupt = interruptSelect != 0 && interruptSelect != 3;
    const auto label = Pm4::DecodeLabelWrite(packet);
    if (!label.has_value()) {
        if (opcode != 0x49) return false;
        const bool storesNothing = (packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0;
        if (!storesNothing) return false;
        if (!endOfPipeInterrupt) {
            ++noOpLabels;
            return true;
        }
    }
    std::uint64_t address = 0;
    std::vector<std::byte> bytes;
    std::vector<DrawPipeline::Range> writes;
    if (label.has_value()) {
        const auto stored = label->Bytes();
        address = label->address;
        bytes.assign(stored.begin(), stored.end());
        writes.emplace_back(address, address + bytes.size());
    }
    if (endOfPipeInterrupt) bumpEpoch(&EpochBumps::drains);
    auto stored = bytes;
    auto labelBytes = bytes;
    const auto pipeSubmitted = endOfPipeInterrupt ? TracePipeSubmitted(queue) : 0;
    const auto pipeStartedAt = endOfPipeInterrupt ? TracePipeStarted(queue) : 0;
    const auto pipeSeen = endOfPipeInterrupt && pipeSubmitted != 0 ? pipeNs(std::chrono::steady_clock::now()) : 0;
    const auto pipeAhead = flipsAhead.load(std::memory_order_relaxed);
    auto enqueue =[this, queue, opcode, address, bytes = std::move(bytes), endOfPipeInterrupt, writes, stored = std::move(stored), pipeSubmitted, pipeStartedAt, pipeSeen, pipeAhead]() mutable {
        DrawPipeline::Queue0().Enqueue([this, queue, opcode, address, bytes = std::move(bytes), endOfPipeInterrupt, pipeSubmitted, pipeStartedAt, pipeSeen, pipeAhead] {
            if (pipeSeen != 0) TracePipeEop(queue, pipeSubmitted, pipeStartedAt, pipeSeen, pipeAhead);
            GuestMemory::SetCurrentPacket(opcode, queue);
            commitLabel(queue, address, bytes, endOfPipeInterrupt);
        }, std::move(writes), address, std::move(stored));
    };
    if (queue == 0 && DrawFinisher::Overlap() && DrawFinisher::Busy()) {
        DrawFinisher::Pending held;
        held.unknown = false;
        held.writes = std::move(writes);
        held.labelAddress = address;
        held.labelBytes = std::move(labelBytes);
        DrawFinisher::Get().Enqueue(std::move(enqueue), std::move(held));
        return true;
    }
    enqueue();
    return true;
}

void Driver::commitLabel(std::uint32_t queue, std::uint64_t address, std::span<const std::byte> bytes, bool endOfPipeInterrupt) {
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    const auto localDevice = device.Load();
    recordDeferredLabels(localDevice.get(), queue);
    bool workOpen = Graphics::Recorder::RecordedWorkSinceSubmit() != 0;
    int reason = localDevice != nullptr ? 0 : 1;
    if (!bytes.empty()) {
        const auto stamp = ++eventSerial;
        if (localDevice != nullptr && queue == 0 && Graphics::StorageTexture::StoreDrawnLinear("label") != 0) workOpen = true;
        reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(address, bytes, stamp, queue) : 4;
        if (reason != 0 && reason != 5 && reason != 6) {
            if (reason != 1 && localDevice != nullptr) localDevice->WaitIdle();
            GuestMemory::Write(address, bytes, 4);
        }
        noteLabelStore(address, bytes, stamp, queue);
        Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
        countLabelOutcome(reason);
        ++immediateLabels;
    } else {
        ++noOpLabels;
    }
    if (endOfPipeInterrupt) {
        bool deferred = false;
        if (bytes.empty() && localDevice != nullptr && queue == 0 && Graphics::StorageTexture::StoreDrawnLinear("interrupt") != 0) workOpen = true;
        if (localDevice != nullptr && (reason == 0 || reason == 5 || reason == 6)) {
            deferred = localDevice->AfterRecordedWork([this, queue, at = std::chrono::steady_clock::now()] { TraceEopDelivered(queue, at, true); deliverGuestEopInterrupt(queue); }, queue == 0);
            if (deferred && workOpen) localDevice->SubmitRecorded(queue == 0);
        }
        if (!deferred) {
            TraceEopDelivered(queue, {}, false);
            deliverGuestEopInterrupt(queue);
        }
    }
    submitDueAfterCommit(localDevice.get());
}

void Driver::submitDueAfterCommit(VulkanDevice* localDevice) {
    if (localDevice == nullptr) return;
    const auto pending = Graphics::Recorder::PendingLabelSince();
    const bool due = pending.has_value() && std::chrono::steady_clock::now() - *pending >= labelFlushDeadline();
    const bool capped = batchCap() != 0 && Graphics::Recorder::RecordedWorkSinceSubmit() >= batchCap();
    if (due || capped) localDevice->SubmitRecorded(true);
}

}
