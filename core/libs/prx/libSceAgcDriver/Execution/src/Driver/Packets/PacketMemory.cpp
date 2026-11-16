#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Core/include/SubmitPolicy.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "ThreadOwned.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <chrono>
#include <cstdio>
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"

namespace AgcDriver::DriverDetail {

namespace {

bool gdsWritesPending(std::span<const std::uint32_t> packet, const VulkanDevice* localDevice) {
    const auto address = Pm4::GdsAddress() + packet[4];
    const auto bytes = static_cast<std::size_t>(Pm4::DmaBytes(packet));
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{address, address + bytes}}};
    if (Graphics::Recorder::SnapshotWriteOverlaps(address, bytes) || Graphics::StorageTexture::AnyPendingOverlaps(range) || Graphics::AnyShadowedOverlaps(address, bytes)) return true;
    if (localDevice == nullptr || !Pm4::GdsImportable()) return false;
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    return localDevice->StoresPendingOver(address, bytes);
}

std::uint64_t dmaAddress(std::uint32_t low, std::uint32_t high) {
    return low | (static_cast<std::uint64_t>(high) << 32u);
}

bool gdsSpan(std::uint64_t offset, std::size_t bytes) {
    return offset <= Pm4::GdsBytes && bytes <= Pm4::GdsBytes - offset;
}

std::optional<Pm4::MemoryCopy> decodeCopy(std::span<const std::uint32_t> packet) {
    if (auto copy = Pm4::DecodeMemoryCopy(packet)) return copy;
    if (packet.size() != 7 || !Pm4::GdsImportable()) return std::nullopt;
    const auto source = Pm4::DmaSource(packet);
    const auto destination = Pm4::DmaDestination(packet);
    const std::size_t bytes = Pm4::DmaBytes(packet);
    if (bytes == 0) return std::nullopt;
    if (source == Pm4::DmaSelectGds && (destination == 0 || destination == 3) && gdsSpan(packet[2], bytes)) return Pm4::MemoryCopy{Pm4::GdsAddress() + packet[2], dmaAddress(packet[4], packet[5]), bytes};
    if ((source == 0 || source == 3) && destination == Pm4::DmaSelectGds && gdsSpan(packet[4], bytes)) return Pm4::MemoryCopy{dmaAddress(packet[2], packet[3]), Pm4::GdsAddress() + packet[4], bytes};
    return std::nullopt;
}

}

bool Driver::preparePacketMemory(const Submission& submission, QueueState& queue, std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode, std::optional<std::uint32_t> interrupt, bool& wroteOnGpu, bool& endOfPipeInterrupt, bool& interruptDeferred, bool& drawPacket, bool& sampleDump) {
    static const bool drainAll = std::getenv("APS5_DRAIN_ALL") != nullptr;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    wroteOnGpu = false;

    bool orderedAlready = false;

    endOfPipeInterrupt = interrupt.has_value();
    interruptDeferred = false;
    if (!drainAll && endOfPipeInterrupt) {
        const auto label = Pm4::DecodeLabelWrite(packet);
        const bool storesNothing = (packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0;
        if (label.has_value() || storesNothing) {
            bumpEpoch(&EpochBumps::interrupts);
            ++DrainEpochCounts().interrupts;
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
                const auto context = *interrupt;
                interruptDeferred = localDevice->AfterRecordedWork([this, queueId, context] { deliverGuestEopInterrupt(queueId, context); }, submission.queue == 0);
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
        const auto copy = opcode == 0x50 ? decodeCopy(packet) : std::nullopt;
        const auto copyBehindPendingSource = [&] {
            if (!copy.has_value() || copy->bytes > gpuStoreLimit || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->source), copy->bytes) || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->destination), copy->bytes, true)) return false;
            const auto end = copy->source + copy->bytes;
            const std::array<std::pair<std::uint64_t, std::uint64_t>, 1> range{{{copy->source, end}}};
            const auto& queued = deferredLabels().labels;
            const bool labelQueued = std::any_of(queued.begin(), queued.end(), [&](const DeferredLabel& label) { return label.address < end && copy->source < label.address + label.size; });
            if (!labelQueued && !Graphics::Recorder::SnapshotWriteOverlaps(copy->source, copy->bytes) && !Graphics::StorageTexture::AnyPendingOverlaps(range) && !Graphics::AnyShadowedOverlaps(copy->source, copy->bytes)) return false;
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Copy);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            const auto localDevice = device.Load();
            if (localDevice == nullptr) return false;
            recordDeferredLabels(localDevice.get(), submission.queue);
            if (!localDevice->StoresPendingOver(copy->source, copy->bytes) || !localDevice->GpuCopyPossible(copy->destination, copy->source, copy->bytes)) return false;
            const auto outcome = localDevice->CopyBuffer(copy->destination, copy->source, copy->bytes, 0, std::numeric_limits<std::size_t>::max(), 0, 0, submission.queue, [](std::span<const std::byte>, std::uint64_t) {});
            if (outcome.path != 1 && outcome.path != 3) throw std::runtime_error("AGC driver: a DMA_DATA copy of up to 64 KiB that GpuCopyPossible accepted took copy path " + std::to_string(outcome.path));
            return true;
        };
        const bool gdsDestination = opcode == 0x50 && packet.size() >= 7 && Pm4::DmaDestination(packet) == Pm4::DmaSelectGds;
        const bool gdsOnGpu = gdsDestination && Pm4::GdsImportable();
        if ((!gdsDestination || gdsOnGpu) && copyBehindPendingSource()) {
            ++copiesBehindPendingSources;
            wroteOnGpu = true;
            drained = false;
        } else if (gdsDestination && !gdsWritesPending(packet, device.Load().get())) {
            orderedAlready = true;
            drained = false;
        } else if (const auto store = gdsDestination && !gdsOnGpu ? std::nullopt : Pm4::ResolveStore(packet, queue, gpuStoreLimit)) {
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
                } else if (reason == 1 || (reason == 3 && opcode == 0x40 && Pm4::CopyDataSource(packet) == Pm4::CopyDataGpuClockSource && !localDevice->StoresPendingOver(store->address, bytes.size()))) {
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
                    } else if (outcome.path == 2 && !Graphics::RegisteredReadableCovers(copy->destination, copy->bytes)) {
                        auto* recorder = Graphics::Recorder::Active();
                        if (recorder != nullptr && localDevice->StoresPendingOver(copy->destination, copy->bytes)) recorder->SyncThrough(copy->destination, copy->bytes);
                        if (!localDevice->StoresPendingOver(copy->destination, copy->bytes)) {
                            Graphics::StorageTexture::FlushPending(copy->source, copy->bytes, nullptr, "buffer copy source", Graphics::PublishScope::Whole);
                            if (recorder != nullptr) recorder->SyncThrough(copy->source, copy->bytes);
                            ++copiesToHostMemory;
                            orderedAlready = true;
                            drained = false;
                        }
                    }
                }
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
        ++DrainEpochCounts().packets[header == FlipPacketHeader ? DrainEpochs::Flip : opcode & 0xffu];
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
    DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::SampleDump, [this, queue, address] {
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
    if (opcode != 0x50 || queue != 0 || packet.size() < 7 || !DrawPipeline::Active() || !deferredLabels().labels.empty()) return false;
    const auto source = ((packet[1] >> 29u) & 3u) | ((packet[6] >> 24u) & 4u) | ((packet[6] >> 25u) & 8u);
    if (Pm4::DmaDestination(packet) == Pm4::DmaSelectGds && gdsWritesPending(packet, device.Load().get())) return false;
    constexpr std::size_t immediateLimit = std::size_t{16} << 20u;
    constexpr std::size_t gpuStoreLimit = 65536;
    if (source == 2 && Pm4::DmaBytes(packet) > gpuStoreLimit && Pm4::DmaDestination(packet) != Pm4::DmaSelectGds) {
        const auto fillAddress = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
        const auto fillBytes = Pm4::DmaBytes(packet);
        const auto pattern = packet[2];
        if (fillAddress % 16 != 0 || fillBytes % 16 != 0 || !GuestMemory::Accessible(reinterpret_cast<const void*>(fillAddress), fillBytes, true) || !device.Load()) return false;
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::DmaFill, [this, queue, fillAddress, fillBytes, pattern] {
            GuestMemory::SetCurrentPacket(0x50, queue);
            if (const auto localDevice = device.Load()) {
                storeBufferFill(queue, fillAddress, fillBytes, {pattern, pattern, pattern, pattern}, localDevice, false, true);
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
        const bool gds = Pm4::DmaDestination(packet) == Pm4::DmaSelectGds;
        const bool metadataFill = Pm4::DmaDestination(packet) != 1;
        const auto fillAddress = packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
        const auto fillBytes = packet[6] & 0x3ffffffu;
        const auto pattern = packet[2];
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::DmaWrite, [this, queue, address, bytes = std::move(bytes), gds, metadataFill, fillAddress, fillBytes, pattern] {
            GuestMemory::SetCurrentPacket(0x50, queue);
            if (gds) {
                GuestMemory::Write(address, bytes, 1);
                return;
            }
            if (metadataFill) {
                Graphics::NoteDepthMetadataFill(fillAddress, fillBytes, pattern);
                Graphics::NoteColorMetadataFill(fillAddress, fillBytes, pattern);
            }
            commitStore(queue, address, bytes);
        }, {{address, address + size}});
        return true;
    }
    if (Pm4::DmaSource(packet) == 0 || Pm4::DmaSource(packet) == 3) Graphics::RequireNoMultisampleTarget(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), packet[6] & 0x3ffffffu, "a DMA copy");
    const auto copy = Pm4::DecodeMemoryCopy(packet);
    if (!copy.has_value() || copy->destination % 4 != 0 || copy->bytes % 4 != 0 || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->source), copy->bytes) || !GuestMemory::Accessible(reinterpret_cast<const void*>(copy->destination), copy->bytes, true)) return false;
    DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::DmaCopy, [this, queue, copy = *copy] {
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
                if (outcome.path == 2 && !Graphics::RegisteredReadableCovers(copy.destination, copy.bytes)) {
                    auto* recorder = Graphics::Recorder::Active();
                    if (recorder != nullptr && localDevice->StoresPendingOver(copy.destination, copy.bytes)) recorder->SyncThrough(copy.destination, copy.bytes);
                    if (!localDevice->StoresPendingOver(copy.destination, copy.bytes)) ++copiesToHostMemory;
                }
            }
        }
        std::vector<std::byte> bytes(copy.bytes);
        GuestMemory::Read(copy.source, bytes, 1);
        commitStore(queue, copy.destination, bytes);
    }, {{copy->destination, copy->destination + copy->bytes}}, 0, {}, {{copy->source, copy->source + copy->bytes}});
    return true;
}

bool Driver::enqueueConstantRamDump(std::span<const std::uint32_t> packet, std::uint32_t opcode, std::uint32_t queue, const QueueState& state) {
    static const bool enabled = std::getenv("APS5_PIPELINE_DRAIN_DUMPS") == nullptr && std::getenv("APS5_CPU_STORES") == nullptr;
    if (!enabled || opcode != 0x83 || queue != 0 || !DrawPipeline::Active() || !deferredLabels().labels.empty()) return false;
    constexpr std::size_t gpuStoreLimit = 65536;
    const auto store = Pm4::ResolveStore(packet, state, gpuStoreLimit);
    if (!store.has_value()) return false;
    const auto stored = store->Bytes();
    if (stored.empty()) return true;
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(store->address), stored.size(), true)) return false;
    const auto address = store->address;
    std::vector<std::byte> bytes(stored.begin(), stored.end());
    auto queued = bytes;
    DrawPipeline::Queue0().EnqueueStore([this, queue, address, bytes = std::move(bytes)] {
        GuestMemory::SetCurrentPacket(0x83, queue);
        commitStore(queue, address, bytes);
    }, address, std::move(queued));
    return true;
}

void Driver::commitStore(std::uint32_t queue, std::uint64_t address, std::span<const std::byte> bytes) {
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

namespace {

constexpr std::size_t LabelRunLabels = 32;
constexpr std::size_t LabelRunBytes = 256;

struct RunLabel {
    std::uint64_t address = 0;
    std::array<std::byte, 8> bytes{};
    std::uint32_t size = 0;
    std::uint32_t opcode = 0;
};

struct LabelRun {
    std::vector<RunLabel> labels;
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint32_t queue = 0;
    std::function<void(LabelRun&)> flush;
};

LabelRun& labelRun() {
    static thread_local LabelRun* run = nullptr;
    return ShaderRecompiler::ThreadOwned(run);
}

std::atomic<std::uint64_t> labelRuns{0}, labelsInRuns{0}, labelsAlone{0};

}

bool LabelsInBatch() {
    static const bool enabled = std::getenv("APS5_LABELS_IN_BATCH") != nullptr && CoreOffDraw();
    return enabled;
}

void FlushLabelRun() {
    if (!LabelsInBatch()) return;
    auto& run = labelRun();
    if (run.labels.empty()) return;
    run.flush(run);
    run.labels.clear();
}

LabelRunCount LabelRunCounts() {
    return LabelRunCount{labelRuns.load(std::memory_order_relaxed), labelsInRuns.load(std::memory_order_relaxed), labelsAlone.load(std::memory_order_relaxed)};
}

bool Driver::labelPacketQueues(std::span<const std::uint32_t> packet, std::uint32_t opcode) {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_LABELS") == nullptr;
    if (!ordered || !DeferLabels() || (opcode != 0x49 && opcode != 0x37) || !deferredLabels().labels.empty()) return false;
    if (Pm4::DecodeLabelWrite(packet).has_value()) return true;
    return opcode == 0x49 && ((packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0);
}

bool Driver::enqueueLabelPacket(std::span<const std::uint32_t> packet, std::uint32_t opcode, std::uint32_t queue, std::optional<std::uint32_t> interrupt) {
    if (!labelPacketQueues(packet, opcode)) {
        FlushLabelRun();
        return false;
    }
    const bool endOfPipeInterrupt = interrupt.has_value();
    const auto label = Pm4::DecodeLabelWrite(packet);
    if (!label.has_value()) {
        if (opcode != 0x49) {
            FlushLabelRun();
            return false;
        }
        const bool storesNothing = (packet[2] >> 29u) == 0 || (packet[3] | (static_cast<std::uint64_t>(packet[4]) << 32u)) == 0;
        if (!storesNothing) {
            FlushLabelRun();
            return false;
        }
        if (!endOfPipeInterrupt) {
            ++noOpLabels;
            return true;
        }
    }
    if (LabelsInBatch()) {
        auto& run = labelRun();
        const auto stored = label.has_value() ? label->Bytes() : std::span<const std::byte>{};
        const bool groups = !endOfPipeInterrupt && label.has_value() && !stored.empty() && stored.size() <= 8;
        const auto labelEnd = label.has_value() ? label->address + stored.size() : 0;
        const bool joins = groups && !run.labels.empty() && run.queue == queue && run.labels.size() < LabelRunLabels && label->address <= run.end && labelEnd >= run.begin
            && std::max(run.end, labelEnd) - std::min(run.begin, label->address) <= LabelRunBytes;
        if (!joins) FlushLabelRun();
        if (groups) {
            if (run.labels.empty()) {
                run.begin = label->address;
                run.end = labelEnd;
                run.queue = queue;
                run.flush = [this](LabelRun& flushed) {
                    const auto flushedQueue = flushed.queue;
                    const auto begin = flushed.begin;
                    std::vector<std::byte> merged(static_cast<std::size_t>(flushed.end - flushed.begin));
                    std::vector<DrawPipeline::Range> ranges;
                    ranges.reserve(flushed.labels.size());
                    for (const auto& item : flushed.labels) {
                        std::memcpy(merged.data() + (item.address - begin), item.bytes.data(), item.size);
                        ranges.emplace_back(item.address, item.address + item.size);
                    }
                    labelRuns.fetch_add(1, std::memory_order_relaxed);
                    (flushed.labels.size() > 1 ? labelsInRuns : labelsAlone).fetch_add(flushed.labels.size(), std::memory_order_relaxed);
                    auto relayedRanges = ranges;
                    auto relayedBytes = merged;
                    auto enqueueRun = [this, flushedQueue, begin, labels = flushed.labels, ranges = std::move(ranges), merged = std::move(merged)]() mutable {
                        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::EndOfPipe, [this, flushedQueue, labels = std::move(labels)] {
                            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
                            std::lock_guard gpuLock(GuestMemory::GpuMutex());
                            const auto localDevice = device.Load();
                            recordDeferredLabels(localDevice.get(), flushedQueue);
                            bool first = true;
                            for (const auto& item : labels) {
                                GuestMemory::SetCurrentPacket(item.opcode, flushedQueue);
                                const auto bytes = std::span<const std::byte>(item.bytes).first(item.size);
                                const auto stamp = ++eventSerial;
                                const int reason = localDevice != nullptr ? localDevice->WriteLabelOnGpu(item.address, bytes, stamp, flushedQueue, first) : 4;
                                first = false;
                                if (reason != 0 && reason != 5 && reason != 6) {
                                    if (reason != 1 && localDevice != nullptr) localDevice->WaitIdle();
                                    GuestMemory::Write(item.address, bytes, 4);
                                }
                                noteLabelStore(item.address, bytes, stamp, flushedQueue);
                                countLabelOutcome(reason);
                                ++immediateLabels;
                            }
                            Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
                            submitDueAfterCommit(localDevice.get());
                        }, std::move(ranges), begin, std::move(merged));
                    };
                    if (flushedQueue == 0 && DrawFinisher::Overlap() && DrawFinisher::Busy()) {
                        DrawFinisher::Pending held;
                        held.unknown = false;
                        held.writes = std::move(relayedRanges);
                        held.labelAddress = begin;
                        held.labelBytes = std::move(relayedBytes);
                        DrawFinisher::Get().Enqueue(std::move(enqueueRun), std::move(held));
                        return;
                    }
                    enqueueRun();
                };
            } else {
                run.begin = std::min(run.begin, label->address);
                run.end = std::max(run.end, labelEnd);
            }
            RunLabel item;
            item.address = label->address;
            std::memcpy(item.bytes.data(), stored.data(), stored.size());
            item.size = static_cast<std::uint32_t>(stored.size());
            item.opcode = opcode;
            run.labels.push_back(item);
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
    if (endOfPipeInterrupt) {
        bumpEpoch(&EpochBumps::queuedInterrupts);
        ++DrainEpochCounts().queuedInterrupts;
    }
    auto stored = bytes;
    auto labelBytes = bytes;
    auto enqueue =[this, queue, opcode, address, bytes = std::move(bytes), interrupt, writes, stored = std::move(stored)]() mutable {
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::EndOfPipe, [this, queue, opcode, address, bytes = std::move(bytes), interrupt] {
            GuestMemory::SetCurrentPacket(opcode, queue);
            commitLabel(queue, address, bytes, interrupt);
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

namespace {

constexpr std::uint64_t EopProbePage = 4096;
constexpr std::size_t EopProbeReadLimit = std::size_t{8} << 20u;
constexpr std::size_t EopProbeBudget = std::size_t{8} << 20u;
constexpr std::uint64_t EopProbeHistory = 4;
constexpr std::size_t EopProbeKindQueries = 32;
constexpr std::size_t EopProbeKinds = static_cast<std::size_t>(Graphics::Recorder::ReadKind::Count);
constexpr std::array<const char*, EopProbeKinds + 1> EopProbeKindNames{"dispatch-element", "gpu-copy", "address-based", "indirect", "storage-upload", "copy-source", "draw-input", "address-based-read-only", "unknown"};

using EopProbeRange = std::pair<std::uint64_t, std::uint64_t>;

constexpr std::size_t EopGuardWhole = 4096;
constexpr std::size_t EopGuardEdge = 256;
constexpr std::size_t EopGuardWindow = 64;
constexpr std::size_t EopGuardWindows = 14;
constexpr std::uint64_t EopGuardQuietLimit = 256;
constexpr std::uint64_t EopGuardKept = 64;

struct EopGuardKey {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    bool operator==(const EopGuardKey&) const = default;
};

struct EopGuardKeyHash {
    std::size_t operator()(const EopGuardKey& key) const {
        return static_cast<std::size_t>((key.begin * 0x9e3779b97f4a7c15ull) ^ (key.end * 0xc2b2ae3d27d4eb4full));
    }
};

struct EopGuardNote {
    std::uint64_t print = 0;
    std::uint64_t seen = 0;
    std::uint64_t quietFrom = 0;
    std::uint32_t changes = 0;
    bool printed = false;
};

struct EopGuardCounts {
    std::uint64_t labels = 0;
    std::uint64_t early = 0;
    std::uint64_t heldOpen = 0;
    std::uint64_t heldBehind = 0;
    std::uint64_t heldAddress = 0;
    std::uint64_t heldUnknown = 0;
    std::uint64_t judged = 0;
    std::uint64_t risky = 0;
    std::uint64_t gpuWritten = 0;
    std::uint64_t published = 0;
    std::array<std::uint64_t, EopProbeKinds> riskyKinds{};
    std::uint64_t lastBegin = 0;
    std::uint64_t lastEnd = 0;
    double guardMs = 0;
};

struct EopGuard {
    std::unordered_map<EopGuardKey, EopGuardNote, EopGuardKeyHash> notes;
    std::vector<Graphics::Recorder::InPlaceRead> finished;
    std::vector<Graphics::Recorder::InPlaceRead> pending;
    std::vector<EopProbeRange> writes;
    std::vector<EopProbeRange> finishedWrites;
    std::uint64_t label = 0;
    EopGuardCounts counts;
};

EopGuard& eopGuard() {
    static EopGuard guard;
    return guard;
}

std::uint64_t eopGuardMix(std::uint64_t hash, const std::byte* data, std::size_t bytes) {
    std::size_t offset = 0;
    for (; offset + 8 <= bytes; offset += 8) {
        std::uint64_t word;
        std::memcpy(&word, data + offset, 8);
        hash = (hash ^ word) * 0x9e3779b97f4a7c15ull;
        hash ^= hash >> 32u;
    }
    if (offset < bytes) {
        std::uint64_t word = 0;
        std::memcpy(&word, data + offset, bytes - offset);
        hash = (hash ^ word) * 0x9e3779b97f4a7c15ull;
        hash ^= hash >> 32u;
    }
    return hash;
}

std::uint64_t eopGuardPrint(std::uint64_t begin, std::size_t bytes) {
    const auto* data = reinterpret_cast<const std::byte*>(begin);
    if (bytes <= EopGuardWhole) return eopGuardMix(bytes, data, bytes);
    auto hash = eopGuardMix(bytes, data, EopGuardEdge);
    const auto step = (bytes - EopGuardWindow) / (EopGuardWindows + 1);
    for (std::size_t window = 1; window <= EopGuardWindows; ++window) hash = eopGuardMix(hash, data + step * window, EopGuardWindow);
    return eopGuardMix(hash, data + bytes - EopGuardEdge, EopGuardEdge);
}

bool eopGuardRisky(EopGuard& guard, Graphics::Recorder& recorder, const Graphics::Recorder::InPlaceRead& read) {
    auto& note = guard.notes[{read.begin, read.end}];
    if (note.seen != guard.label) {
        const bool consecutive = note.seen + 1 == guard.label;
        note.seen = guard.label;
        ++guard.counts.judged;
        const auto bytes = static_cast<std::size_t>(read.end - read.begin);
        const auto after = std::upper_bound(guard.writes.begin(), guard.writes.end(), read.begin, [](std::uint64_t address, const EopProbeRange& write) { return address < write.second; });
        if (after != guard.writes.end() && after->first < read.end) {
            note.printed = false;
            ++guard.counts.gpuWritten;
        } else if (!GuestMemory::Accessible(reinterpret_cast<const void*>(read.begin), bytes)) {
            note.printed = false;
            note.quietFrom = guard.label + 1;
        } else {
            const auto print = eopGuardPrint(read.begin, bytes);
            if (note.printed && (consecutive || note.changes != 0) && print != note.print) {
                note.changes = std::min<std::uint32_t>(note.changes + 1, 6);
                note.quietFrom = guard.label + std::min<std::uint64_t>(EopGuardQuietLimit, std::uint64_t{4} << note.changes);
                recorder.NoteChangingRead(read.begin, read.end);
                ++guard.counts.published;
            } else if (!note.printed || !consecutive) {
                note.quietFrom = std::max(note.quietFrom, guard.label + 2);
            }
            note.print = print;
            note.printed = true;
        }
    }
    return guard.label < note.quietFrom;
}

bool eopGuardSettled() {
    auto& guard = eopGuard();
    auto& counts = guard.counts;
    const auto start = std::chrono::steady_clock::now();
    ++counts.labels;
    if (++guard.label == 1) guard.notes.reserve(std::size_t{1} << 15u);
    auto* recorder = Graphics::Recorder::Active();
    if (recorder == nullptr) {
        ++counts.early;
        return true;
    }
    const bool lost = recorder->TakeFinishedReads(guard.finished, guard.finishedWrites);
    const auto opened = recorder->UnexecutedReads(guard.pending);
    recorder->PendingWriteRanges(guard.writes);
    guard.writes.insert(guard.writes.end(), guard.finishedWrites.begin(), guard.finishedWrites.end());
    std::sort(guard.writes.begin(), guard.writes.end());
    std::size_t merged = 0;
    for (const auto& write : guard.writes) {
        if (merged != 0 && write.first <= guard.writes[merged - 1].second) guard.writes[merged - 1].second = std::max(guard.writes[merged - 1].second, write.second);
        else guard.writes[merged++] = write;
    }
    guard.writes.resize(merged);
    for (const auto& read : guard.finished) static_cast<void>(eopGuardRisky(guard, *recorder, read));
    bool address = false, open = false, flight = false;
    for (std::size_t index = 0; index < guard.pending.size(); ++index) {
        const auto& read = guard.pending[index];
        if (read.kind == Graphics::Recorder::ReadKind::AddressBased || read.kind == Graphics::Recorder::ReadKind::AddressBasedReadOnly) {
            address = true;
            continue;
        }
        if (!eopGuardRisky(guard, *recorder, read)) continue;
        ++counts.risky;
        ++counts.riskyKinds[static_cast<std::size_t>(read.kind)];
        counts.lastBegin = read.begin;
        counts.lastEnd = read.end;
        (index < opened ? open : flight) = true;
    }
    if (guard.label % EopGuardKept == 0) std::erase_if(guard.notes, [&](const auto& note) { return note.second.seen + EopGuardKept < guard.label; });
    const auto judged = std::chrono::steady_clock::now();
    counts.guardMs += std::chrono::duration<double, std::milli>(judged - start).count();
    if (lost || !Graphics::Recorder::ReadTracking()) {
        ++counts.heldUnknown;
        return false;
    }
    if (address) {
        ++counts.heldAddress;
        return false;
    }
    if (open) {
        ++counts.heldOpen;
        return false;
    }
    if (flight) {
        ++counts.heldBehind;
        return false;
    }
    ++counts.early;
    return true;
}

struct EopProbeRun {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::size_t offset = 0;
    std::size_t kind = EopProbeKinds;
};

struct EopProbeFrame {
    std::uint64_t done = 0;
    std::uint64_t last = 0;
    std::uint64_t guest = 0;
    bool early = false;
    std::chrono::steady_clock::time_point at;
    std::vector<EopProbeRun> runs;
    std::vector<std::byte> bytes;
    std::vector<EopProbeRun> smalls;
    std::vector<std::byte> smallBytes;
    std::vector<EopProbeRange> writes;
    std::shared_ptr<EopProbeFrame> previous;
};

struct EopProbePageNote {
    std::uint64_t frame = 0;
    std::size_t kind = EopProbeKinds;
};

struct EopProbeExample {
    std::uint64_t count = 0;
    std::uint64_t changed = 0;
    std::uint64_t first = 0;
    std::uint64_t behind = 0;
    std::size_t kind = 0;
};

struct EopProbeCounts {
    std::uint64_t labels = 0;
    std::uint64_t ended = 0;
    std::uint64_t batches = 0;
    std::uint64_t missing = 0;
    std::uint64_t writes = 0;
    std::uint64_t noted = 0;
    std::uint64_t notedBytes = 0;
    std::uint64_t large = 0;
    std::uint64_t uncovered = 0;
    std::uint64_t inaccessible = 0;
    std::uint64_t overBudget = 0;
    std::uint64_t checkedBytes = 0;
    std::uint64_t rewrittenBytes = 0;
    std::uint64_t shotBytes = 0;
    std::uint64_t shotRuns = 0;
    std::uint64_t stamped = 0;
    std::uint64_t noReads = 0;
    std::uint64_t noMoved = 0;
    std::uint64_t noRewritten = 0;
    std::uint64_t noGuest = 0;
    std::uint64_t earlyLabels = 0;
    std::uint64_t earlyChecked = 0;
    std::uint64_t earlyRewritten = 0;
    std::uint64_t smallChecked = 0;
    std::uint64_t smallRewritten = 0;
    std::array<std::uint64_t, EopProbeKinds + 1> checked{};
    std::array<std::uint64_t, EopProbeKinds + 1> rewritten{};
    std::array<std::uint64_t, EopProbeKinds + 1> aged{};
    std::array<std::uint64_t, EopProbeKinds + 1> moved{};
    std::array<std::uint64_t, EopProbeKinds + 1> gpuWritten{};
    double shotMs = 0;
    double tailMs = 0;
    double stampedTailMs = 0;
    double leftReads = 0;
    double leftMoved = 0;
    double leftRewritten = 0;
    double leftGuest = 0;
};

struct EopProbe {
    std::mutex mutex;
    std::map<std::uint64_t, EopProbePageNote> pages;
    std::vector<EopProbeRun> plan;
    std::size_t planBytes = 0;
    std::uint64_t cursor = 0;
    std::uint64_t frame = 0;
    std::shared_ptr<EopProbeFrame> previous;
    std::vector<std::vector<std::byte>> spare;
    std::vector<Graphics::Recorder::InPlaceRead> pending;
    std::map<EopProbeRange, EopProbeExample> examples;
    EopProbeCounts counts;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

EopProbe& eopProbe() {
    static EopProbe probe;
    return probe;
}

const std::byte* eopProbeBytes(const EopProbeFrame& frame, std::uint64_t begin, std::uint64_t end, std::size_t* kind) {
    const auto found = std::upper_bound(frame.runs.begin(), frame.runs.end(), begin, [](std::uint64_t address, const EopProbeRun& run) { return address < run.begin; });
    if (found == frame.runs.begin()) return nullptr;
    const auto& run = *std::prev(found);
    if (end > run.end) return nullptr;
    if (kind != nullptr) *kind = run.kind;
    return frame.bytes.data() + run.offset + (begin - run.begin);
}

void reportEopProbe(EopProbe& probe) {
    const auto& counts = probe.counts;
    std::string kinds;
    std::uint64_t checked = 0, rewritten = 0, aged = 0, moved = 0, gpuWritten = 0;
    for (std::size_t kind = 0; kind <= EopProbeKinds; ++kind) {
        checked += counts.checked[kind];
        rewritten += counts.rewritten[kind];
        aged += counts.aged[kind];
        moved += counts.moved[kind];
        gpuWritten += counts.gpuWritten[kind];
        if (counts.checked[kind] == 0 && counts.aged[kind] == 0 && counts.gpuWritten[kind] == 0) continue;
        kinds += " " + std::string(EopProbeKindNames[kind]) + " " + std::to_string(counts.rewritten[kind]) + "/" + std::to_string(counts.checked[kind]) + " (" + std::to_string(counts.moved[kind]) + "/" + std::to_string(counts.aged[kind]) + " changed since the previous label, " + std::to_string(counts.gpuWritten[kind]) + " under a recorded GPU write)";
    }
    std::string examples;
    std::vector<std::pair<EopProbeRange, EopProbeExample>> sorted(probe.examples.begin(), probe.examples.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
    for (std::size_t index = 0; index < sorted.size() && index < 12; ++index) {
        char text[192];
        std::snprintf(text, sizeof(text), " %llx+%llu %s x%llu (%llu bytes differ from offset %llu, %llu batches before the newest)", static_cast<unsigned long long>(sorted[index].first.first), static_cast<unsigned long long>(sorted[index].first.second - sorted[index].first.first), EopProbeKindNames[sorted[index].second.kind], static_cast<unsigned long long>(sorted[index].second.count), static_cast<unsigned long long>(sorted[index].second.changed), static_cast<unsigned long long>(sorted[index].second.first), static_cast<unsigned long long>(sorted[index].second.behind));
        examples += text;
    }
    const double perLabel = counts.labels != 0 ? 1.0 / static_cast<double>(counts.labels) : 0.0;
    const double perEnd = counts.ended != 0 ? 1.0 / static_cast<double>(counts.ended) : 0.0;
    const double perStamped = counts.stamped != 0 ? 1.0 / static_cast<double>(counts.stamped) : 0.0;
    const auto value = [](std::uint64_t count) { return static_cast<unsigned long long>(count); };
    AgcDriver::ProfilePrint_nid_no_patch("[eop] %llu end-of-pipe interrupt labels over 10 s; unfinished at the label: %.1f batches (%llu without a completion record), %.1f recorded GPU write ranges, %.1f in-place reads (%.0f KiB); in-place reads compared by bytes with their copy made at the label once the recorded work finished, %.2f ms later: %llu rewritten of %llu (%llu of %llu bytes), of which %llu of %llu at the %llu labels whose interrupt was delivered at the label, and %llu of %llu on reads of at most 4 KiB copied exactly; by reader, rewritten/compared:%s; not compared: %llu outside the pages copied, %llu under a recorded GPU write, %llu over 8 MiB, %llu inaccessible, %llu followed pages left out of a copy by its budget; changed since the previous label: %llu of %llu; GPU time left of the %.2f ms after the label once the newest unfinished batch has finished that reads in place %.2f ms (%llu labels with none), that reads a range changed since the previous label %.2f ms (%llu with none), that reads a rewritten range %.2f ms (%llu with none), that touches guest memory %.2f ms (%llu with none), over %llu labels with stamped batches; copy at the label: %.1f runs, %.0f KiB, %.3f ms; rewritten ranges:%s\n", value(counts.labels), static_cast<double>(counts.batches) * perEnd, value(counts.missing), static_cast<double>(counts.writes) * perLabel, static_cast<double>(counts.noted) * perEnd, static_cast<double>(counts.notedBytes) * perEnd / 1024.0, counts.tailMs * perEnd, value(rewritten), value(checked), value(counts.rewrittenBytes), value(counts.checkedBytes), value(counts.earlyRewritten), value(counts.earlyChecked), value(counts.earlyLabels), value(counts.smallRewritten), value(counts.smallChecked), kinds.c_str(), value(counts.uncovered), value(gpuWritten), value(counts.large), value(counts.inaccessible), value(counts.overBudget), value(moved), value(aged), counts.stampedTailMs * perStamped, counts.leftReads * perStamped, value(counts.noReads), counts.leftMoved * perStamped, value(counts.noMoved), counts.leftRewritten * perStamped, value(counts.noRewritten), counts.leftGuest * perStamped, value(counts.noGuest), value(counts.stamped), static_cast<double>(counts.shotRuns) * perLabel, static_cast<double>(counts.shotBytes) * perLabel / 1024.0, counts.shotMs * perLabel, examples.empty() ? " none" : examples.c_str());
    auto& guard = eopGuard().counts;
    const double perGuard = guard.labels != 0 ? 1.0 / static_cast<double>(guard.labels) : 0.0;
    std::string changed;
    for (std::size_t kind = 0; kind < EopProbeKinds; ++kind) {
        if (guard.riskyKinds[kind] != 0) changed += (changed.empty() ? "" : ", ") + std::string(EopProbeKindNames[kind]) + " " + std::to_string(guard.riskyKinds[kind]);
    }
    if (changed.empty()) changed = "none";
    AgcDriver::ProfilePrint_nid_no_patch("[eop-guard] %llu end-of-pipe interrupt labels: %llu delivered at the label, held behind the recorded work: %llu for a changed or new range read by the open batch, %llu for one read by a batch in flight, %llu for an address-based read, %llu without the reads of the frame; per label %.1f in-place reads judged, %.1f of them under a GPU write of the frame, %.2f unexecuted on a changed or new range (%s; last %llx+%llu), %llu changes published to the draw path, %.3f ms\n", value(guard.labels), value(guard.early), value(guard.heldOpen), value(guard.heldBehind), value(guard.heldAddress), value(guard.heldUnknown), static_cast<double>(guard.judged) * perGuard, static_cast<double>(guard.gpuWritten) * perGuard, static_cast<double>(guard.risky) * perGuard, changed.c_str(), value(guard.lastBegin), value(guard.lastEnd - guard.lastBegin), value(guard.published), guard.guardMs * perGuard);
    guard = {};
    probe.examples.clear();
    probe.counts = {};
}

std::shared_ptr<EopProbeFrame> beginEopProbe(bool early) {
    auto* recorder = Graphics::Recorder::Active();
    if (recorder == nullptr) return nullptr;
    auto& probe = eopProbe();
    std::lock_guard lock(probe.mutex);
    const auto start = std::chrono::steady_clock::now();
    if (start - probe.lastReport > std::chrono::seconds(10)) {
        probe.lastReport = start;
        reportEopProbe(probe);
    }
    auto frame = std::make_shared<EopProbeFrame>();
    frame->early = early;
    if (early) ++probe.counts.earlyLabels;
    const auto unsignaled = static_cast<std::uint64_t>(recorder->UnsignaledBatches());
    frame->done = std::max(recorder->CompletedThrough(), recorder->Submissions() - std::min(recorder->Submissions(), unsignaled));
    frame->last = recorder->Recording() ? recorder->OpenSerial() : recorder->Submissions();
    const auto guest = recorder->NewestGuestWork();
    frame->guest = guest.has_value() ? *guest : frame->last;
    recorder->PendingWriteRanges(frame->writes);
    static_cast<void>(recorder->UnexecutedReads(probe.pending));
    std::sort(probe.pending.begin(), probe.pending.end(), [](const auto& a, const auto& b) { return a.begin != b.begin ? a.begin < b.begin : a.end < b.end; });
    for (std::size_t index = 0; index < probe.pending.size(); ++index) {
        const auto& read = probe.pending[index];
        const auto bytes = static_cast<std::size_t>(read.end - read.begin);
        if (bytes > EopGuardWhole || (index != 0 && probe.pending[index - 1].begin == read.begin && probe.pending[index - 1].end == read.end)) continue;
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(read.begin), bytes)) {
            ++probe.counts.inaccessible;
            continue;
        }
        frame->smalls.push_back({read.begin, read.end, frame->smallBytes.size(), static_cast<std::size_t>(read.kind)});
        frame->smallBytes.insert(frame->smallBytes.end(), reinterpret_cast<const std::byte*>(read.begin), reinterpret_cast<const std::byte*>(read.end));
    }
    if (!probe.spare.empty()) {
        frame->bytes = std::move(probe.spare.back());
        probe.spare.pop_back();
    }
    if (frame->bytes.size() < probe.planBytes) frame->bytes.resize(probe.planBytes);
    frame->runs.reserve(probe.plan.size());
    std::size_t queries = 0;
    for (auto run : probe.plan) {
        const auto bytes = static_cast<std::size_t>(run.end - run.begin);
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(run.begin), bytes)) {
            ++probe.counts.inaccessible;
            continue;
        }
        if (run.kind == EopProbeKinds && queries < EopProbeKindQueries) {
            ++queries;
            if (const auto info = recorder->DescribePendingRead(run.begin, bytes)) run.kind = static_cast<std::size_t>(info->kind);
        }
        std::memcpy(frame->bytes.data() + run.offset, reinterpret_cast<const void*>(run.begin), bytes);
        frame->runs.push_back(run);
        probe.counts.shotBytes += bytes;
    }
    frame->previous = std::exchange(probe.previous, frame);
    ++probe.counts.labels;
    probe.counts.shotRuns += frame->runs.size();
    probe.counts.writes += frame->writes.size();
    frame->at = std::chrono::steady_clock::now();
    probe.counts.shotMs += std::chrono::duration<double, std::milli>(frame->at - start).count();
    return frame;
}

void endEopProbe(const std::shared_ptr<EopProbeFrame>& frame) {
    auto& probe = eopProbe();
    std::lock_guard lock(probe.mutex);
    auto& counts = probe.counts;
    const double tailMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frame->at).count();
    ++counts.ended;
    counts.tailMs += tailMs;
    for (const auto& run : frame->runs) {
        if (run.kind == EopProbeKinds) continue;
        for (auto page = probe.pages.lower_bound(run.begin / EopProbePage); page != probe.pages.end() && page->first < run.end / EopProbePage; ++page) {
            if (page->second.kind == EopProbeKinds) page->second.kind = run.kind;
        }
    }
    std::vector<Graphics::Recorder::Completed> tail;
    std::size_t missing = 0;
    auto* recorder = Graphics::Recorder::Active();
    if (recorder != nullptr && frame->last > frame->done) tail = recorder->CompletedBatches(frame->done, std::min(frame->last, frame->done + Graphics::Recorder::CompletedRingSize / 2), missing);
    counts.missing += missing;
    counts.batches += tail.size();
    const auto* previous = frame->previous.get();
    bool stamped = !tail.empty();
    double lastEnd = 0, readsEnd = 0, movedEnd = 0, rewrittenEnd = 0, guestEnd = 0;
    for (const auto& batch : tail) {
        if (batch.gpuEndNs <= batch.gpuStartNs) stamped = false;
        lastEnd = std::max(lastEnd, batch.gpuEndNs);
        if (batch.serial <= frame->guest) guestEnd = std::max(guestEnd, batch.gpuEndNs);
        bool moved = false, rewritten = false;
        for (const auto& [begin, end] : batch.reads) {
            const auto bytes = static_cast<std::size_t>(end - begin);
            ++counts.noted;
            counts.notedBytes += bytes;
            if (bytes > EopProbeReadLimit) {
                ++counts.large;
                continue;
            }
            if (bytes <= EopGuardWhole) continue;
            for (auto page = begin / EopProbePage; page <= (end - 1) / EopProbePage; ++page) probe.pages[page].frame = probe.frame;
            std::size_t kind = EopProbeKinds;
            const auto* shot = eopProbeBytes(*frame, begin, end, &kind);
            if (shot == nullptr) {
                ++counts.uncovered;
                continue;
            }
            if (std::any_of(frame->writes.begin(), frame->writes.end(), [&](const EopProbeRange& write) { return write.first < end && begin < write.second; })) {
                ++counts.gpuWritten[kind];
                continue;
            }
            if (const auto* before = previous != nullptr ? eopProbeBytes(*previous, begin, end, nullptr) : nullptr) {
                ++counts.aged[kind];
                if (std::memcmp(before, shot, bytes) != 0) {
                    ++counts.moved[kind];
                    moved = true;
                }
            }
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) {
                ++counts.inaccessible;
                continue;
            }
            const auto* current = reinterpret_cast<const std::byte*>(begin);
            ++counts.checked[kind];
            counts.checkedBytes += bytes;
            if (frame->early) ++counts.earlyChecked;
            if (std::memcmp(current, shot, bytes) == 0) continue;
            std::uint64_t changed = 0, first = bytes;
            for (std::size_t offset = 0; offset < bytes; ++offset) {
                if (current[offset] == shot[offset]) continue;
                if (changed == 0) first = offset;
                ++changed;
            }
            ++counts.rewritten[kind];
            counts.rewrittenBytes += changed;
            if (frame->early) ++counts.earlyRewritten;
            rewritten = true;
            auto& example = probe.examples[{begin, end}];
            ++example.count;
            example.changed = changed;
            example.first = first;
            example.behind = frame->last - batch.serial;
            example.kind = kind;
        }
        if (!batch.reads.empty()) readsEnd = std::max(readsEnd, batch.gpuEndNs);
        if (moved) movedEnd = std::max(movedEnd, batch.gpuEndNs);
        if (rewritten) rewrittenEnd = std::max(rewrittenEnd, batch.gpuEndNs);
    }
    for (const auto& small : frame->smalls) {
        const auto bytes = static_cast<std::size_t>(small.end - small.begin);
        if (std::any_of(frame->writes.begin(), frame->writes.end(), [&](const EopProbeRange& write) { return write.first < small.end && small.begin < write.second; })) {
            ++counts.gpuWritten[small.kind];
            continue;
        }
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(small.begin), bytes)) {
            ++counts.inaccessible;
            continue;
        }
        const auto* current = reinterpret_cast<const std::byte*>(small.begin);
        const auto* shot = frame->smallBytes.data() + small.offset;
        ++counts.checked[small.kind];
        ++counts.smallChecked;
        counts.checkedBytes += bytes;
        if (frame->early) ++counts.earlyChecked;
        if (std::memcmp(current, shot, bytes) == 0) continue;
        std::uint64_t changed = 0, first = bytes;
        for (std::size_t offset = 0; offset < bytes; ++offset) {
            if (current[offset] == shot[offset]) continue;
            if (changed == 0) first = offset;
            ++changed;
        }
        ++counts.rewritten[small.kind];
        ++counts.smallRewritten;
        counts.rewrittenBytes += changed;
        if (frame->early) ++counts.earlyRewritten;
        auto& example = probe.examples[{small.begin, small.end}];
        ++example.count;
        example.changed = changed;
        example.first = first;
        example.behind = 0;
        example.kind = small.kind;
    }
    if (stamped) {
        const auto left = [&](double end) { return end > 0 ? std::min(tailMs, (lastEnd - end) / 1e6) : tailMs; };
        ++counts.stamped;
        counts.stampedTailMs += tailMs;
        counts.leftReads += left(readsEnd);
        counts.leftMoved += left(movedEnd);
        counts.leftRewritten += left(rewrittenEnd);
        counts.leftGuest += left(guestEnd);
        if (readsEnd == 0) ++counts.noReads;
        if (movedEnd == 0) ++counts.noMoved;
        if (rewrittenEnd == 0) ++counts.noRewritten;
        if (guestEnd == 0) ++counts.noGuest;
    }
    if (auto older = std::move(frame->previous); older != nullptr && older.use_count() == 1 && probe.spare.size() < 4) probe.spare.push_back(std::move(older->bytes));
    ++probe.frame;
    std::erase_if(probe.pages, [&](const auto& page) { return page.second.frame + EopProbeHistory < probe.frame; });
    probe.plan.clear();
    probe.planBytes = 0;
    const std::size_t room = EopProbeBudget / EopProbePage;
    const std::size_t taken = std::min(room, probe.pages.size());
    counts.overBudget += probe.pages.size() - taken;
    std::vector<std::pair<std::uint64_t, std::size_t>> chosen;
    chosen.reserve(taken);
    auto next = probe.pages.lower_bound(probe.cursor);
    for (std::size_t index = 0; index < taken; ++index) {
        if (next == probe.pages.end()) next = probe.pages.begin();
        if (index == taken / EopProbeHistory) probe.cursor = taken < probe.pages.size() ? next->first : 0;
        chosen.emplace_back(next->first, next->second.kind);
        ++next;
    }
    std::sort(chosen.begin(), chosen.end());
    for (const auto& [page, kind] : chosen) {
        const auto begin = page * EopProbePage;
        if (!probe.plan.empty() && probe.plan.back().end == begin) {
            probe.plan.back().end += EopProbePage;
            if (probe.plan.back().kind == EopProbeKinds) probe.plan.back().kind = kind;
        } else {
            probe.plan.push_back({begin, begin + EopProbePage, probe.planBytes, kind});
        }
        probe.planBytes += EopProbePage;
    }
}

}

void Driver::commitLabel(std::uint32_t queue, std::uint64_t address, std::span<const std::byte> bytes, std::optional<std::uint32_t> interrupt) {
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Label);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    const auto localDevice = device.Load();
    recordDeferredLabels(localDevice.get(), queue);
    static const bool probed = Graphics::Recorder::FlipReadCheck();
    static const bool earlyInterrupt = std::getenv("APS5_EARLY_EOP_INTERRUPT") != nullptr;
    const bool early = earlyInterrupt && interrupt.has_value() && eopGuardSettled();
    bool workOpen = Graphics::Recorder::RecordedWorkSinceSubmit() != 0;
    int reason = localDevice != nullptr ? 0 : 1;
    if (!bytes.empty()) {
        const auto stamp = ++eventSerial;
        if (localDevice != nullptr && queue == 0 && Graphics::StorageTexture::StoreDrawnLinear("label") != 0) workOpen = true;
        reason = early ? 4 : localDevice != nullptr ? localDevice->WriteLabelOnGpu(address, bytes, stamp, queue) : 4;
        if (reason != 0 && reason != 5 && reason != 6) {
            if (!early && reason != 1 && localDevice != nullptr) localDevice->WaitIdle();
            GuestMemory::Write(address, bytes, 4);
        }
        noteLabelStore(address, bytes, stamp, queue);
        Graphics::Recorder::CloseLabelGroup(GuestMemory::TrackerGeneration());
        countLabelOutcome(reason);
        ++immediateLabels;
    } else {
        ++noOpLabels;
    }
    if (interrupt.has_value()) {
        const auto context = *interrupt;
        bool deferred = false;
        if (bytes.empty() && localDevice != nullptr && queue == 0 && Graphics::StorageTexture::StoreDrawnLinear("interrupt") != 0) workOpen = true;
        if (probed && queue == 0 && localDevice != nullptr) {
            if (const auto frame = beginEopProbe(early); frame != nullptr && !localDevice->AfterRecordedWork([frame] { endEopProbe(frame); }, false)) endEopProbe(frame);
        }
        if (!early && localDevice != nullptr && (reason == 0 || reason == 5 || reason == 6)) {
            deferred = localDevice->AfterRecordedWork([this, queue, context] { deliverGuestEopInterrupt(queue, context); }, queue == 0);
            if (deferred && workOpen) localDevice->SubmitRecorded(queue == 0);
        }
        if (auto* recorder = earlyInterrupt ? Graphics::Recorder::Active() : nullptr) recorder->NoteEndOfPipe(early);
        if (!deferred) deliverGuestEopInterrupt(queue, context);
    }
    submitDueAfterCommit(localDevice.get());
}

void Driver::submitDueAfterCommit(VulkanDevice* localDevice) {
    if (localDevice == nullptr) return;
    const auto pending = Graphics::Recorder::PendingLabelSince();
    const bool due = pending.has_value() && std::chrono::steady_clock::now() - *pending >= labelFlushDeadline();
    const bool capped = batchCap() != 0 && Graphics::Recorder::RecordedWorkSinceSubmit() >= batchCap();
    if (due || capped) {
        localDevice->SubmitRecorded(true);
        return;
    }
    if (!SubmitEarly()) return;
    Core::SubmitState state;
    state.gpuLoad = localDevice->OpenGpuLoad();
    Core::SubmitLimits limits;
    limits.idleLoad = SubmitEarlyLoad();
    if (state.gpuLoad < limits.idleLoad) return;
    state.queued = 1;
    state.gpuIdle = localDevice->GpuIdle();
    if (Core::AfterOp(state, limits) != Core::SubmitCause::Idle) return;
    localDevice->SubmitRecorded(false);
    EarlySubmits().fetch_add(1, std::memory_order_relaxed);
}

}
