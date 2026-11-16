#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Core/include/CoarseClock.hpp"
#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Core/include/DecoderRing.hpp"
#include "prx/libSceAgcDriver/Core/include/DecoderShadow.hpp"
#include "ThreadOwned.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadScratch.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferCache.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {


struct RingOrder {
    std::uint32_t queue = 0xffffffffu;
    std::uint64_t serial = 0;
    bool afterDispatch = false;
};

thread_local RingOrder ringOrder;

void noteRingPacket(std::uint32_t queueId, std::uint64_t serial, std::uint32_t opcode, std::uint32_t header) {
    if (ringOrder.queue != queueId || ringOrder.serial != serial) ringOrder = {queueId, serial, false};
    if (opcode == 0x15 || opcode == 0x16 || opcode == 0x76) return;
    if (opcode == 0x10) {
        const auto sub = (header >> 2u) & 0x3fu;
        if ((header & 0xfcu) == 0 || sub == 0x0b || sub == 0x0c) return;
    }
    ringOrder.afterDispatch = false;
}

bool orderedFills() {
    static const bool ordered = std::getenv("APS5_PIPELINE_DRAIN_FILLS") == nullptr;
    return ordered;
}

bool keepsDrawFinisher(std::uint32_t header, std::uint32_t opcode) {
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return false;
    if (Pm4::DrawOpcode(opcode)) return true;
    switch (opcode) {
        case 0x13: case 0x26: case 0x2a: case 0x2f: case 0x69: case 0x76: case 0x79: return true;
        case 0x10: return ((header >> 2u) & 0x3fu) == 0;
        default: return false;
    }
}

bool waitTakesPendingLabel(std::span<const std::uint32_t> packet) {
    if (packet.size() < 5) return false;
    const auto awaitedBytes = Pm4::WaitAwaitedBytes(packet);
    const std::uint64_t awaited = (packet[2] & (awaitedBytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
    const auto value = DrawPipeline::Queue0().PendingLabel(awaited, awaitedBytes);
    return value.has_value() && Pm4::WaitComparesValue(packet, *value);
}

bool keepsDrawPipeline(std::span<const std::uint32_t> packet, std::uint32_t header, std::uint32_t opcode) {
    if (header == FlipPacketHeader || header == RenderingWaitPacketHeader) return false;
    if (Pm4::DrawOpcode(opcode)) return true;
    switch (opcode) {
        case 0x11: case 0x12: case 0x13: case 0x20: case 0x26: case 0x2a: case 0x2f: case 0x3f: case 0x42: case 0x58: case 0x59: case 0x69: case 0x76: case 0x79: case 0x7a: case 0x81: return true;
        case 0x46: return packet.size() >= 2 && (packet[1] & 0x3fu) != 0x39u;
        case 0x15: return orderedFills();
        case 0x40:
            return packet.size() >= 6 && !Pm4::CopyDataToRegister(packet) && Pm4::CopyDataSource(packet) == Pm4::CopyDataGpuClockSource &&
                   !DrawPipeline::Queue0().Overlaps(packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u), Pm4::CopyDataBytes(packet));
        case 0x63: case 0x64: case 0x9f:
            return packet.size() >= 5 && !DrawPipeline::Queue0().Overlaps(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u), static_cast<std::size_t>(packet[4]) * 8);
        case 0x50: {
            if (packet.size() < 7) return false;
            const auto destination = Pm4::DmaDestination(packet);
            if (destination == Pm4::DmaDestinationNowhere) return true;
            const auto bytes = Pm4::DmaBytes(packet);
            if (bytes == 0) return true;
            const auto source = Pm4::DmaSource(packet);
            if ((source == 0 || source == 3) && DrawPipeline::Queue0().Overlaps(packet[2] | (static_cast<std::uint64_t>(packet[3]) << 32u), bytes)) return false;
            if (source == Pm4::DmaSelectGds && DrawPipeline::Queue0().Overlaps(Pm4::GdsAddress() + packet[2], bytes)) return false;
            const auto address = destination == Pm4::DmaSelectGds ? Pm4::GdsAddress() + packet[4] : packet[4] | (static_cast<std::uint64_t>(packet[5]) << 32u);
            return !DrawPipeline::Queue0().Overlaps(address, bytes);
        }
        case 0x3c: case 0x93: {
            if (packet.size() < 5) return false;
            const auto bytes = Pm4::WaitAwaitedBytes(packet);
            const std::uint64_t awaited = (packet[2] & (bytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
            if (DrawPipeline::Queue0().Overlaps(awaited, bytes)) return false;
            GuestMemory::CheckRange(reinterpret_cast<const void*>(awaited), bytes, bytes);
            return Pm4::WaitSatisfiedUnchecked(packet);
        }
        case 0x10: {
            const auto operation = (header >> 2u) & 0x3fu;
            return operation == 0 || operation == 0x09 || operation == 0x0b || operation == 0x0c || operation == 0x1a;
        }
        default: return false;
    }
}

bool sealsWriteOnly(std::span<const std::uint32_t> packet, std::uint32_t opcode) {
    if (opcode == 0x58) return true;
    if (opcode != 0x46 || packet.size() < 2) return false;
    switch (packet[1] & 0x3fu) {
        case 0x07: case 0x0f: case 0x10: case 0x16: case 0x2a: case 0x2c: case 0x2e: case 0x31: return true;
        default: return false;
    }
}

bool& drawnSinceSeal() {
    static thread_local bool drawn = false;
    return drawn;
}

struct LaneCounts {
    std::atomic<std::uint64_t> runs{0}, packets{0}, draws{0}, refused{0}, cut{0}, lists{0}, flips{0}, verified{0}, listed{0}, listDrains{0};
};

LaneCounts& laneCounts() {
    static LaneCounts counts;
    return counts;
}

void laneBump(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

bool lanePacket(const Submission& submission, std::size_t cursor, std::uint32_t header, std::uint32_t opcode) {
    const auto lane = Core::LaneOf(header);
    if (lane == Core::Lane::Heavy || lane == Core::Lane::Skip) return false;
    if (!keepsDrawFinisher(header, opcode) && DrawFinisher::Busy() && !DrawFinisher::Overlap()) {
        laneBump(laneCounts().lists);
        return false;
    }
    if (lane == Core::Lane::RegisterList) {
        const auto listed = submission.registerLists.find(cursor);
        if (listed == submission.registerLists.end() || !Pm4::IndirectRegistersValid(listed->second)) {
            laneBump(laneCounts().lists);
            return false;
        }
    }
    return true;
}

struct DecoderSeat {
    void* target = nullptr;
    std::uint64_t (*call)(void* target, const Core::DecodedOp& op) = nullptr;
    Core::DecoderRing ring;
    Core::Decoder decoder;
    std::atomic<std::uint64_t> runs{0}, draws{0}, packets{0}, batches{0}, runNs{0}, enqueueNs{0}, heldKnown{0}, heldUnknown{0}, heldDecoded{0}, shapesKept{0}, statesKept{0};
    struct Held {
        bool unknown = true;
        std::vector<DrawFinisher::Range> writes;
    };
    std::unordered_map<std::uint64_t, Held> held;

    DecoderSeat() : ring(Core::DecoderRing::Settings{1024, 8, false, 0}), decoder(Core::Decoder::Settings{512}, Core::DecoderHost{this, &DecoderSeat::draw, &DecoderSeat::heavy, &DecoderSeat::retired, &DecoderSeat::await}) {}

    static std::uint64_t draw(void* context, const Core::DecodedOp& op) {
        auto& seat = *static_cast<DecoderSeat*>(context);
        return seat.call(seat.target, op);
    }
    static Core::DecoderHeavy heavy(void*, std::size_t cursor, std::span<const std::uint32_t>, QueueState&) { return {cursor, 0, true}; }
    static std::uint64_t retired(void* context) { return static_cast<DecoderSeat*>(context)->ring.Retired(); }
    static void await(void* context, std::uint64_t count) { static_cast<DecoderSeat*>(context)->ring.AwaitRetired(count); }
};

DecoderSeat& decoderSeat() {
    static DecoderSeat seat;
    return seat;
}

bool decoderEnabled() {
    static const bool enabled = std::getenv("APS5_DECODER") != nullptr && DrawPipeline::OpsAtExit() && DrawFinisher::Overlap();
    return enabled;
}

constexpr std::uint32_t DecoderBatchDraws = 12;

bool decoderKeepsState(std::uint32_t header, std::uint32_t opcode) {
    switch (opcode) {
        case 0x3c: case 0x93: case 0x42: case 0x46: case 0x49: case 0x58: return true;
        case 0x10: {
            const auto operation = (header >> 2u) & 0x3fu;
            return operation == 0 || operation == 0x0b || operation == 0x0c;
        }
        default: return false;
    }
}

void decoderReport() {
    static std::atomic<std::int64_t> lastReport{0};
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (nowMs - last < 5000 || !lastReport.compare_exchange_strong(last, nowMs)) return;
    auto& seat = decoderSeat();
    const auto value = [](const std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.load(std::memory_order_relaxed)); };
    const auto decoded = seat.decoder.Counted();
    const auto ring = seat.ring.Counted();
    const auto draws = seat.draws.load(std::memory_order_relaxed);
    std::fprintf(stderr, "[decoder] run (cumulative): %llu runs, %llu packets decoded, %llu draws posted in %llu batches; %.2f us per draw in the runs, %.2f us of them handing batches to the draw finisher; snapshots: shader %llu, context %llu, user configuration %llu; %llu waits for a snapshot, %llu for room in the ring; batches queued with their target ranges %llu, without %llu, %llu draw shapes decoded for their ranges, %llu draws kept the shape of the draw before, %llu runs kept the register state of the run before%c", value(seat.runs), value(seat.packets), value(seat.draws), value(seat.batches), draws != 0 ? static_cast<double>(seat.runNs.load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(draws) : 0.0, draws != 0 ? static_cast<double>(seat.enqueueNs.load(std::memory_order_relaxed)) / 1000.0 / static_cast<double>(draws) : 0.0, static_cast<unsigned long long>(decoded.shaderSnapshots), static_cast<unsigned long long>(decoded.contextSnapshots), static_cast<unsigned long long>(decoded.userConfigSnapshots), static_cast<unsigned long long>(decoded.snapshotWaits), static_cast<unsigned long long>(ring.roomWaits), value(seat.heldKnown), value(seat.heldUnknown), value(seat.heldDecoded), value(seat.shapesKept), value(seat.statesKept), 10);
}

void laneReport() {
    static std::atomic<std::int64_t> lastReport{0};
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (nowMs - last < 5000 || !lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto& counts = laneCounts();
    const auto value = [](const std::atomic<std::uint64_t>& counter) { return static_cast<unsigned long long>(counter.load(std::memory_order_relaxed)); };
    std::fprintf(stderr, "[core-lane] (cumulative) %llu light runs, %llu packets (%llu draws) taken by the light lane, %llu runs ended by a packet the lane leaves, %llu by the run limit, %llu register lists or indexed registers left, %llu flips left, %llu register lists taken, %llu light packets verified on the heavy path, %llu of them register lists the heavy path drains the pipeline for%c", value(counts.runs), value(counts.packets), value(counts.draws), value(counts.refused), value(counts.cut), value(counts.lists), value(counts.flips), value(counts.listed), value(counts.verified), value(counts.listDrains), 10);
}

}

void Driver::deliverGuestEopInterrupt(std::uint32_t queue, std::uint32_t context) {
    Graphics::BufferCache::EndDrawRun(Graphics::BufferCache::RunEnd::Interrupt);
    AgcDriverDeliverEopInterrupt(queue, context);
}

std::optional<std::uint32_t> Driver::releaseInterrupt(std::span<const std::uint32_t> packet, std::uint32_t opcode, bool carriesFlip) {
    if (opcode != 0x49 || carriesFlip || packet.size() < 3) return std::nullopt;
    const auto interruptSelect = (packet[2] >> 24u) & 7u;
    if (interruptSelect == 0 || interruptSelect == 3) return std::nullopt;
    if (packet.size() < 8) throw std::runtime_error("AGC: an interrupting release has no interrupt context dword");
    const auto context = packet[7] & 0x0fffffffu;
    if ((context & 0x08000000u) != 0) throw std::runtime_error("AGC: a release asks for typed interrupt context " + std::to_string(context) + ", which is not delivered as a graphics end-of-pipe event; not implemented");
    return context;
}

template <typename TWork>
void Driver::timed(double WorkerProfile::*bucket, TWork&& work) {
    static const bool report = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!report) {
        work();
        return;
    }
    struct Slot {};
    auto& profile = ThreadScratch<WorkerProfile, Slot>();
    const auto begin = std::chrono::steady_clock::now();
    work();
    const auto end = std::chrono::steady_clock::now();
    profile.*bucket += std::chrono::duration<double, std::milli>(end - begin).count();
    if (end - profile.reported > std::chrono::seconds(10)) {
        profile.reported = end;
        AgcDriver::ProfilePrint_nid_no_patch( "[gpu] worker at %.0f s: dispatch %.1f s, draw %.1f s, wait %.1f s\n", std::chrono::duration<double>(end - profile.start).count(), profile.dispatchMs / 1000, profile.drawMs / 1000, profile.waitMs / 1000);
    }
    if (report && bucket != &WorkerProfile::waitMs) PerformanceCounters::AddDriverNanoseconds(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
}

void Driver::execute(Submission& submission) {
    auto* submissionTiming = includeTimingSubmission(submission, true);
    struct FlipHold {
        Driver& driver;
        std::uint32_t queue;
        bool held;
        bool working;
        std::shared_ptr<FlipWork> work;
        void Release() {
            if (!held) return;
            held = false;
            driver.releaseFlipHold();
        }
        void Finish() {
            if (!working) return;
            const auto workSerial = driver.submitFlipWork(queue);
            working = false;
            driver.finishFlipWork(work, workSerial);
        }
        ~FlipHold() {
            Release();
            if (working) driver.finishFlipWork(work, 0);
        }
    } flipHold{*this, submission.queue, submission.holdsFlip, submission.holdsFlip, submission.flipWork};
    const bool pipelined = submission.queue == 0 && DrawPipeline::Depth() != 0;
    static const bool orderedWaits = std::getenv("APS5_PIPELINE_DRAIN_LABELS") == nullptr;
    DrawPipeline::Active() = pipelined;
    struct PipelineDrain {
        bool pipelined;
        ~PipelineDrain() {
            if (!pipelined || std::uncaught_exceptions() == 0) return;
            try {
                DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
            } catch (...) {
            }
        }
    } pipelineDrain{pipelined};
    static const bool orderedCompletion = std::getenv("APS5_PIPELINE_DRAIN_SUBMISSIONS") == nullptr && orderedWaits;
    if (pipelined && (submission.suspend || !orderedCompletion)) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    if (submission.suspend) {
        PerformanceContext timingContext(submissionTiming);
        PerformanceTimer timing("Driver.Suspend");

        static const bool suspendDrain = std::getenv("APS5_SUSPEND_DRAIN") != nullptr;
        static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
        ++suspendPoints;
        auto& costs = submissionCosts(submission.queue);

        if (suspendDrain || !deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
            const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
            std::lock_guard gpuLock(GuestMemory::GpuMutex());
            const auto localDevice = device.Load();
            recordDeferredLabels(localDevice.get(), submission.queue);
            if (localDevice != nullptr) {
                if (suspendDrain) localDevice->WaitIdle();
                else localDevice->SubmitRecorded(submission.queue == 0);
            }
            ++costs.suspends;
            if (profile) costs.suspendNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
        } else {
            ++costs.suspendsSkipped;
        }
        resetGraphics = true;
        return;
    }
    QueueState* state = nullptr;
    {
        std::lock_guard lock(mutex);
        if (submission.queue == 0 && resetGraphics) {
            queues.erase(0);
            resetGraphics = false;
        }
        state = &queues[submission.queue];
    }
    auto& queue = *state;
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) std::fprintf(stderr, "[gpu] %.1f execute serial=%llu queue=0x%x dwords=%zu\n", TraceMs(), static_cast<unsigned long long>(submission.serial), submission.queue, submission.commands.size());

    static const long dumpQueue = [] { const char* text = std::getenv("APS5_DUMP_QUEUE"); return text ? std::strtol(text, nullptr, 16) : -1L; }();
    if (static_cast<long>(submission.queue) == dumpQueue) {

        static int dumped = 0;
        if (dumped++ < 40) {
            std::string text = "[queue] submission " + std::to_string(submission.serial) + ":\n";
            for (std::size_t cursor = 0; cursor < submission.commands.size();) {
                const auto header = submission.commands[cursor];
                const auto count = Pm4::PacketWords(header);
                char line[200];
                int length = std::snprintf(line, sizeof(line), "[queue]   %s", Pm4::Name(header).c_str());
                for (std::size_t i = 1; i < count && i < 10 && length < 180; ++i) length += std::snprintf(line + length, sizeof(line) - length, " %08x", submission.commands[cursor + i]);
                text += line;
                text += "\n";
                cursor += count;
            }
            std::fputs(text.c_str(), stderr);
        }
    }
    PacketHistory recent{submission.commands};

    static const bool profilePackets = std::getenv("APS5_PROFILE_DRAW") != nullptr;

    thread_local PacketProfile* packetProfileSlot = nullptr;
    auto& packetProfile = ShaderRecompiler::ThreadOwned(packetProfileSlot);
    ++packetProfile.submissions;
    auto& packetCounters = PacketCounters::ThisThread();

    static const bool epochEach = std::getenv("APS5_SUBMISSION_EPOCH_EACH") != nullptr;
    if (epochEach || submission.enqueuedAt == std::chrono::steady_clock::time_point{} || submission.enqueuedAt > lastEpochBump()) {
        if (!epochEach) noteReleasedTails(submission);
        bumpEpoch(&EpochBumps::submissions);
    }
    clearResolvedAhead();
    gateOpened() = true;
    struct ClearAhead {
        ~ClearAhead() { clearResolvedAhead(); }
    } clearAhead;
    struct SettleFinisher {
        bool queue0;
        ~SettleFinisher() {
            if (queue0) DrawFinisher::Get().Settle();
        }
    } settleFinisher{submission.queue == 0};
    std::size_t nextSegment = 0;
    auto* const executingPacket = workerPacket();
    static const bool lightLane = Graphics::CoreLightLane();
    static const bool verifyLane = Graphics::VerifyCoreLightLane();
    static const bool laneStats = Graphics::CoreRunStats();
    static const bool verifyDecoder = Core::DecoderShadow::Enabled();
    thread_local Core::DecoderShadow* decoderShadowSlot = nullptr;
    auto* const decoderShadow = verifyDecoder ? &ShaderRecompiler::ThreadOwned(decoderShadowSlot) : nullptr;
    if (decoderShadow != nullptr) decoderShadow->Begin(queue);
    const bool laneOpen = (lightLane || verifyLane) && pipelined && !profilePackets && !traceGpu && !packetEpoch() && !CaptureTrace::Enabled() && static_cast<long>(submission.queue) != dumpQueue;
    const auto lightRun = [&](std::size_t& cursor, std::size_t end) {
        std::size_t at = cursor;
        while (at < end && Pm4::FillerPacket(submission.commands[at])) ++at;
        if (at >= end || !lanePacket(submission, at, submission.commands[at], (submission.commands[at] >> 8u) & 0xffu) || submission.flips.contains(at)) return false;
        std::uint64_t done = 0, draws = 0;
        {
            std::shared_lock deviceUse(deviceReplacement);
            GuestMemory::SetCurrentPacket((submission.commands[at] >> 8u) & 0xffu, submission.queue);
            flushBetweenPackets(submission.queue, submission.commands[at], false);
            struct Progress {
                Driver& driver;
                std::uint64_t& done;
                ~Progress() {
                    --driver.packetsInFlight;
                    driver.packetsDone += done;
                }
            } progress{*this, done};
            ++packetsInFlight;
            while (at < end) {
                const auto header = submission.commands[at];
                if (Pm4::FillerPacket(header)) {
                    ++at;
                    continue;
                }
                const auto opcode = (header >> 8u) & 0xffu;
                if (!lanePacket(submission, at, header, opcode)) {
                    laneBump(laneCounts().refused);
                    break;
                }
                if (!submission.flips.empty() && submission.flips.contains(at)) {
                    laneBump(laneCounts().flips);
                    break;
                }
                const auto count = Pm4::PacketWords(header);
                const auto packet = std::span(submission.commands).subspan(at, count);
                if (decoderShadow != nullptr) decoderShadow->Step(submission.commands, at, submission.registerLists, queue, false);
                queue.runMask.Packet(opcode);
                if (executingPacket != nullptr) executingPacket->store((static_cast<std::uint64_t>(at + 1) << 8u) | opcode, std::memory_order_relaxed);
                recent.Record(at);
                if (Pm4::DrawOpcode(opcode)) {
                    GuestMemory::SetCurrentPacket(opcode, submission.queue);
                    landPendingWaits(submission.queue);
                    packetCounters.CountDraw();
                    drawnSinceSeal() = true;
                    try {
                        std::string rejected;
                        if (draw(queue, packet, submission, at, rejected) == DrawVerdict::Rejected) throw std::runtime_error(rejected);
                    } catch (const RefusedDraw& refused) {
                        ReportRefusedDraw(refused);
                    }
                    Graphics::Recorder::CountRecordedWork();
                    ++draws;
                } else if (Pm4::IndirectRegisterOpcode(opcode)) {
                    Pm4::ExecuteIndirectRegisters(packet, submission.registerLists.at(at), queue);
                    laneBump(laneCounts().listed);
                } else {
                    Pm4::Execute(packet, queue);
                }
                ++done;
                at += count;
            }
        }
        if (at >= end && end < submission.commands.size()) laneBump(laneCounts().cut);
        laneBump(laneCounts().runs);
        laneBump(laneCounts().packets, done);
        laneBump(laneCounts().draws, draws);
        if (laneStats && (laneCounts().runs.load(std::memory_order_relaxed) & 0xffu) == 0) laneReport();
        cursor = at;
        return true;
    };
    static const bool decoderOn = decoderEnabled();
    std::shared_ptr<Submission> decoderCopy;
    bool decoderStale = true;
    bool decoderShaped = false;
    std::uint64_t decoderShape = 0;
    const auto decoderRun = [&](std::size_t& cursor, std::size_t end) {
        std::size_t at = cursor;
        while (at < end && Pm4::FillerPacket(submission.commands[at])) ++at;
        if (at >= end || !lanePacket(submission, at, submission.commands[at], (submission.commands[at] >> 8u) & 0xffu)) return false;
        if (const auto flip = submission.flips.lower_bound(at); flip != submission.flips.end()) end = std::min(end, flip->first);
        if (at >= end) return false;
        auto& seat = decoderSeat();
        const auto started = std::chrono::steady_clock::now();
        std::uint64_t enqueueNs = 0;
        std::uint32_t pending = 0;
        std::uint64_t posted = 0;
        bool batchUnknown = false;
        std::uint64_t batchShape = 0;
        std::vector<DrawFinisher::Range> batchWrites;
        const auto heldOf = [&](std::uint64_t& shape, std::uint32_t runBits) -> const DecoderSeat::Held& {
            static const DecoderSeat::Held unknown;
            if (decoderShaped && (runBits & (Core::RunMask::State | Core::RunMask::Other)) == 0) {
                if (const auto kept = seat.held.find(decoderShape); kept != seat.held.end()) {
                    shape = decoderShape;
                    laneBump(seat.shapesKept);
                    return kept->second;
                }
            }
            decoderShaped = false;
            const auto localDevice = device.Load();
            if (localDevice == nullptr) return unknown;
            drawRegisterKey(queue, *submission.shaders, localDevice->Serial(), &shape);
            decoderShape = shape;
            decoderShaped = true;
            if (const auto found = seat.held.find(shape); found != seat.held.end()) return found->second;
            if (seat.held.size() >= 8192) seat.held.clear();
            DecoderSeat::Held fresh;
            try {
                const auto decoded = decodeDraw(queue, submission, localDevice->graphicsContext());
                bool stores = false;
                for (std::size_t index = 0; index < decoded->programs.size() && !stores; ++index) {
                    if (index < decoded->roles.size() && decoded->roles[index] == ShaderRecompiler::ProgramRole::GeometryBack) continue;
                    const auto& program = decoded->programs[index];
                    stores = program.snapshot == nullptr || ProgramWritesMemory(*program.snapshot, program.codeOffset) || (program.tailContinuation != nullptr && ProgramWritesMemory(*program.tailContinuation, 0));
                }
                if (!stores) {
                    fresh.writes = DrawWriteRanges(decoded->state, {}, ExactDrawWrites());
                    fresh.unknown = false;
                }
            } catch (const std::exception&) {
                fresh = DecoderSeat::Held{};
            }
            laneBump(seat.heldDecoded);
            return seat.held.emplace(shape, std::move(fresh)).first->second;
        };
        const auto closeBatch = [&] {
            if (pending == 0) return;
            DrawFinisher::Pending heldBatch;
            heldBatch.unknown = batchUnknown;
            if (!batchUnknown) heldBatch.writes = std::move(batchWrites);
            laneBump(batchUnknown ? seat.heldUnknown : seat.heldKnown);
            batchUnknown = false;
            batchShape = 0;
            batchWrites.clear();
            const auto batch = seat.ring.TryClaim(pending);
            if (batch.count != pending) throw std::logic_error("APS5_DECODER: the draws posted for a batch are not the ones the ring hands back");
            pending = 0;
            if (decoderCopy == nullptr) {
                decoderCopy = std::make_shared<Submission>();
                decoderCopy->serial = submission.serial;
                decoderCopy->queue = submission.queue;
                decoderCopy->shaders = submission.shaders;
                decoderCopy->commands = submission.commands;
            }
            const auto epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
            const auto from = std::chrono::steady_clock::now();
            DrawFinisher::Get().Enqueue([this, batch, epoch, copy = decoderCopy] {
                auto& ringSeat = decoderSeat();
                struct Done {
                    DecoderSeat& seat;
                    Core::DecoderRing::Batch batch;
                    ~Done() { seat.ring.Complete(batch, nullptr, nullptr); }
                } done{ringSeat, batch};
                struct PipelineScope {
                    bool previous;
                    PipelineScope() : previous(std::exchange(DrawPipeline::Active(), true)) {}
                    ~PipelineScope() { DrawPipeline::Active() = previous; }
                } pipelineScope;
                thread_local Core::DecodedView* viewSlot = nullptr;
                auto& view = ShaderRecompiler::ThreadOwned(viewSlot);
                DrawPipeline::FollowEpoch(epoch);
                for (std::uint32_t index = 0; index < batch.count; ++index) {
                    const auto& op = ringSeat.ring.At(batch.first + index);
                    Core::ApplyDecoded(op.state, view);
                    if (index == 0) view.queue.runMask.bits |= Core::RunMask::State | Core::RunMask::Other;
                    GuestMemory::SetCurrentPacket(op.opcode, copy->queue);
                    const auto words = std::span<const std::uint32_t>(copy->commands).subspan(op.offset, op.words);
                    try {
                        std::string rejected;
                        if (draw(view.queue, words, *copy, op.offset, rejected) == DrawVerdict::Rejected) throw std::runtime_error(rejected);
                    } catch (const RefusedDraw& refused) {
                        ReportRefusedDraw(refused);
                    }
                }
            }, std::move(heldBatch));
            enqueueNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - from).count());
            laneBump(seat.batches);
        };
        const auto postDraw = [&](const Core::DecodedOp& op) -> std::uint64_t {
            if (executingPacket != nullptr) executingPacket->store((static_cast<std::uint64_t>(op.offset + 1) << 8u) | op.opcode, std::memory_order_relaxed);
            recent.Record(op.offset);
            GuestMemory::SetCurrentPacket(op.opcode, submission.queue);
            landPendingWaits(submission.queue);
            packetCounters.CountDraw();
            drawnSinceSeal() = true;
            Graphics::Recorder::CountRecordedWork();
            std::uint64_t shape = 0;
            const auto& heldDraw = heldOf(shape, op.state.runBits);
            if (heldDraw.unknown != batchUnknown && pending != 0) closeBatch();
            batchUnknown = heldDraw.unknown;
            if (!heldDraw.unknown && (pending == 0 || shape != batchShape)) {
                for (const auto& range : heldDraw.writes) {
                    if (std::find(batchWrites.begin(), batchWrites.end(), range) == batchWrites.end()) batchWrites.push_back(range);
                }
                batchShape = shape;
            }
            const auto order = seat.ring.Post(op);
            ++pending;
            ++posted;
            if (pending == DecoderBatchDraws || batchUnknown) closeBatch();
            return order;
        };
        const auto before = seat.decoder.Counted().packets;
        Core::Decoder::Result result;
        {
            std::shared_lock deviceUse(deviceReplacement);
            GuestMemory::SetCurrentPacket((submission.commands[at] >> 8u) & 0xffu, submission.queue);
            flushBetweenPackets(submission.queue, submission.commands[at], false);
            if (decoderStale) {
                Pm4::RefreshDigests(queue);
                seat.decoder.Touch(Core::DecoderEvery);
                decoderShaped = false;
                decoderStale = false;
            } else {
                laneBump(seat.statesKept);
            }
            seat.target = const_cast<void*>(static_cast<const void*>(&postDraw));
            seat.call = [](void* target, const Core::DecodedOp& op) { return (*static_cast<decltype(postDraw)*>(target))(op); };
            Core::Decoder::Input input;
            input.commands = submission.commands;
            input.cursor = at;
            input.limit = end;
            input.lists = &submission.registerLists;
            input.submission = submission.serial;
            struct Progress {
                Driver& driver;
                ~Progress() { --driver.packetsInFlight; }
            } progress{*this};
            ++packetsInFlight;
            result = seat.decoder.Decode(queue, input);
            closeBatch();
        }
        const auto done = seat.decoder.Counted().packets - before;
        packetsDone += done;
        laneBump(seat.runs);
        laneBump(seat.packets, done);
        laneBump(seat.draws, posted);
        laneBump(seat.enqueueNs, enqueueNs);
        laneBump(seat.runNs, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
        if ((seat.runs.load(std::memory_order_relaxed) & 0xffu) == 0) decoderReport();
        if (result.cursor == at) return false;
        cursor = result.cursor;
        return true;
    };
    for (std::size_t cursor = 0; cursor < submission.commands.size();) {
        if (packetEpoch()) bumpEpoch(&EpochBumps::packets);
        CheckFailure();
        while (nextSegment < submission.indirectSegments.size() && submission.indirectSegments[nextSegment].start <= cursor) {
            if (submission.indirectSegments[nextSegment].start == cursor && refreshIndirectSegment(submission, nextSegment)) {
                recent.commands = submission.commands;
                decoderCopy.reset();
                continue;
            }
            ++nextSegment;
        }
        if (cursor >= submission.commands.size()) break;
        const bool laneNow = laneOpen && queue.predication.operation == 0 && deferredLabels().labels.empty();
        if (laneNow && (lightLane || decoderOn)) {
            const bool decodes = decoderOn && submission.queue == 0 && decoderShadow == nullptr;
            const auto limit = nextSegment < submission.indirectSegments.size() ? std::min<std::size_t>(submission.indirectSegments[nextSegment].start, submission.commands.size()) : submission.commands.size();
            const auto run = Core::ScanLane(submission.commands, cursor, limit, decodes ? 4096 : 64);
            if (run.packets != 0 && (decodes ? decoderRun(cursor, run.end) : lightLane && lightRun(cursor, run.end))) continue;
        }
        const auto header = submission.commands[cursor];
        if (Pm4::FillerPacket(header)) { ++cursor; continue; }
        const auto count = Pm4::PacketWords(header);
        const auto packet = std::span(submission.commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        if (decoderShadow != nullptr) decoderShadow->Step(submission.commands, cursor, submission.registerLists, queue, false);
        if (decoderOn && !decoderKeepsState(header, opcode)) decoderStale = true;
        if (opcode != 0x10 || (((header >> 2u) & 0x3fu) != 0 && ((header >> 2u) & 0x3fu) != 0x0b && ((header >> 2u) & 0x3fu) != 0x0c)) queue.runMask.Packet(opcode);
        if (executingPacket != nullptr) executingPacket->store((static_cast<std::uint64_t>(cursor + 1) << 8u) | opcode, std::memory_order_relaxed);
        auto nextCursor = cursor + count;
        const bool predicated = Pm4::Predicated(header) && queue.predication.operation != 0;
        if (!(pipelined && !predicated && (opcode == 0x49 || opcode == 0x37) && header != FlipPacketHeader && header != RenderingWaitPacketHeader && !submission.flips.contains(cursor))) FlushLabelRun();
        const bool laneChecked = verifyLane && laneNow && !predicated && lanePacket(submission, cursor, header, opcode) && !submission.flips.contains(cursor);
        const auto laneDiffers = [&](const char* what) {
            throw std::runtime_error(std::string("APS5_VERIFY_CORE_LIGHT_LANE: a packet the light lane would take (") + Pm4::Name(header) + ") " + what);
        };
        if (laneChecked && submission.queue == 0 && DrawFinisher::Busy() && !keepsDrawFinisher(header, opcode) && !(DrawFinisher::Overlap() && keepsDrawPipeline(packet, header, opcode)) && !Pm4::IndirectRegisterOpcode(opcode)) laneDiffers("drains the draw finisher on the heavy path");
        if (submission.queue == 0 && DrawFinisher::Busy() && !keepsDrawFinisher(header, opcode)) {
            const bool queuesLabel = pipelined && !predicated && header != FlipPacketHeader && header != RenderingWaitPacketHeader && !submission.flips.contains(cursor) && labelPacketQueues(packet, opcode);
            const bool heldLabelWait = pipelined && !predicated && orderedWaits && (opcode == 0x3c || opcode == 0x93) && header != RenderingWaitPacketHeader && waitTakesPendingLabel(packet);
            const bool beside = DrawFinisher::Overlap() && opcode != 0x15 && (queuesLabel || heldLabelWait || keepsDrawPipeline(packet, header, opcode));
            if (!beside) DrawFinisher::Get().Drain();
            if (DrawFinisher::Overlap()) {
                DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Packet, !beside);
                if (!beside) DrawFinisher::NoteDrainedOpcode(header == FlipPacketHeader ? 0xffu : opcode, keepsDrawPipeline(packet, header, opcode));
            }
        }
        if (APS5_ENABLE_TIMING_LOG && submission.queue == 0 && pendingFrameTiming == nullptr) includeTimingSubmission(submission, false);
        static thread_local Core::CoarseClock packetClock{nullptr, nullptr, Graphics::CoreClockPeriod()};
        const bool timedPacket = !lightLane || header == FlipPacketHeader || submission.flips.contains(cursor) || packetClock.Tick();
        PerformanceContext timingContext(!timedPacket ? nullptr : submission.queue == 0 ? pendingFrameTiming.get() : FrameTiming::Async());
        if (submission.queue != 0 && Graphics::Recorder::DispatchOverlapEnabled()) noteRingPacket(submission.queue, submission.serial, opcode, header);
        PerformanceTimer timing("Driver.Packet");
        if (!predicated) noteLabelIssue(packet, submission.queue);
        if (pipelined && !predicated && header != FlipPacketHeader && header != RenderingWaitPacketHeader && !submission.flips.contains(cursor) && (enqueueLabelPacket(packet, opcode, submission.queue, releaseInterrupt(packet, opcode, false)) || enqueueDmaPacket(packet, opcode, submission.queue, queue) || enqueueConstantRamDump(packet, opcode, submission.queue, queue) || enqueueSampleDumpPacket(packet, opcode, submission.queue))) {
            if (laneChecked) laneDiffers("is queued as a label, a copy or a dump on the heavy path");
            traceLabel(packet, submission.queue);
            recent.Record(cursor);
            cursor = nextCursor;
            continue;
        }
        if (pipelined && !predicated && (opcode == 0x3c || opcode == 0x93) && header != RenderingWaitPacketHeader && orderedWaits && DrawPipeline::Queue0().Busy()) {
            const auto awaitedBytes = Pm4::WaitAwaitedBytes(packet);
            const std::uint64_t awaited = (packet[2] & (awaitedBytes == 8 ? ~7u : ~3u)) | (static_cast<std::uint64_t>(packet[3]) << 32u);
            const auto value = DrawPipeline::Queue0().PendingLabel(awaited, awaitedBytes);
            if (value.has_value() && Pm4::WaitComparesValue(packet, *value)) {
                ++waitOutcomes().fromRecorderSameQueue;
                gateOpened() = true;
                noteLabelPass(packet, submission.queue);
                recent.Record(cursor);
                cursor = nextCursor;
                continue;
            }
            if (!DrawPipeline::Queue0().Overlaps(awaited, awaitedBytes) && Pm4::WaitSatisfiedUnchecked(packet) && !OwnStoreRejects(packet, awaited, awaitedBytes, submission.queue)) {
                bumpWaitEpoch(awaited, awaitedBytes, true);
                ++waitOutcomes().atEntry;
                gateOpened() = true;
                noteLabelPass(packet, submission.queue);
                recent.Record(cursor);
                cursor = nextCursor;
                continue;
            }
        }
        timing.Mark("pipeline_prepare");
        if (laneChecked && !keepsDrawPipeline(packet, header, opcode)) {
            if (!Pm4::IndirectRegisterOpcode(opcode)) laneDiffers("drains the draw pipeline on the heavy path");
            laneBump(laneCounts().listDrains);
        }
        if (pipelined && DrawPipeline::Queue0().Busy() && !keepsDrawPipeline(packet, header, opcode)) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, header == FlipPacketHeader ? 0xffu : opcode);
        if (pipelined && predicated && DrawPipeline::Queue0().Busy() && DrawPipeline::Queue0().Overlaps(queue.predication.address, queue.predication.operation == 3 ? 8 : 4)) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, opcode);
        timing.Mark("pipeline_drain");
        std::shared_lock deviceUse(deviceReplacement, std::defer_lock);
        if (opcode != 0x3c && opcode != 0x93 && header != RenderingWaitPacketHeader && header != FlipPacketHeader) deviceUse.lock();

        timing.Mark("device_lock");
        GuestMemory::SetCurrentPacket(header == FlipPacketHeader ? 0xffffu : opcode, submission.queue);
        CaptureTrace::Log("packet submission=%llu queue=%x offset=%zu header=%08x words=%zu", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, header, packet.size());
        if (predicated) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            if (!Pm4::PredicationPasses(queue)) {
                cursor = opcode == 0x3f ? submission.conditionalEnds.at(cursor) : nextCursor;
                continue;
            }
            noteLabelIssue(packet, submission.queue);
        }
        if (opcode == 0x3f) {
            cursor = nextCursor;
            continue;
        }

        const auto flushStart = profilePackets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        flushBetweenPackets(submission.queue, header, opcode == 0x49 || opcode == 0x37);
        timing.Mark("flush_between_packets");
        PacketTimer packetTimer{profilePackets, header == FlipPacketHeader ? 0xffffu : opcode, submission.queue, packetProfile, profilePackets ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};

        if (profilePackets) {
            packetStartedAt() = packetTimer.start;
            pendingDispatchPhases() = {};
            pendingDrawPhases() = {};
        }
        const auto finishDispatchPacket = [&](bool indirect) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDispatchPhases();
            if (pending.phases) {
                pending.ms[PhaseEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
                addDriverPhases(submission.queue != 0 ? OtherQueues : indirect ? Queue0Indirect : Queue0Direct, pending.ms, pending.hit, pending.validated);
            }
            if (indirect) return;
            auto& row = packetProfile.dispatchOutcomes[static_cast<std::size_t>(pending.outcome)];
            ++row.first;
            row.second += std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
        };
        const auto finishDrawPacket = [&](bool drawn) {
            if (!profilePackets) return;
            const auto now = std::chrono::steady_clock::now();
            auto& pending = pendingDrawPhases();
            std::array<double, DrawDriverPhaseCount> ms{};
            if (drawn && pending.phases) {
                ms = pending.ms;
                ms[DrawRowEpilogue] = std::chrono::duration<double, std::milli>(now - pending.tailAt).count();
            } else {
                ms[DrawRowSkipped] = std::chrono::duration<double, std::milli>(now - packetTimer.start).count();
            }
            addDrawPhases(ms, drawn && pending.phases, pending.captures);
        };
        if (profilePackets) packetProfile.flushMs += std::chrono::duration<double, std::milli>(packetTimer.start - flushStart).count();

        bool wroteOnGpu = false, endOfPipeInterrupt = false, interruptDeferred = false, drawPacket = false, sampleDump = false;
        const bool drains = preparePacketMemory(submission, queue, packet, header, opcode, releaseInterrupt(packet, opcode, submission.flips.contains(cursor)), wroteOnGpu, endOfPipeInterrupt, interruptDeferred, drawPacket, sampleDump);
        traceLabel(packet, submission.queue);
        if (laneChecked) {
            if (drains || wroteOnGpu || endOfPipeInterrupt || interruptDeferred || sampleDump) laneDiffers("writes, drains or interrupts in its memory preparation on the heavy path");
            if (drawPacket != (Core::LaneOf(header) == Core::Lane::Draw)) laneDiffers("is classed differently by the heavy path");
            if (!drawPacket && (opcode == 0x15 || opcode == 0x16 || opcode == 0x3c || opcode == 0x93 || opcode == 0x22 || opcode == 0x91 || opcode == 0x3f || sealsWriteOnly(packet, opcode) || opcode == 0x42 || opcode == 0x46 || opcode == 0x58)) laneDiffers("takes a branch of the heavy path the lane does not have");
            laneBump(laneCounts().verified);
            if (laneStats && (laneCounts().verified.load(std::memory_order_relaxed) & 0xfffu) == 0) laneReport();
        }
        timing.Mark("prepare_memory");

        const bool waitPacket = opcode == 0x3c || opcode == 0x93 || header == RenderingWaitPacketHeader;
        struct Progress {
            Driver& driver;
            bool counted;
            ~Progress() {
                if (!counted) return;
                --driver.packetsInFlight;
                ++driver.packetsDone;
            }
        } progress{*this, !waitPacket};
        if (!waitPacket) ++packetsInFlight;
        recent.Record(cursor);
        if (header == RenderingWaitPacketHeader) {
            timed(&WorkerProfile::waitMs, [&] { submission.renderingWaits.at(cursor)->Wait(); });
        } else if (submission.flips.contains(cursor)) {
            if (opcode == 0x49) {
                if (!wroteOnGpu) {
                    Pm4::Execute(packet, queue);
                    if (const auto label = Pm4::DecodeLabelWrite(packet)) noteLabelStore(label->address, label->Bytes(), ++eventSerial, submission.queue);
                }
            }
            if (deviceUse.owns_lock()) deviceUse.unlock();
            CheckFailure();
            std::uint64_t batchesAtFlip = 0, unsignaledAtFlip = 0;
            Graphics::StorageTexture::NoteFlip();
            if (!drains) {

                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                std::lock_guard gpuLock(GuestMemory::GpuMutex());
                const auto localDevice = device.Load();
                if (localDevice != nullptr) static_cast<void>(Graphics::StorageTexture::StoreDrawnLinear("flip"));

                if (!recordLabelsForPacket(localDevice.get(), submission.queue) && localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0);
                if (localDevice != nullptr) localDevice->FlipBatches(batchesAtFlip, unsignaledAtFlip);
            }
            ++flipsCounted;
            Graphics::NoteHostImportFrame();
            if (batchesAtFlip != 0) flipSerial = batchesAtFlip;
            flipBatchesUnsignaled += unsignaledAtFlip;
            if (Graphics::AbConfigured()) {
                static const auto abClock = std::chrono::steady_clock::now();
                std::vector<std::string> abLines;
                {
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    abLines = Graphics::AbAtPresent(flipsCounted.load(), std::chrono::duration<double>(std::chrono::steady_clock::now() - abClock).count());
                }
                for (const auto& line : abLines) AgcDriver::ProfilePrint_nid_no_patch("%s", line.c_str());
            }

            timing.Mark("flip_flush");
            timing.FinishPacket(submission.serial, cursor, 0xffffu, count);
            const bool queueTiming = APS5_ENABLE_TIMING_LOG && submission.queue == 0;
            auto frame = queueTiming ? std::move(pendingFrameTiming) : std::make_shared<FrameTiming>(++frameSerial);
            require(frame != nullptr, "flip has no frame timing");
            const auto now = FrameTiming::Clock::now();
            if (!queueTiming) frame->IncludeSubmission(submission.serial, now, now, now, true);
            frame->SetFlip(submission.serial, cursor, queueTiming ? submission.receivedAt : now, now);
            if (queueTiming) frame->CollectBackground();
            frame->NoteFlipBatches(batchesAtFlip, unsignaledAtFlip);
            CaptureTrace::Log("flip frame=%llu submission=%llu offset=%zu batch=%llu unsignaled=%llu", static_cast<unsigned long long>(frameSerial.load()), static_cast<unsigned long long>(submission.serial), cursor, static_cast<unsigned long long>(batchesAtFlip), static_cast<unsigned long long>(unsignaledAtFlip));
            flipHold.Release();
            submission.flips.at(cursor)->GpuReady(frame);
        } else if (opcode == 0x15 || opcode == 0x16) {
            packetCounters.CountDispatch();
            landPendingWaits(submission.queue);
            currentPacketOffset() = cursor;
            if (gateOpened()) {
                gateOpened() = false;
                if (!pipelineDispatch() || submission.queue != 0) {
                    if (pipelined && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, opcode);
                    timed(&WorkerProfile::dispatchMs, [&] { resolveGroupAhead(submission, queue, cursor); });
                }
            }
            const bool ringOverlap = submission.queue != 0 && Graphics::Recorder::DispatchOverlapEnabled();
            if (ringOverlap) Graphics::Recorder::SetDispatchOrder(submission.queue + 1u, ringOrder.afterDispatch);
            struct ClearOrder {
                bool active;
                ~ClearOrder() {
                    if (active) Graphics::Recorder::SetDispatchOrder(0, false);
                }
            } clearOrder{ringOverlap};
            if (ringOverlap) ringOrder.afterDispatch = true;
            if (opcode == 0x15) timed(&WorkerProfile::dispatchMs, [&] { dispatch(queue, packet, submission); });
            else timed(&WorkerProfile::dispatchMs, [&] { dispatchIndirect(queue, packet, submission); });
            Pm4::NoteDispatchRegisters(packet, queue);
            Graphics::Recorder::CountRecordedWork();
            finishDispatchPacket(opcode == 0x16);
        } else if (opcode == 0x3c || opcode == 0x93) {
            static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
            const auto waitStart = traceGpu ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            timed(&WorkerProfile::waitMs, [&] { waitMemory(packet, submission.queue, recent, submission.received, submission.heldAtSubmit.contains(cursor)); });
            noteLabelPass(packet, submission.queue);
            gateOpened() = true;
            if (traceGpu && std::chrono::steady_clock::now() - waitStart > std::chrono::milliseconds(200)) {

                for (std::size_t next = cursor + count, shown = 0; next < submission.commands.size() && shown < 48; ++shown) {
                    const auto nextHeader = submission.commands[next];
                    const auto nextCount = Pm4::PacketWords(nextHeader);
                    const auto nextPacket = std::span(submission.commands).subspan(next, nextCount);
                    std::fprintf(stderr, "[gpu]   then %s", Pm4::Name(nextHeader).c_str());
                    for (std::size_t i = 1; i < nextPacket.size() && i < 7; ++i) std::fprintf(stderr, " %08x", nextPacket[i]);
                    std::fprintf(stderr, "\n");
                    next += nextCount;
                }
            }
        } else if (drawPacket) {
            landPendingWaits(submission.queue);
            packetCounters.CountDraw();
            drawnSinceSeal() = true;
            bool drawn = false;
            timed(&WorkerProfile::drawMs, [&] {
                static const bool traceDraws = std::getenv("APS5_TRACE_DRAWS") != nullptr;
                static const bool profileDraws = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                const auto color = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
                const auto started = profileDraws ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                const auto countSkip = [&](Graphics::DrawSkip kind) {
                    if (profileDraws) Graphics::CountDrawSkip(kind, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count());
                };

                try {
                    std::string rejected;
                    const auto verdict = draw(queue, packet, submission, cursor, rejected);
                    drawn = verdict == DrawVerdict::Drawn;
                    CaptureTrace::Log("draw submission=%llu queue=%x offset=%zu target=%llx mask=%x verdict=%d reason=%.256s", static_cast<unsigned long long>(submission.serial), submission.queue, cursor, static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e), static_cast<int>(verdict), rejected.c_str());
                    if (verdict == DrawVerdict::Rejected) {
                        countSkip(Graphics::DrawSkip::Prechecked);
                        throw std::runtime_error(rejected);
                    } else if (verdict == DrawVerdict::Nothing) {
                        countSkip(Graphics::DrawSkip::Nothing);
                    } else if (traceDraws) {
                        std::fprintf(stderr, "[draw] target 0x%llx mask 0x%x ok\n",static_cast<unsigned long long>(color), readRegister(queue.context, 0x8e));
                    }
                } catch (const RefusedDraw& refused) {
                    CaptureTrace::Log("draw-refused submission=%llu offset=%zu reason=%.256s", static_cast<unsigned long long>(submission.serial), cursor, refused.what());
                    countSkip(Graphics::DrawSkip::Thrown);
                    ReportRefusedDraw(refused);
                } catch (const std::exception& error) {
                    CaptureTrace::Log("draw-error submission=%llu offset=%zu reason=%.256s", static_cast<unsigned long long>(submission.serial), cursor, error.what());
                    countSkip(Graphics::DrawSkip::Thrown);
                    throw;
                }
            });
            finishDrawPacket(drawn);
        } else if (opcode == 0x22) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            const auto condition = Pm4::ReadCondition(packet);
            if (condition == 0) nextCursor = submission.conditionalEnds.at(cursor);
            if (traceGpu) std::fprintf(stderr, "[gpu] %.1f queue 0x%x COND_EXEC at DWORD %zu reads 0x%x at 0x%llx: %s %zu dwords\n", TraceMs(), submission.queue, cursor, condition, static_cast<unsigned long long>(packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u)), condition == 0 ? "skips" : "executes", submission.conditionalEnds.at(cursor) - cursor - count);
        } else if (opcode == 0x91) {
            landPendingWaits(submission.queue);
            recordQueuedLabelsBeforeRead(submission.queue);
            Pm4::Execute(packet, queue);
        } else if (sampleDump && !wroteOnGpu) {
            dumpSampleCounters(Pm4::SampleDumpAddress(packet));
        } else if (submission.queue == 0 && sealsWriteOnly(packet, opcode)) {
            if (std::exchange(drawnSinceSeal(), false)) {
                const auto seal = [this] {
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Flush);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    if (const auto localDevice = device.Load()) localDevice->SealPassWriteOnly();
                };
                if (pipelined && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Seal, seal, {});
                else seal();
            }
        } else if (opcode != 0x42 && opcode != 0x46 && opcode != 0x58) {
            if (!wroteOnGpu) {
                if (Pm4::IndirectRegisterOpcode(opcode)) {
                    const auto& listed = submission.registerLists.at(cursor);
                    if (Pm4::IndirectRegistersValid(listed)) {
                        Pm4::ExecuteIndirectRegisters(packet, listed, queue);
                    } else {
                        landPendingWaits(submission.queue);
                        recordQueuedLabelsBeforeRead(submission.queue);
                        const auto current = Pm4::ReadIndirectRegisters(packet);
                        static const bool skipInvalid = std::getenv("APS5_SKIP_INVALID_INDIRECT_REGISTERS") != nullptr;
                        static std::atomic<bool> reported{false};
                        if (!reported.exchange(true)) {
                            std::uint32_t invalid = 0;
                            for (std::size_t i = 0; i < listed.size(); i += 2) {
                                if (!Pm4::IndirectRegistersValid(std::span(&listed[i], 1))) {
                                    invalid = listed[i];
                                    break;
                                }
                            }
                            std::fprintf(stderr, "[gpu] queue 0x%x %s at DWORD %zu: register list entry with offset dword 0x%08x read again where the packet executes%s\n", submission.queue, Pm4::Name(packet[0]).c_str(), cursor, invalid, skipInvalid ? "; invalid entries are skipped (APS5_SKIP_INVALID_INDIRECT_REGISTERS)" : "");
                        }
                        Pm4::ExecuteIndirectRegisters(packet, current, queue, skipInvalid);
                    }
                } else {
                    Pm4::Execute(packet, queue);
                }
                if (opcode == 0x49 || opcode == 0x37) {
                    if (const auto label = Pm4::DecodeLabelWrite(packet)) noteLabelStore(label->address, label->Bytes(), ++eventSerial, submission.queue);
                }
            }
            if (endOfPipeInterrupt && !interruptDeferred) deliverGuestEopInterrupt(submission.queue, packet[7] & 0x0fffffffu);
        }
        if (drawPacket || (sampleDump && wroteOnGpu)) Graphics::Recorder::CountRecordedWork();
        timing.Mark(header == RenderingWaitPacketHeader ? "rendering_wait" : opcode == 0x3c || opcode == 0x93 ? "memory_wait" : opcode == 0x15 || opcode == 0x16 ? "dispatch" : drawPacket ? "draw" : "execute");
        if (submission.queue == 0) timing.FinishPacket(submission.serial, cursor, opcode, count);
        else timing.Finish();
        cursor = nextCursor;
    }

    if (decoderShadow != nullptr && decoderShadow->ReportDue()) std::fprintf(stderr, "%s%c", decoderShadow->Report().c_str(), 10);
    FlushLabelRun();
    const bool queuesEnd = pipelined && orderedCompletion && deferredLabels().labels.empty();
    const bool relaysEnd = queuesEnd && DrawFinisher::RelaysSubmissionEnd();
    const auto rewindReleased = [&] { return submission.rewindTail != nullptr && (std::atomic_ref<std::uint32_t>(*const_cast<std::uint32_t*>(submission.rewindTail - 1)).load(std::memory_order_acquire) & 0x80000000u) != 0; };
    bool rewinds = relaysEnd && rewindReleased();
    if (relaysEnd && (submission.rewindTail == nullptr || rewinds)) {
        if (DrawFinisher::Busy()) DrawFinisher::NoteDrain(DrawFinisher::DrainSite::SubmissionEnd, false);
        DrawFinisher::Get().CheckFailure();
        settleFinisher.queue0 = false;
    } else {
        if (submission.queue == 0) DrawFinisher::Get().DrainAt(DrawFinisher::DrainSite::SubmissionEnd);
        rewinds = queuesEnd && rewindReleased();
    }
    PerformanceContext endContext(submission.queue == 0 ? frameTiming() : FrameTiming::Async());
    PerformanceTimer endTiming("Driver.EndSubmission");
    static const bool submitAtEnd = std::getenv("APS5_SUBMIT_AT_END") != nullptr;
    if (queuesEnd) {
        if (submission.rewindTail == nullptr) {
            flipHold.working = false;
            deferCompletion(submission);
            return;
        }
        if (rewinds) {
            const bool holdsFlip = std::exchange(flipHold.working, false);
            DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::SubmitEnd, [this, holdsFlip, flipWork = submission.flipWork, progress = submission.progress] {
                if (holdsFlip) finishFlipWork(flipWork, submitFlipWork(0));
                else submitOpenWork(progress);
                wakeQueueZeroForCompletions();
            }, {});
            endTiming.Finish();
            executeRewindTail(submission);
            return;
        }
    }
    if (pipelined) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
    if (!deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0) {
        auto& costs = submissionCosts(submission.queue);
        if (!submitAtEnd && submission.rewindTail == nullptr && submission.queue == 0 && workerQueued() != nullptr && workerQueued()->load(std::memory_order_acquire) != 0) {
            ++costs.endSkipped;
            flipHold.Finish();
            return;
        }
        ++costs.endSubmits;
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        const auto localDevice = device.Load();

        recordDeferredLabels(localDevice.get(), submission.queue);
        if (localDevice != nullptr) localDevice->SubmitRecorded(submission.queue == 0);
    }
    flipHold.Finish();
    endTiming.Finish();
    if (submission.rewindTail != nullptr) {
        if (submission.progress) submitOpenWork(submission.progress);
        executeRewindTail(submission);
    }
}

}
