#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/SubmissionRing.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/SynchronizationStatistics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/WaitMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Core/include/SubmitPolicy.hpp"
#include <algorithm>
#include <chrono>
#include <array>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <string>

namespace AgcDriver::DriverDetail {

namespace {

constexpr auto IdleResultsQuietPeriod = std::chrono::milliseconds(50);

std::uint64_t SubmitGroupWork() {
    static const std::uint64_t value = [] {
        const char* text = std::getenv("APS5_SUBMIT_GROUP_WORK");
        return text ? std::strtoull(text, nullptr, 10) : 64ull;
    }();
    return value;
}

std::atomic<std::uint64_t> submitsHeld{0}, heldSubmittedLast{0}, heldSubmittedDue{0}, heldSubmittedWork{0}, heldSubmittedElsewhere{0};

std::chrono::steady_clock::time_point HeldSubmitSince(std::chrono::steady_clock::rep since) {
    return std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(since));
}

constexpr std::size_t SubmitCauses = static_cast<std::size_t>(Core::SubmitCause::Count);
std::array<std::atomic<std::uint64_t>, SubmitCauses> endCauses{}, heldCauses{};
std::atomic<std::uint64_t> heldLongestNs{0}, heldOverdue{0}, heldCheckedNs{0};

std::uint64_t PolicyNs(std::chrono::steady_clock::time_point time) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
    return ns > 0 ? static_cast<std::uint64_t>(ns) : 1u;
}

Core::SubmitState PolicyState(std::chrono::steady_clock::time_point now, bool held, std::chrono::steady_clock::rep heldSince, std::size_t queued) {
    Core::SubmitState state;
    state.now = PolicyNs(now);
    if (const auto pending = Graphics::Recorder::PendingLabelSince(); pending.has_value()) state.oldestLabel = PolicyNs(*pending);
    if (held) state.heldSince = PolicyNs(HeldSubmitSince(heldSince));
    state.work = static_cast<std::uint32_t>(std::min<std::uint64_t>(Graphics::Recorder::RecordedWorkSinceSubmit(), std::numeric_limits<std::uint32_t>::max()));
    state.queued = static_cast<std::uint32_t>(std::min<std::size_t>(queued, std::numeric_limits<std::uint32_t>::max()));
    state.waiter = BlockedReaders().load(std::memory_order_acquire) != 0;
    return state;
}

void AddGpuState(Core::SubmitState& state, const std::shared_ptr<VulkanDevice>& localDevice) {
    if (!SubmitEarly() || localDevice == nullptr || VulkanDevice::OpenGpuLoadHint() < SubmitEarlyLoad()) return;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    state.gpuLoad = localDevice->OpenGpuLoad();
    state.gpuIdle = state.gpuLoad >= SubmitEarlyLoad() && localDevice->GpuIdle();
}

Core::SubmitLimits PolicyLimits(std::chrono::microseconds deadline, std::uint64_t cap) {
    Core::SubmitLimits limits;
    if (SubmitEarly()) limits.idleLoad = SubmitEarlyLoad();
    limits.deadline = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline).count());
    limits.groupWork = static_cast<std::uint32_t>(std::min<std::uint64_t>(SubmitGroupWork(), std::numeric_limits<std::uint32_t>::max()));
    limits.workCap = static_cast<std::uint32_t>(std::min<std::uint64_t>(cap, std::numeric_limits<std::uint32_t>::max()));
    return limits;
}

void ReportOffDraw(std::chrono::steady_clock::time_point now) {
    static const bool reported = std::getenv("APS5_PIPELINE_STATS") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr || VerifyCoreOffDraw();
    if (!reported) return;
    static std::chrono::steady_clock::time_point last = now;
    if (now - last < std::chrono::seconds(10)) return;
    last = now;
    static std::array<std::uint64_t, SubmitCauses> reportedEnds{}, reportedHeld{};
    static VulkanDevice::OffDrawCount reportedCounts{};
    static std::uint64_t reportedOverdue = 0;
    std::string ends;
    std::string held;
    for (std::size_t i = 0; i < SubmitCauses; ++i) {
        const auto name = Core::SubmitCauseName(static_cast<Core::SubmitCause>(i));
        const auto end = endCauses[i].load(std::memory_order_relaxed);
        const auto later = heldCauses[i].load(std::memory_order_relaxed);
        if (end != reportedEnds[i]) ends += " " + std::string(name) + " " + std::to_string(end - reportedEnds[i]);
        if (later != reportedHeld[i]) held += " " + std::string(name) + " " + std::to_string(later - reportedHeld[i]);
        reportedEnds[i] = end;
        reportedHeld[i] = later;
    }
    const auto counts = VulkanDevice::OffDrawCounts();
    const auto overdue = heldOverdue.load(std::memory_order_relaxed);
    const auto since = [](std::uint64_t current, std::uint64_t before) { return static_cast<unsigned long long>(current - before); };
    AgcDriver::ProfilePrint_nid_no_patch("[core-off-draw] (10 s): submission ends by cause:%s; held batches submitted by cause:%s; longest hold %.1f us, %llu kept by a check past the deadline; labels %llu, %llu first in a block of their batch, %llu repeated, %llu past the table; verify: %llu repeated marks collected, %llu over a CPU write; label reaps skipped %llu, verify: %llu made anyway, %llu retired a batch; %llu batches submitted early to an idle GPU\n",
                                         ends.empty() ? " none" : ends.c_str(), held.empty() ? " none" : held.c_str(), static_cast<double>(heldLongestNs.exchange(0, std::memory_order_relaxed)) / 1e3, since(overdue, reportedOverdue),
                                         since(counts.labels, reportedCounts.labels), since(counts.firstMarks, reportedCounts.firstMarks), since(counts.repeatedMarks, reportedCounts.repeatedMarks), since(counts.overflowedMarks, reportedCounts.overflowedMarks),
                                         since(counts.verifiedMarks, reportedCounts.verifiedMarks), since(counts.marksOverWrites, reportedCounts.marksOverWrites), since(counts.reapsSkipped, reportedCounts.reapsSkipped), since(counts.reapsVerified, reportedCounts.reapsVerified), since(counts.reapsLate, reportedCounts.reapsLate), static_cast<unsigned long long>(EarlySubmits().exchange(0, std::memory_order_relaxed)));
    reportedCounts = counts;
    reportedOverdue = overdue;
}

}

FrameTiming* Driver::frameTiming() {
    if (!APS5_ENABLE_TIMING_LOG) return nullptr;
    if (pendingFrameTiming == nullptr) pendingFrameTiming = std::make_shared<FrameTiming>(++frameSerial, true);
    return pendingFrameTiming.get();
}

FrameTiming* Driver::includeTimingSubmission(const Submission& submission, bool firstSegment) {
    if (!APS5_ENABLE_TIMING_LOG) return nullptr;
    auto* frame = submission.queue == 0 ? frameTiming() : FrameTiming::Async();
    require(submission.receivedAt != FrameTiming::Clock::time_point{} && submission.receivedAt <= submission.enqueuedAt && submission.enqueuedAt <= submission.dequeuedAt && submission.dequeuedAt <= submission.orderedAt, "invalid submission timing order");
    if (submission.queue == 0) frame->IncludeSubmission(submission.serial, submission.receivedAt, submission.enqueuedAt, submission.orderedAt, firstSegment);
    else if (firstSegment) {
        frame->Add(frame->Get("Submission", "accept"), submission.enqueuedAt - submission.receivedAt);
        frame->Add(frame->Get("Submission", "queue"), submission.orderedAt - submission.enqueuedAt);
    }
    if (firstSegment) {
        frame->Add(frame->Get("Submission", "dequeue"), submission.dequeuedAt - submission.enqueuedAt);
        frame->Add(frame->Get("Submission", "order_wait"), submission.orderedAt - submission.dequeuedAt);
        if (!submission.suspend) {
            frame->Add(frame->Get("Submission", "copy"), submission.copiedAt - submission.receivedAt, submission.commands.size() * sizeof(std::uint32_t));
            frame->Add(frame->Get("Submission", "validate"), submission.validatedAt - submission.copiedAt);
            frame->Add(frame->Get("Submission", "flip_room_wait"), submission.roomReadyAt - submission.validatedAt);
            frame->Add(frame->Get("Submission", "reserve_enqueue"), submission.enqueuedAt - submission.roomReadyAt);
        }
    }
    return frame;
}

void Driver::markCompleted(std::uint64_t serial) {
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) std::fprintf(stderr, "[gpu] %.1f completed serial=%llu (in order up to %llu)\n", TraceMs(), static_cast<unsigned long long>(serial), static_cast<unsigned long long>(completed));
    completedOutOfOrder.insert(serial);
    while (!completedOutOfOrder.empty() && *completedOutOfOrder.begin() == completed + 1) {
        completed = *completedOutOfOrder.begin();
        completedOutOfOrder.erase(completedOutOfOrder.begin());
    }
}

bool& Driver::completionDeferred() {
    static thread_local bool deferred = false;
    return deferred;
}

void Driver::deferCompletion(const Submission& submission) {
    std::uint64_t received = submission.received;
    {
        std::lock_guard lock(mutex);
        if (queue0Executing != 0) received = std::min(received, queue0Executing);
        queue0Uncommitted.push_back(received);
    }
    completionDeferred() = true;
    static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
    if (traceGpu) std::fprintf(stderr, "[gpu] %.1f deferred completion serial=%llu to the draw pipeline\n", TraceMs(), static_cast<unsigned long long>(submission.serial));
    DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Completion, [this, serial = submission.serial, received, labelWrites = submission.labelWrites, progress = submission.progress, flipWork = submission.holdsFlip ? submission.flipWork : nullptr] {
        completeSubmission(serial, received, labelWrites, progress, flipWork);
    }, {});
}

bool Driver::holdSubmit() {
    if (!CoreOffDraw() || DrawPipeline::Flat()) return false;
    const auto now = std::chrono::steady_clock::now();
    const bool held = heldSubmit.load(std::memory_order_acquire);
    const auto outstanding = DrawPipeline::Queue0().Outstanding();
    auto state = PolicyState(now, held, heldSince.load(std::memory_order_acquire), outstanding > 1 ? outstanding - 1 : 0);
    AddGpuState(state, device.Load());
    const auto cause = Core::AtSubmissionEnd(state, PolicyLimits(labelFlushDeadline(), batchCap()));
    endCauses[static_cast<std::size_t>(cause)].fetch_add(1, std::memory_order_relaxed);
    ReportOffDraw(now);
    if (cause != Core::SubmitCause::Held) return false;
    if (!held) {
        heldCheckedNs.store(0, std::memory_order_relaxed);
        heldSince.store(now.time_since_epoch().count(), std::memory_order_release);
        heldSubmit.store(true, std::memory_order_release);
    }
    submitsHeld.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void Driver::SubmitHeldWork(bool last) {
    const bool work = !deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0;
    if (!work) {
        if (heldSubmit.exchange(false, std::memory_order_acq_rel)) heldSubmittedElsewhere.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto since = HeldSubmitSince(heldSince.load(std::memory_order_acquire));
    auto state = PolicyState(now, true, heldSince.load(std::memory_order_acquire), last ? 0 : 1);
    AddGpuState(state, device.Load());
    const auto cause = Core::AfterOp(state, PolicyLimits(labelFlushDeadline(), batchCap()));
    const auto heldNs = static_cast<std::uint64_t>(std::max<std::chrono::nanoseconds::rep>(0, std::chrono::duration_cast<std::chrono::nanoseconds>(now - since).count()));
    const auto deadlineNs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(labelFlushDeadline()).count());
    if (cause == Core::SubmitCause::Held) {
        heldCheckedNs.store(heldNs, std::memory_order_relaxed);
        return;
    }
    if (heldCheckedNs.exchange(0, std::memory_order_relaxed) >= deadlineNs) heldOverdue.fetch_add(1, std::memory_order_relaxed);
    heldCauses[static_cast<std::size_t>(cause)].fetch_add(1, std::memory_order_relaxed);
    const bool due = cause == Core::SubmitCause::Deadline;
    const bool full = cause == Core::SubmitCause::Work || cause == Core::SubmitCause::Cap;
    if (heldNs > heldLongestNs.load(std::memory_order_relaxed)) heldLongestNs.store(heldNs, std::memory_order_relaxed);
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    const auto localDevice = device.Load();
    recordDeferredLabels(localDevice.get(), 0);
    if (localDevice != nullptr) localDevice->SubmitRecorded(true);
    heldSubmit.store(false, std::memory_order_release);
    (due ? heldSubmittedDue : full ? heldSubmittedWork : heldSubmittedLast).fetch_add(1, std::memory_order_relaxed);
}

Driver::SubmitGroupCounts Driver::SubmitGroupCounted() {
    return SubmitGroupCounts{submitsHeld.load(std::memory_order_relaxed), heldSubmittedLast.load(std::memory_order_relaxed), heldSubmittedDue.load(std::memory_order_relaxed), heldSubmittedWork.load(std::memory_order_relaxed), heldSubmittedElsewhere.load(std::memory_order_relaxed)};
}

void Driver::submitOpenWork(const std::shared_ptr<SubmissionProgress>& progress, bool mayHold) {
    const bool work = !deferredLabels().labels.empty() || Graphics::Recorder::PendingLabelSince().has_value() || Graphics::Recorder::RecordedWorkSinceSubmit() != 0;
    if (!work && !progress) return;
    if (mayHold && work && !progress && holdSubmit()) return;
    ++submissionCosts(0).endSubmits;
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    const auto localDevice = device.Load();
    if (work) {
        recordDeferredLabels(localDevice.get(), 0);
        if (localDevice != nullptr) localDevice->SubmitRecorded(true);
        if (mayHold) heldSubmit.store(false, std::memory_order_release);
    }
    if (progress) progress->Publish(localDevice, localDevice ? localDevice->SubmittedEpoch() : 0);
}

void Driver::wakeQueueZeroForCompletions() {
    if (!completionsPending()) return;
    std::lock_guard lock(mutex);
    changed.notify_all();
}

std::uint64_t Driver::submitFlipWork(std::uint32_t queue) {
    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::End);
    std::lock_guard gpuLock(GuestMemory::GpuMutex());
    const auto localDevice = device.Load();
    recordDeferredLabels(localDevice.get(), queue);
    return localDevice != nullptr ? localDevice->SubmitAndEpoch() : 0;
}

void Driver::completeSubmission(std::uint64_t serial, std::uint64_t received, std::span<const std::uint64_t> labelWrites, const std::shared_ptr<SubmissionProgress>& progress, const std::shared_ptr<FlipWork>& flipWork) {
    struct Complete {
        Driver& driver;
        std::uint64_t serial, received;
        std::span<const std::uint64_t> labelWrites;
        ~Complete() {
            bool notify = true;
            {
                std::lock_guard lock(driver.mutex);
                driver.markCompleted(serial);
                driver.forgetUnfinishedWrites(driver.workers.at(0), labelWrites);
                const auto found = std::find(driver.queue0Uncommitted.begin(), driver.queue0Uncommitted.end(), received);
                if (found != driver.queue0Uncommitted.end()) driver.queue0Uncommitted.erase(found);
                notify = driver.idleWaiters != 0 || driver.orderHolders.load(std::memory_order_acquire) != 0 || completionsPending();
            }
            if (notify) driver.changed.notify_all();
        }
    } complete{*this, serial, received, labelWrites};
    if (flipWork != nullptr) finishFlipWork(flipWork, submitFlipWork(0));
    else submitOpenWork(progress, true);
}

const std::atomic<std::uint64_t>*& Driver::workerQueued() {
    static thread_local const std::atomic<std::uint64_t>* queued = nullptr;
    return queued;
}

std::atomic<std::uint64_t>*& Driver::workerPacket() {
    static thread_local std::atomic<std::uint64_t>* packet = nullptr;
    return packet;
}

void Driver::reapCompletionLabels() {
    std::unique_lock gpuLock(GuestMemory::GpuMutex(), std::try_to_lock);
    if (!gpuLock.owns_lock()) {
        ++triesFailed[TryIdle];
        return;
    }

    if (!completionsPending()) return;

    bumpEpoch(&EpochBumps::reaps);
    if (const auto localDevice = device.Load()) localDevice->ReapRecorded();
}

void Driver::run(std::uint32_t id) noexcept {
    onWorkerThread() = true;
    char role[32];
    std::snprintf(role, sizeof role, "queue worker 0x%x", id);
    PinWorkerThread(role);
    char threadName[16];
    std::snprintf(threadName, sizeof threadName, "qworker 0x%x", id);
    NameDriverThread(threadName);

    GuestMemory::TagGpuLockThread(id);
    StartWorkerSampler(id);
    Submission submission;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static const bool idleStore = std::getenv("APS5_NO_IDLE_STORE") == nullptr;
    auto& costs = submissionCosts(id);
    try {
        PerformanceTrace::BindThread(role);
        for (;;) {
            submission = Submission{};
            struct Running {
                std::atomic<std::uint32_t>& count;
                std::atomic<std::uint64_t>& idleEpoch;
                std::condition_variable& changed;
                bool active = false;
                Running(std::atomic<std::uint32_t>& count, std::atomic<std::uint64_t>& idleEpoch, std::condition_variable& changed)
                    : count(count), idleEpoch(idleEpoch), changed(changed) {}
                void Start() {
                    count.fetch_add(1, std::memory_order_acq_rel);
                    active = true;
                }
                ~Running() {
                    if (active && count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        idleEpoch.fetch_add(1, std::memory_order_release);
                        changed.notify_all();
                    }
                }
            } running{runningWorkers, idleEpoch, changed};
            static const bool traceGpu = std::getenv("APS5_TRACE_GPU") != nullptr;
            {
                PerformanceContext timingContext(id == 0 ? frameTiming() : FrameTiming::Async());
                PerformanceTimer timing("Driver.WorkerWait");
                std::unique_lock lock(mutex);
                timing.Mark("mutex_wait");
                auto& worker = workers.at(id);
                auto& pending = worker.pending;
                workerQueued() = &worker.queued;
                workerPacket() = &worker.executingPacket;
                if (traceGpu && pending.empty()) std::fprintf(stderr, "[gpu] %.1f idle queue=0x%x\n", TraceMs(), id);
                const auto ready = [&] { return failure || stopping || !pending.empty(); };
                const auto allQueuesIdle = [&] {
                    return runningWorkers.load(std::memory_order_acquire) == 0
                        && std::all_of(workers.begin(), workers.end(), [](const auto& item) { return item.second.pending.empty(); });
                };
                const auto idleResultsPending = [&] {
                    return idleStore && id == 0 && Graphics::StorageTexture::AnyPending() && allQueuesIdle() && !completionsPending();
                };
                const auto poll = [&](const auto& done) {
#ifdef _WIN32
                    lock.unlock();
                    PollSleep();
                    lock.lock();
                    return done();
#else
                    return changed.wait_for(lock, std::chrono::milliseconds(1), done);
#endif
                };

                while (!ready()) {
                    if (!completionsPending() || (id != 0 && Graphics::Recorder::PendingCompletionLabels() == 0)) {
                        const auto idleEpochAtCheck = idleEpoch.load(std::memory_order_acquire);
                        const bool storeIdleResults = idleResultsPending();
                        if (storeIdleResults) {
                            queue0Dormant.store(true, std::memory_order_relaxed);
                            const auto quietUntil = std::chrono::steady_clock::now() + IdleResultsQuietPeriod;
                            const auto idleEpochAtWait = idleEpoch.load(std::memory_order_acquire);
                            changed.wait_until(lock, quietUntil, [&] {
                                return ready() || idleEpoch.load(std::memory_order_acquire) != idleEpochAtWait;
                            });
                            queue0Dormant.store(false, std::memory_order_relaxed);
                            if (ready()) break;
                            if (idleEpoch.load(std::memory_order_acquire) != idleEpochAtWait) {
                                continue;
                            }
                            if (!idleResultsPending()) continue;
                        }
                        lock.unlock();
                        if (storeIdleResults) {
                            if (const auto localDevice = device.Load()) localDevice->StoreIdleResults(true, [&] { return worker.queued.load(std::memory_order_acquire) != 0 || runningWorkers.load(std::memory_order_acquire) != 0; });
                        }
                        lock.lock();
                        if (ready()) break;
                        if (id == 0 && allQueuesIdle() && !completionsPending()
                            && ((idleStore && !storeIdleResults && Graphics::StorageTexture::AnyPending()) || idleEpoch.load(std::memory_order_acquire) != idleEpochAtCheck)) continue;
                        if (id == 0) queue0Dormant.store(true, std::memory_order_relaxed);
                        const auto idleEpochAtWait = idleEpoch.load(std::memory_order_acquire);
                        changed.wait(lock, [&] { return ready() || (id == 0 && (idleEpoch.load(std::memory_order_acquire) != idleEpochAtWait || completionsPending())); });
                        if (id == 0) queue0Dormant.store(false, std::memory_order_relaxed);
                        continue;
                    }
                    if (poll(ready)) break;
                    lock.unlock();
                    reapCompletionLabels();
                    lock.lock();
                }
                timing.Mark("idle_and_completion_poll");
                rethrowFailure();
                if (stopping || shutdownToken.stop_requested() || pending.empty()) {
                    break;
                }
                submission = std::move(pending.front());
                pending.pop_front();
                worker.executingPacket.store(0, std::memory_order_relaxed);
                if (APS5_ENABLE_TIMING_LOG) submission.dequeuedAt = std::chrono::steady_clock::now();
                if (id == 0) queue0Executing = submission.suspend ? 0 : submission.received;
                struct OrderHold {
                    std::atomic<std::uint32_t>& holders;
                    explicit OrderHold(std::atomic<std::uint32_t>& holders) : holders(holders) { holders.fetch_add(1, std::memory_order_acq_rel); }
                    ~OrderHold() { holders.fetch_sub(1, std::memory_order_acq_rel); }
                    OrderHold(const OrderHold&) = delete;
                    OrderHold& operator=(const OrderHold&) = delete;
                };
                if (id == 0 && !submission.suspend) {
                    {
                        OrderHold hold{orderHolders};
                        const auto released = [&] { return failure || stopping || !waitFreeBefore(submission.received); };
                        StallWatch stall(id);
                        while (!released() && !poll(released)) stall.Check("order behind other queues for event", submission.received, &lock);
                    }
                    rethrowFailure();
                }
                if (id != 0) {
                    worker.executingWaitFree = submission.waitFree ? submission.received : 0;
                    worker.executingWaits = !submission.waitFree && !submission.suspend;
                }
                if (submission.waitFree) {
                    {
                        OrderHold hold{orderHolders};
                        const auto released = [&] { return failure || stopping || orderReleased(id, submission.received); };
                        StallWatch stall(id);
                        while (!released() && !poll(released)) stall.Check("order behind queue 0 for event", submission.received, &lock);
                    }
                    rethrowFailure();
                }
                if (APS5_ENABLE_TIMING_LOG) submission.orderedAt = std::chrono::steady_clock::now();
                timing.Mark("dequeue_order");
                worker.queued.fetch_sub(1, std::memory_order_acq_rel);
                running.Start();
                if (profile && submission.enqueuedAt != std::chrono::steady_clock::time_point{}) costs.dequeueNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - submission.enqueuedAt).count());
            }
            retireLabelTakes(id, submission.received);
            execute(submission);
            if (submission.progress && submission.rewindTail == nullptr && !completionDeferred()) submitOpenWork(submission.progress);
            if (traceGpu) std::fprintf(stderr, "[gpu] %.1f done serial=%llu queue=0x%x\n", TraceMs(), static_cast<unsigned long long>(submission.serial), id);
            const auto completeStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            PerformanceContext completionContext(id == 0 ? frameTiming() : FrameTiming::Async());
            PerformanceTimer completionTiming("Driver.Completion");
            bool notify = true;
            {
                std::lock_guard lock(mutex);
                rethrowFailure();
                if (!std::exchange(completionDeferred(), false)) {
                    markCompleted(submission.serial);
                    forgetUnfinishedWrites(workers.at(id), submission);
                }
                if (id == 0) queue0Executing = 0;
                if (id != 0) {
                    workers.at(id).executingWaitFree = 0;
                    workers.at(id).executingWaits = false;
                }

                notify = idleWaiters != 0 || orderHolders.load(std::memory_order_acquire) != 0;
            }
            if (notify) changed.notify_all();
            else ++costs.notifiesSkipped;
            if (profile) costs.completeNs += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - completeStart).count());
        }
    } catch (const ProcessShutdown&) {
        submission = Submission{};
    } catch (...) {
        const auto error = std::current_exception();
        for (const auto& [offset, flip] : submission.flips) flip->Fail(error);
        ReportFailure(error);
    }
}

}
