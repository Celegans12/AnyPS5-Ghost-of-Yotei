#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawSites.hpp"
#include "prx/libc/include/HostThread.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/MovedRead.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/HandoffSpin.hpp"
#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileCounter.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/WorkerSampler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::DrainReason::Count)> DrainNames{"packet", "flush", "labels", "capture", "indirect", "submission"};

constexpr std::array<const char*, static_cast<std::size_t>(DrawPipeline::CommitKind::Count)> KindNames{"other", "draw", "indirect", "colormeta", "dispatch", "sampledump", "dmafill", "dmawrite", "dmacopy", "store", "endofpipe", "seal", "submitend", "completion", "bufferfill"};

std::atomic<unsigned> runningCommitters{0};
ProfileCounter drawKeys;
ProfileCounter drawKeysSameAsPrevious;
ProfileCounter drawKeysSeenRecently;
ProfileCounter drawShapesSameAsPrevious;
ProfileCounter drawShapesSeenRecently;

std::uint64_t elapsedNs(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

ProfileCounter heldReadRanges;
ProfileCounter heldReadQueries;
ProfileCounter heldReadHits;

template<typename TItem>
struct ClearOnExit {
    std::vector<TItem>& items;
    ~ClearOnExit() { items.clear(); }
};

}

namespace {
thread_local bool finisherThread = false;
thread_local bool finisherProducer = false;
thread_local DrawPipeline::Order runningOrder{};

DrawPipeline::Order nextOrder() {
    static std::atomic<std::uint64_t> producers{0};
    thread_local std::uint64_t producer = 0;
    thread_local std::uint64_t sequence = 0;
    if (producer == 0) producer = producers.fetch_add(1, std::memory_order_relaxed) + 1;
    return {producer, ++sequence};
}
}

DrawFinisher& DrawFinisher::Get() {
    static DrawFinisher* finisher = new DrawFinisher();
    return *finisher;
}

bool DrawFinisher::Enabled() {
    return Graphics::DrawPipelineEnabled();
}

bool DrawPipeline::MetadataPassesPipelined() {
    return Graphics::DrawPipelineEnabled();
}

namespace {
ProfileCounter pipelinedMetadataPasses;
}

std::uint64_t DrawPipeline::PipelinedMetadataPasses() {
    return pipelinedMetadataPasses.Sum();
}

void DrawPipeline::CountPipelinedMetadataPass() {
    pipelinedMetadataPasses.Add(1);
}

bool DrawFinisher::Overlap() {
    return Graphics::DrawPipelineEnabled();
}

bool DrawFinisher::Relays() {
    static const bool relays = Overlap() && std::getenv("APS5_NO_FINISHER_RELAY") == nullptr;
    return relays;
}

bool DrawFinisher::RelaysSubmissionEnd() {
    static const bool relays = Relays() && std::getenv("APS5_NO_RELAYED_SUBMISSION_END") == nullptr;
    return relays;
}

bool DrawFinisher::PendingOverlaps(std::uint64_t begin, std::uint64_t end) {
    if (!Busy() || !finisherProducer) return false;
    std::lock_guard lock(mutex);
    for (const auto& held : pending) {
        if (held.unknown) return true;
        for (const auto& [first, last] : held.writes) {
            if (begin < last && first < end) return true;
        }
    }
    return false;
}

DrawFinisher::LabelLookup DrawFinisher::PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t& value) {
    if (!Busy() || !finisherProducer) return LabelLookup::NotHeld;
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (auto held = pending.rbegin(); held != pending.rend(); ++held) {
        if (held->unknown) return LabelLookup::Unknown;
        const bool overlaps = std::any_of(held->writes.begin(), held->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (held->labelBytes.empty() || address < held->labelAddress || end > held->labelAddress + held->labelBytes.size()) return LabelLookup::Unknown;
        value = 0;
        std::memcpy(&value, held->labelBytes.data() + (address - held->labelAddress), bytes);
        return LabelLookup::Value;
    }
    return LabelLookup::NotHeld;
}

bool DrawFinisher::PendingOverlapsAny(std::span<const Range> ranges) {
    if (!Busy() || !finisherProducer) return false;
    std::lock_guard lock(mutex);
    for (const auto& held : pending) {
        if (held.unknown) return true;
        for (const auto& [first, last] : held.writes) {
            for (const auto& [begin, end] : ranges) {
                if (begin < last && first < end) return true;
            }
        }
    }
    return false;
}

bool DrawFinisher::PendingKey(std::uint64_t key, std::uint64_t shape) {
    if (!Busy() || !finisherProducer) return false;
    std::lock_guard lock(mutex);
    return std::any_of(pending.begin(), pending.end(), [&](const Pending& held) { return held.key == key || (shape != 0 && held.shape == shape); });
}

namespace {
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawFinisher::DrainSite::Count)> finisherDrains{}, finisherSkips{}, finisherDrainsEver{}, finisherSkipsEver{}, finisherSiteWaitNs{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawPipeline::CommitKind::Count)> itemDrains{}, itemRelays{};
std::atomic<std::uint64_t> finisherWaits{0}, finisherWaitNs{0}, finisherRoomWaits{0}, finisherRoomWaitNs{0};
}

void DrawFinisher::NoteDrain(DrainSite site, bool drained) {
    (drained ? finisherDrains : finisherSkips)[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
    (drained ? finisherDrainsEver : finisherSkipsEver)[static_cast<std::size_t>(site)].fetch_add(1, std::memory_order_relaxed);
}

void DrawFinisher::NoteItemDrain(DrawPipeline::CommitKind kind) {
    itemDrains[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

void DrawFinisher::NoteItemRelay(DrawPipeline::CommitKind kind) {
    itemRelays[static_cast<std::size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t DrawFinisher::DrainCount(DrainSite site, bool drained) {
    return (drained ? finisherDrainsEver : finisherSkipsEver)[static_cast<std::size_t>(site)].load(std::memory_order_relaxed);
}

namespace {
std::array<std::atomic<std::uint64_t>, 257> drainedOpcodes{}, drainedBeside{};
}

void DrawFinisher::NoteDrainedOpcode(std::uint32_t opcode, bool besideCommits) {
    drainedOpcodes[std::min<std::uint32_t>(opcode, 256u)].fetch_add(1, std::memory_order_relaxed);
    if (besideCommits) drainedBeside[std::min<std::uint32_t>(opcode, 256u)].fetch_add(1, std::memory_order_relaxed);
}

std::string DrawFinisher::DrainText() {
    constexpr std::array<const char*, static_cast<std::size_t>(DrainSite::Count)> names{"lookup", "reads", "packet", "kept draw", "submission end", "item"};
    std::string text = "; worker drains of the finisher (drained/skipped):";
    for (std::size_t i = 0; i < names.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu/%llu", names[i], static_cast<unsigned long long>(finisherDrains[i].exchange(0)), static_cast<unsigned long long>(finisherSkips[i].exchange(0)));
        text += item;
    }
    text += "; drained by opcode (of them beside commits):";
    for (int shown = 0; shown < 6; ++shown) {
        std::size_t top = 0;
        for (std::size_t i = 1; i < drainedOpcodes.size(); ++i) {
            if (drainedOpcodes[i].load() > drainedOpcodes[top].load()) top = i;
        }
        const auto count = drainedOpcodes[top].exchange(0);
        if (count == 0) break;
        char item[48];
        std::snprintf(item, sizeof(item), " 0x%02zx x%llu (%llu)", top, static_cast<unsigned long long>(count), static_cast<unsigned long long>(drainedBeside[top].exchange(0)));
        text += item;
    }
    const auto milliseconds = [](std::atomic<std::uint64_t>& nanoseconds) { return static_cast<double>(nanoseconds.exchange(0)) / 1e6; };
    char waits[256];
    std::snprintf(waits, sizeof(waits), "; worker waits on the finisher: %llu drains (%.1f ms), of them submission end (%.1f ms) item (%.1f ms), %llu for room (%.1f ms); items that drained:", static_cast<unsigned long long>(finisherWaits.exchange(0)), milliseconds(finisherWaitNs), milliseconds(finisherSiteWaitNs[static_cast<std::size_t>(DrainSite::SubmissionEnd)]), milliseconds(finisherSiteWaitNs[static_cast<std::size_t>(DrainSite::Item)]), static_cast<unsigned long long>(finisherRoomWaits.exchange(0)), milliseconds(finisherRoomWaitNs));
    text += waits;
    const auto kinds = [&text](std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrawPipeline::CommitKind::Count)>& counts) {
        for (std::size_t i = 0; i < counts.size(); ++i) {
            const auto count = counts[i].exchange(0);
            if (count == 0) continue;
            char item[48];
            std::snprintf(item, sizeof(item), " %s %llu", KindNames[i], static_cast<unsigned long long>(count));
            text += item;
        }
    };
    kinds(itemDrains);
    text += "; items relayed:";
    kinds(itemRelays);
    return text;
}

bool DrawFinisher::OnFinisherThread() {
    return finisherThread;
}

void DrawFinisher::Enqueue(Job job) {
    Enqueue(std::move(job), Pending{});
}

void DrawFinisher::Enqueue(Job job, Pending held) {
    finisherProducer = true;
    const auto epoch = GuestMemory::HeldCollectEpoch();
    const auto order = nextOrder();
    constexpr std::size_t depth = 16;
    struct ReclaimStorage {};
    auto& reclaim = HostThreadLocal<std::vector<Job>, ReclaimStorage>();
    const ClearOnExit<Job> clearReclaim{reclaim};
    std::unique_lock lock(mutex);
    if (!ran.empty()) std::swap(ran, reclaim);
    if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
    if (jobs.size() >= depth && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher full while the enqueuing thread holds the GPU lock");
    if (jobs.size() >= depth) {
        const auto start = std::chrono::steady_clock::now();
        idle.wait(lock, [&] { return jobs.size() < depth; });
        finisherRoomWaits.fetch_add(1, std::memory_order_relaxed);
        finisherRoomWaitNs.fetch_add(elapsedNs(start), std::memory_order_relaxed);
    }
    jobs.push_back(std::move(job));
    epochs.push_back(epoch);
    orders.push_back(order);
    pending.push_back(std::move(held));
    outstanding.fetch_add(1, std::memory_order_release);
    if (!running) {
        running = true;
        thread = std::thread([this] {
            finisherThread = true;
            run();
        });
    }
    if (sleeping) wake.notify_one();
}

void DrawFinisher::run() {
    NameDriverThread("draw-finish");
    PerformanceTrace::BindThread("draw finisher");
    ReserveFinisherCore();
    StartFinisherSampler();
    for (;;) {
        Job* job = nullptr;
        bool failed = false;
        std::uint64_t epoch = 0;
        SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        {
            std::unique_lock lock(mutex);
            sleeping = true;
            wake.wait(lock, [&] { return !jobs.empty(); });
            sleeping = false;
            job = &jobs.front();
            epoch = epochs.front();
            runningOrder = orders.front();
            failed = failure != nullptr;
        }
        if (DrawPipeline::SharedCollectEpoch()) GuestMemory::AdoptCollectEpoch(epoch);
        std::exception_ptr error;
        bool shutdown = false;
        try {
            if (!failed) (*job)();
        } catch (const ProcessShutdown&) {
            shutdown = true;
            error = std::current_exception();
        } catch (...) {
            error = std::current_exception();
        }
        if (error != nullptr && !shutdown && RelaysSubmissionEnd()) Driver::Get().ReportFailure(error);
        std::lock_guard lock(mutex);
        if (error != nullptr && failure == nullptr) failure = error;
        ran.push_back(std::move(jobs.front()));
        jobs.pop_front();
        epochs.pop_front();
        orders.pop_front();
        pending.pop_front();
        outstanding.fetch_sub(1, std::memory_order_release);
        idle.notify_all();
    }
}

void DrawFinisher::Settle() {
    if (!Busy() || OnFinisherThread()) return;
    if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher settle under the GPU lock");
    std::unique_lock lock(mutex);
    idle.wait(lock, [&] { return jobs.empty(); });
}

void DrawFinisher::Drain() {
    if (OnFinisherThread()) return;
    const bool waits = Busy();
    if (waits && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw finisher drain under the GPU lock");
    const auto start = waits ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    SpinUntil(HandoffSpinWindow() * 2, [&] { return outstanding.load(std::memory_order_acquire) == 0; });
    std::unique_lock lock(mutex);
    if (!jobs.empty()) idle.wait(lock, [&] { return jobs.empty(); });
    if (waits) {
        finisherWaits.fetch_add(1, std::memory_order_relaxed);
        finisherWaitNs.fetch_add(elapsedNs(start), std::memory_order_relaxed);
    }
    if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
}

void DrawFinisher::CheckFailure() {
    std::lock_guard lock(mutex);
    if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
}

void DrawFinisher::DrainAt(DrainSite site) {
    struct Waited {
        DrainSite site;
        bool waits;
        std::chrono::steady_clock::time_point start;
        ~Waited() {
            if (!waits) return;
            NoteDrain(site, true);
            finisherSiteWaitNs[static_cast<std::size_t>(site)].fetch_add(elapsedNs(start), std::memory_order_relaxed);
        }
    };
    const bool waits = Busy() && !OnFinisherThread();
    const Waited waited{site, waits, waits ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
    Drain();
}

void DrawFinisher::ReleaseRan() {
    std::vector<Job> done;
    std::lock_guard lock(mutex);
    std::swap(done, ran);
}

CollectWalker& CollectWalker::Get() {
    static CollectWalker* walker = new CollectWalker();
    return *walker;
}

bool CollectWalker::Enabled() {
    static const bool enabled = std::getenv("APS5_NO_COLLECT_WALKER") == nullptr;
    return enabled;
}

bool CollectWalker::Offer(std::uint64_t epoch, std::uint64_t address, std::uint64_t bytes) {
    if (closed.load(std::memory_order_acquire)) {
        rethrowFailure();
        dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const auto at = tail.load(std::memory_order_relaxed);
    if (at - knownHead >= Capacity) knownHead = head.load(std::memory_order_acquire);
    if (at - knownHead >= Capacity) {
        dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    ring[at % Capacity] = {address, bytes, epoch};
    tail.store(at + 1, std::memory_order_seq_cst);
    handed.fetch_add(1, std::memory_order_relaxed);
    if (!started.load(std::memory_order_acquire)) start();
    if (parked.load(std::memory_order_seq_cst) != 0) {
        parked.store(0, std::memory_order_seq_cst);
        WakeOne(parked);
    }
    return true;
}

bool CollectWalker::Idle() const {
    return closed.load(std::memory_order_acquire) || head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
}

CollectWalker::Counts CollectWalker::Counted() const {
    return {handed.load(std::memory_order_relaxed), collected.load(std::memory_order_relaxed), dropped.load(std::memory_order_relaxed), collectNs.load(std::memory_order_relaxed), sleeps.load(std::memory_order_relaxed)};
}

void CollectWalker::start() {
    std::lock_guard lock(lifecycle);
    if (started.load(std::memory_order_relaxed) || closed.load(std::memory_order_relaxed)) return;
    try {
        thread = Libc::StartHostThread([this] { run(); });
        started.store(true, std::memory_order_release);
    } catch (const std::system_error& error) {
        closed.store(true, std::memory_order_seq_cst);
        std::fprintf(stderr, "[collects] no host thread for the collect walker, the committer keeps its walks: %s\n", error.what());
    }
}

void CollectWalker::run() {
    NameDriverThread("collect-walk");
    PerformanceTrace::BindThread("collect walker");
    static const bool timed = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_PIPELINE_STATS") != nullptr;
    std::exception_ptr error;
    try {
        while (!closed.load(std::memory_order_acquire)) {
            const auto at = head.load(std::memory_order_relaxed);
            if (tail.load(std::memory_order_acquire) == at) {
                if (SpinUntil(HandoffSpinWindow(), [&] { return tail.load(std::memory_order_acquire) != at || closed.load(std::memory_order_acquire); })) continue;
                parked.store(1u, std::memory_order_seq_cst);
                if (tail.load(std::memory_order_seq_cst) == at && !closed.load(std::memory_order_seq_cst)) {
                    sleeps.fetch_add(1, std::memory_order_relaxed);
                    WaitWhile(parked, 1u);
                }
                parked.store(0u, std::memory_order_seq_cst);
                continue;
            }
            const auto entry = ring[at % Capacity];
            GuestMemory::AdoptCollectEpoch(entry.epoch);
            const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            GuestMemory::CollectWrites(entry.address, static_cast<std::size_t>(entry.bytes));
            if (timed) collectNs.fetch_add(elapsedNs(began), std::memory_order_relaxed);
            collected.fetch_add(1, std::memory_order_relaxed);
            head.store(at + 1, std::memory_order_release);
        }
    } catch (const ProcessShutdown&) {
    } catch (const std::exception& cause) {
        error = std::make_exception_ptr(std::runtime_error(std::string("collect walker: ") + cause.what()));
    } catch (...) {
        error = std::current_exception();
    }
    if (error != nullptr) {
        std::lock_guard lock(lifecycle);
        failure = error;
        failed.store(true, std::memory_order_release);
    }
    closed.store(true, std::memory_order_seq_cst);
}

void CollectWalker::rethrowFailure() {
    if (!failed.load(std::memory_order_acquire)) return;
    std::exception_ptr error;
    {
        std::lock_guard lock(lifecycle);
        error = std::exchange(failure, nullptr);
        failed.store(false, std::memory_order_release);
    }
    if (error != nullptr) std::rethrow_exception(error);
}

void CollectWalker::Stop() {
    std::thread walker;
    {
        std::lock_guard lock(lifecycle);
        closed.store(true, std::memory_order_seq_cst);
        walker = std::move(thread);
    }
    parked.store(0, std::memory_order_seq_cst);
    WakeOne(parked);
    if (walker.joinable()) walker.join();
}

DrawPipeline& DrawPipeline::Queue0() {
    static DrawPipeline* pipeline = new DrawPipeline();
    return *pipeline;
}

std::size_t DrawPipeline::Depth() {
    static const std::size_t depth = [] {
        const char* text = std::getenv("APS5_PIPELINED_DRAWS");
        if (std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DRAW_DRAIN") != nullptr || std::getenv("APS5_DRAIN_ALL") != nullptr || std::getenv("APS5_NO_WORDWISE_CAPTURE") != nullptr) return std::size_t{0};
        if (text == nullptr) return std::size_t{32};
        const auto value = std::strtoull(text, nullptr, 10);
        return value == 0 ? std::size_t{0} : value == 1 ? std::size_t{8} : static_cast<std::size_t>(std::min<unsigned long long>(value, 256));
    }();
    return depth;
}

std::size_t DrawPipeline::QueuedItems() {
    auto& pipeline = Queue0();
    std::lock_guard lock(pipeline.mutex);
    return pipeline.items.size();
}

bool& DrawPipeline::Active() {
    static thread_local bool active = false;
    return active;
}

std::atomic<std::uint64_t>& DrawPipeline::EpochToken() {
    static std::atomic<std::uint64_t> token{1};
    return token;
}

void DrawPipeline::FollowEpoch(std::uint64_t token) {
    if (SharedCollectEpoch()) return;
    static thread_local std::uint64_t followed = 0;
    if (followed == token) return;
    followed = token;
    GuestMemory::BumpCollectEpoch();
}

bool DrawPipeline::SharedCollectEpoch() {
    static const bool shared = std::getenv("APS5_NO_SHARED_COLLECT_EPOCH") == nullptr;
    return shared;
}

bool DrawPipeline::VerifiesOrder() {
    static const bool verifies = std::getenv("APS5_VERIFY_PIPELINE_ORDER") != nullptr;
    return verifies;
}

bool DrawPipeline::CommitOrder::Accept(Order next) {
    const auto found = std::find_if(last.begin(), last.end(), [&](const Order& seen) { return seen.producer == next.producer; });
    if (found == last.end()) {
        last.push_back(next);
        return true;
    }
    if (next.sequence < found->sequence) return false;
    found->sequence = next.sequence;
    return true;
}

unsigned DrawPipeline::Committers() {
    return runningCommitters.load(std::memory_order_acquire);
}

void DrawPipeline::rethrowFailure() {
    if (failure == nullptr) return;
    auto error = std::exchange(failure, nullptr);
    std::rethrow_exception(error);
}

void DrawPipeline::Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, bool draw) {
    enqueue({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), std::move(reads), draw});
}

void DrawPipeline::Enqueue(CommitKind kind, Commit commit, std::vector<Range> writes, std::uint64_t labelAddress, std::vector<std::byte> labelBytes, std::vector<Range> reads, bool draw) {
    enqueue({std::move(commit), std::move(writes), labelAddress, std::move(labelBytes), std::move(reads), draw, false, kind});
}

void DrawPipeline::NoteDrawKey(std::uint64_t key, std::uint64_t shape) {
    constexpr std::size_t RecentSlots = 4096;
    thread_local std::array<std::uint64_t, RecentSlots> recent{};
    thread_local std::uint64_t previous = 0;
    thread_local std::array<std::uint64_t, RecentSlots> recentShapes{};
    thread_local std::uint64_t previousShape = 0;
    const auto shapeSlot = static_cast<std::size_t>((shape ^ (shape >> 29u)) % RecentSlots);
    const auto slot = static_cast<std::size_t>((key ^ (key >> 29u)) % RecentSlots);
    drawKeys.Add(1);
    if (key == previous) drawKeysSameAsPrevious.Add(1);
    if (recent[slot] == key) drawKeysSeenRecently.Add(1);
    if (shape == previousShape) drawShapesSameAsPrevious.Add(1);
    if (recentShapes[shapeSlot] == shape) drawShapesSeenRecently.Add(1);
    recent[slot] = key;
    previous = key;
    recentShapes[shapeSlot] = shape;
    previousShape = shape;
}

void DrawPipeline::EnqueueStore(Commit commit, std::uint64_t address, std::vector<std::byte> bytes) {
    const auto end = address + bytes.size();
    enqueue({std::move(commit), {{address, end}}, address, std::move(bytes), {}, false, true, CommitKind::Store});
}

void DrawPipeline::enqueue(Item item) {
    item.collectEpoch = GuestMemory::HeldCollectEpoch();
    if (DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread()) {
        if (DrawFinisher::Relays() && item.reads.empty()) {
            relay(std::move(item));
            return;
        }
        DrawFinisher::NoteItemDrain(item.kind);
        DrawFinisher::Get().DrainAt(DrawFinisher::DrainSite::Item);
    }
    push(std::move(item));
}

void DrawPipeline::relay(Item item) {
    const auto kind = item.kind;
    DrawFinisher::Pending held;
    held.unknown = false;
    held.writes = item.writes;
    held.labelAddress = item.labelAddress;
    held.labelBytes = item.labelBytes;
    auto carried = std::make_shared<Item>(std::move(item));
    DrawFinisher::Get().Enqueue([this, carried] { push(std::move(*carried)); }, std::move(held));
    DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Item, false);
    DrawFinisher::NoteItemRelay(kind);
}

void DrawPipeline::push(Item item) {
    item.order = DrawFinisher::OnFinisherThread() ? runningOrder : nextOrder();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    struct ReclaimStorage {};
    auto& reclaim = HostThreadLocal<std::vector<Item>, ReclaimStorage>();
    const ClearOnExit<Item> clearReclaim{reclaim};
    const auto lockStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::unique_lock lock(mutex);
    if (!retired.empty()) std::swap(retired, reclaim);
    if (profile) lockWaitNs += elapsedNs(lockStart);
    rethrowFailure();
    if (closing) {
        lock.unlock();
        item.commit();
        return;
    }
    if (items.size() >= Depth()) {
        if (GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline full while the enqueuing thread holds the GPU lock");
        const auto start = std::chrono::steady_clock::now();
        resumeAt = Depth() - 1;
        ++idleWaiters;
        idle.wait(lock, [&] { return items.size() <= resumeAt; });
        --idleWaiters;
        resumeAt = 0;
        fullWaitNs += elapsedNs(start);
        rethrowFailure();
    }
    depthSum += items.size();
    ++enqueued;
    heldReadRanges.Add(item.reads.size());
    countWrites(item.writes, true);
    items.push_back(std::move(item));
    outstanding.fetch_add(1, std::memory_order_release);
    if (!thread.joinable()) {
        thread = Libc::StartHostThread([this] {
            GuestMemory::MarkProofThread();
            run();
        });
        runningCommitters.fetch_add(1, std::memory_order_acq_rel);
    }
    if (committerWaiting) {
        ++wakes;
        wake.notify_one();
    }
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr;
    if (!profile && !stats) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport >= std::chrono::seconds(10)) report(now);
}

void DrawPipeline::report(std::chrono::steady_clock::time_point now) {
    lastReport = now;
    std::string drainText;
    for (std::size_t i = 0; i < drains.size(); ++i) {
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f ms)", DrainNames[i], static_cast<unsigned long long>(drains[i]), static_cast<double>(drainWaitNs[i]) / 1e6);
        drainText += item;
    }
    drainText += "; packet drains by opcode:";
    for (int shown = 0; shown < 5; ++shown) {
        const auto top = std::max_element(drainOpcodes.begin(), drainOpcodes.begin() + 0x100);
        if (*top == 0) break;
        char item[32];
        std::snprintf(item, sizeof(item), " 0x%02x x%llu", static_cast<unsigned>(top - drainOpcodes.begin()), static_cast<unsigned long long>(*top));
        drainText += item;
        *top = 0;
    }
    drainOpcodes = {};
    const auto perCommit = commits == 0 ? 0.0 : static_cast<double>(commitNs) / 1e3 / static_cast<double>(commits);
    const auto perDraw = enqueued == 0 ? 0.0 : static_cast<double>(fullWaitNs) / 1e3 / static_cast<double>(enqueued);
    const auto queued = enqueued == 0 ? 0.0 : static_cast<double>(depthSum) / static_cast<double>(enqueued);
    const auto decodeDrainCount = decodeDrains.load(std::memory_order_relaxed);
    char reads[96];
    std::snprintf(reads, sizeof(reads), " %llu read ranges, %llu stores held, %llu decode drains", static_cast<unsigned long long>(heldReadRanges.Sum()), static_cast<unsigned long long>(heldReadHits.Take()), static_cast<unsigned long long>(decodeDrainCount - reportedDecodeDrains));
    reportedDecodeDrains = decodeDrainCount;
    AgcDriver::ProfilePrint_nid_no_patch("[draw] pipelined (10 s, depth %zu): %llu commits at %.1f us each on the committer, %llu failed; %llu enqueued behind %.2f on average, waiting %.1f us per draw for room, %.2f us for the lock; %llu committer wakes, idle %.1f ms in %llu waits;%s; drains:%s\n", Depth(), static_cast<unsigned long long>(commits), perCommit, static_cast<unsigned long long>(commitErrors), static_cast<unsigned long long>(enqueued), queued, perDraw, enqueued == 0 ? 0.0 : static_cast<double>(lockWaitNs) / 1e3 / static_cast<double>(enqueued), static_cast<unsigned long long>(wakes), static_cast<double>(idleNs) / 1e6, static_cast<unsigned long long>(idleWaits), reads, drainText.c_str());
    std::string kindText;
    for (std::size_t i = 0; i < kindCommits.size(); ++i) {
        const auto count = kindCommits[i].exchange(0, std::memory_order_relaxed);
        const auto ns = kindNs[i].exchange(0, std::memory_order_relaxed);
        if (count == 0) continue;
        char item[64];
        std::snprintf(item, sizeof(item), " %s %llu (%.1f us)", KindNames[i], static_cast<unsigned long long>(count), static_cast<double>(ns) / 1e3 / static_cast<double>(count));
        kindText += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[commits] by kind (10 s):%s\n", kindText.c_str());
    const auto vk = Graphics::TakeVkDrawCounts();
    const auto sites = Graphics::TakeDrawSiteCounts();
    AgcDriver::ProfilePrint_nid_no_patch("[vkdraw] (10 s): %llu vkCmdDraw, %llu vkCmdDrawIndexed, %llu vkCmdDrawIndirect calls, %llu mesh; draw keys %llu, same as the previous draw %llu, seen within the last 4096 draws %llu; draw shapes (state without per-draw user words) same as the previous %llu, seen within the last 4096 %llu; recorded draws %llu, %llu with a shape, %llu of a shape seen before, %llu of them with the shape's last resources template, pipeline and framebuffer\n", static_cast<unsigned long long>(vk.direct), static_cast<unsigned long long>(vk.indexed), static_cast<unsigned long long>(vk.indirect), static_cast<unsigned long long>(vk.mesh), static_cast<unsigned long long>(drawKeys.Take()), static_cast<unsigned long long>(drawKeysSameAsPrevious.Take()), static_cast<unsigned long long>(drawKeysSeenRecently.Take()), static_cast<unsigned long long>(drawShapesSameAsPrevious.Take()), static_cast<unsigned long long>(drawShapesSeenRecently.Take()), static_cast<unsigned long long>(sites.draws), static_cast<unsigned long long>(sites.shaped), static_cast<unsigned long long>(sites.seen), static_cast<unsigned long long>(sites.stable));
    static GuestMemory::CollectCounts reportedCollects{};
    const auto collects = GuestMemory::CollectCounted();
    static EpochBumps reportedBumps{};
    const auto bumps = EpochBumpsCounted();
    AgcDriver::ProfilePrint_nid_no_patch("[collects] (10 s): %llu GetWriteWatch calls, %llu of them on the committer; %llu collect epochs; %llu collect-memo hits; %llu epochs with a memoized collect that walked; epoch bumps by reason: submissions %llu, waits %llu (%llu more ordered by an earlier wait, %llu by a flag stored before the epoch, after %llu collects of awaited pages), drains %llu, interrupts %llu, queued label interrupts %llu, reaps %llu, packets %llu\n", static_cast<unsigned long long>(collects.watchCalls - reportedCollects.watchCalls), static_cast<unsigned long long>(collects.committerWatchCalls - reportedCollects.committerWatchCalls), static_cast<unsigned long long>(collects.epochBumps - reportedCollects.epochBumps), static_cast<unsigned long long>(collects.memoHits - reportedCollects.memoHits), static_cast<unsigned long long>(collects.walkedEpochs - reportedCollects.walkedEpochs), static_cast<unsigned long long>(bumps.submissions - reportedBumps.submissions), static_cast<unsigned long long>(bumps.waits - reportedBumps.waits), static_cast<unsigned long long>(bumps.waitsOrdered - reportedBumps.waitsOrdered), static_cast<unsigned long long>(bumps.waitsBeforeEpoch - reportedBumps.waitsBeforeEpoch), static_cast<unsigned long long>(bumps.awaitedPageCollects - reportedBumps.awaitedPageCollects), static_cast<unsigned long long>(bumps.drains - reportedBumps.drains), static_cast<unsigned long long>(bumps.interrupts - reportedBumps.interrupts), static_cast<unsigned long long>(bumps.queuedInterrupts - reportedBumps.queuedInterrupts), static_cast<unsigned long long>(bumps.reaps - reportedBumps.reaps), static_cast<unsigned long long>(bumps.packets - reportedBumps.packets));
    static CollectWalker::Counts reportedWalker{};
    const auto walker = CollectWalker::Get().Counted();
    AgcDriver::ProfilePrint_nid_no_patch("[collect-walker] (10 s): %llu ranges handed over, %llu collected by it in %.1f ms, %llu left to the committer, %llu sleeps\n", static_cast<unsigned long long>(walker.handed - reportedWalker.handed), static_cast<unsigned long long>(walker.collected - reportedWalker.collected), static_cast<double>(walker.collectNs - reportedWalker.collectNs) / 1e6, static_cast<unsigned long long>(walker.dropped - reportedWalker.dropped), static_cast<unsigned long long>(walker.sleeps - reportedWalker.sleeps));
    reportedCollects = collects;
    reportedBumps = bumps;
    reportedWalker = walker;
    const auto dirty = GuestMemory::TakeDirtyBlockCounts();
    const auto share = [](std::uint64_t part, std::uint64_t whole) { return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole); };
    AgcDriver::ProfilePrint_nid_no_patch("[dirty] (10 s): %llu distinct 64 KiB blocks found written by the CPU, %llu of them in more than one collect epoch, %llu (block, epoch) findings; collects %.1f MiB, %.1f %% in blocks written in this or the previous 10 s; variant validations %.1f MiB, %.1f %% in such blocks\n", static_cast<unsigned long long>(dirty.distinct), static_cast<unsigned long long>(dirty.recurrent), static_cast<unsigned long long>(dirty.findings), static_cast<double>(dirty.collectBytes) / 1048576.0, share(dirty.collectDirtyBytes, dirty.collectBytes), static_cast<double>(dirty.variantBytes) / 1048576.0, share(dirty.variantDirtyBytes, dirty.variantBytes));
    static Graphics::DrawProofCounts reportedProofs{};
    const auto proofs = Graphics::DrawProofCounted();
    static GuestMemory::ProofBreakCounts reportedBreaks{};
    const auto breaks = GuestMemory::ProofBreaksCounted();
    std::string breakText;
    for (std::size_t i = 0; i < breaks.breaks.size(); ++i) {
        const auto count = breaks.breaks[i] - reportedBreaks.breaks[i];
        const auto wide = breaks.wide[i] - reportedBreaks.wide[i];
        const auto missed = breaks.serialMisses[i] - reportedBreaks.serialMisses[i];
        if (count == 0 && missed == 0) continue;
        char item[128];
        std::snprintf(item, sizeof(item), " %s %llu (%llu of them wide, %llu serial misses after it)", GuestMemory::ProofBreakName(static_cast<GuestMemory::ProofBreak>(i)), static_cast<unsigned long long>(count), static_cast<unsigned long long>(wide), static_cast<unsigned long long>(missed));
        breakText += item;
    }
    static GuestMemory::WriteIndexStats reportedIndex{};
    const auto index = GuestMemory::WriteIndexCounted();
    constexpr std::array<const char*, static_cast<std::size_t>(GuestMemory::WriteBumpSource::Count)> bumpNames{"cpu", "driver", "pending", "unwatch", "forget", "release", "import", "break", "registry"};
    std::string bumpText;
    for (std::size_t i = 0; i < bumpNames.size(); ++i) {
        const auto count = index.bySource[i] - reportedIndex.bySource[i];
        if (count == 0) continue;
        char item[48];
        std::snprintf(item, sizeof(item), " %s %llu", bumpNames[i], static_cast<unsigned long long>(count));
        bumpText += item;
    }
    AgcDriver::ProfilePrint_nid_no_patch("[proofs] (10 s): %s; breaks:%s; subscriptions %llu over %llu blocks, %llu bumps (%s ) in %llu lookups (%.1f ms), %llu sweeps, %llu refused\n", Graphics::DrawProofText(proofs, reportedProofs).c_str(), breakText.c_str(), static_cast<unsigned long long>(index.subscriptions - reportedIndex.subscriptions), static_cast<unsigned long long>(index.blocks - reportedIndex.blocks), static_cast<unsigned long long>(index.bumps - reportedIndex.bumps), bumpText.c_str(), static_cast<unsigned long long>(index.lookups - reportedIndex.lookups), static_cast<double>(index.lookupNanoseconds - reportedIndex.lookupNanoseconds) / 1e6, static_cast<unsigned long long>(index.sweeps - reportedIndex.sweeps), static_cast<unsigned long long>(index.sweepFailures - reportedIndex.sweepFailures));
    reportedIndex = index;
    reportedProofs = proofs;
    const auto depth = Graphics::TakeDepthViewCounts();
    AgcDriver::ProfilePrint_nid_no_patch("[depth-view] (10 s): %llu depth views; depth plane syncs %llu, %llu of them found nothing written; stencil syncs %llu, %llu found nothing; HTILE store-back checks %llu, %llu returned no clear; skipped by depth proofs: %llu depth plane syncs, %llu stencil syncs, %llu HTILE checks, %llu memory notes\n", static_cast<unsigned long long>(depth.calls), static_cast<unsigned long long>(depth.depthSyncs), static_cast<unsigned long long>(depth.quietDepthSyncs), static_cast<unsigned long long>(depth.stencilSyncs), static_cast<unsigned long long>(depth.quietStencilSyncs), static_cast<unsigned long long>(depth.htileChecks), static_cast<unsigned long long>(depth.quietHtileChecks), static_cast<unsigned long long>(depth.depthSkips), static_cast<unsigned long long>(depth.stencilSkips), static_cast<unsigned long long>(depth.htileSkips), static_cast<unsigned long long>(depth.noteSkips));
    reportedBreaks = breaks;
    const auto contentRecipes = DriverDetail::ContentRecipesHeld();
    AgcDriver::ProfilePrint_nid_no_patch("%s", VulkanDevice::DrawRecipeDrawsReport(contentRecipes.held, contentRecipes.evicted).c_str());
    AgcDriver::ProfilePrint_nid_no_patch("%s", MovedReadsReport().c_str());
    static ShaderRecompiler::DescriptorLeafCounts reportedLeaves;
    const auto leaves = ShaderRecompiler::DescriptorLeafTotals();
    AgcDriver::ProfilePrint_nid_no_patch("[descriptor-leaves] (10 s): %llu stage captures over %llu buffers, %llu captures traced for %llu planned leaf dwords: %llu leaves traced, %llu left out as aliased\n", static_cast<unsigned long long>(leaves.captures - reportedLeaves.captures), static_cast<unsigned long long>(leaves.buffers - reportedLeaves.buffers), static_cast<unsigned long long>(leaves.traced - reportedLeaves.traced), static_cast<unsigned long long>(leaves.planned - reportedLeaves.planned), static_cast<unsigned long long>(leaves.leaves - reportedLeaves.leaves), static_cast<unsigned long long>(leaves.aliased - reportedLeaves.aliased));
    reportedLeaves = leaves;
    static BufferWordCounts reportedWords;
    const auto bufferWords = BufferWordTotals();
    const auto wordsSince = [](std::uint64_t now, std::uint64_t before) { return static_cast<unsigned long long>(now - before); };
    AgcDriver::ProfilePrint_nid_no_patch("[buffer-words] (10 s): of %llu descriptor leaves of the stages kept, %llu became volatile positions (%llu outside the stage's runs, %llu bound to no buffer element, %llu of a written element, %llu not the bound word); of %llu descriptor dwords held in user words, %llu tagged; %llu stages reused with %llu table words patched, %llu refused for another specialization\n", wordsSince(bufferWords.leaves, reportedWords.leaves), wordsSince(bufferWords.positions, reportedWords.positions), wordsSince(bufferWords.unmapped, reportedWords.unmapped), wordsSince(bufferWords.unbound, reportedWords.unbound), wordsSince(bufferWords.written, reportedWords.written), wordsSince(bufferWords.mismatched, reportedWords.mismatched), wordsSince(bufferWords.userLeaves, reportedWords.userLeaves), wordsSince(bufferWords.userTagged, reportedWords.userTagged), wordsSince(bufferWords.stages, reportedWords.stages), wordsSince(bufferWords.words, reportedWords.words), wordsSince(bufferWords.refused, reportedWords.refused));
    reportedWords = bufferWords;
    if (DrawSiteAccess::Enabled()) AgcDriver::ProfilePrint_nid_no_patch("%s", DrawSiteAccess::Report().c_str());
    static GuestMemory::LockCounts reportedLocks{};
    const auto locks = GuestMemory::LockCounted();
    AgcDriver::ProfilePrint_nid_no_patch("[locks] (10 s): %s\n", GuestMemory::LockReport(reportedLocks, locks).c_str());
    reportedLocks = locks;
    commits = commitNs = commitErrors = enqueued = fullWaitNs = depthSum = wakes = lockWaitNs = idleWaits = idleNs = 0;
    drains = {};
    drainWaitNs = {};
}


void DrawPipeline::breakProofsAfter(CommitKind kind, const std::vector<Range>& writes) {
    for (const auto& [begin, end] : writes) {
        if (end > begin) GuestMemory::BreakProofs(GuestMemory::ProofBreak::Commit, begin, static_cast<std::size_t>(end - begin));
    }
    const bool storesThroughDriver = kind == CommitKind::Seal || kind == CommitKind::SubmitEnd || kind == CommitKind::Completion || kind == CommitKind::EndOfPipe;
    if (writes.empty() && !storesThroughDriver) GuestMemory::BreakProofs(GuestMemory::ProofBreak::Commit);
}

void DrawPipeline::run() {
    NameDriverThread("draw-commit");
    PerformanceTrace::BindThread("draw committer");
    ReserveCommitterCore();
    StartCommitterSampler();
    static const bool timed = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_PIPELINE_STATS") != nullptr;
    CommitOrder committed;
    for (;;) {
        Commit* commit = nullptr;
        bool isDraw = false;
        std::uint64_t epoch = 0;
        Order order;
        const std::vector<Range>* writes = nullptr;
        auto kind = CommitKind::Other;
        const bool starved = timed && outstanding.load(std::memory_order_acquire) == 0;
        const auto idleStart = starved ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        SpinUntil(HandoffSpinWindow(), [&] { return outstanding.load(std::memory_order_acquire) != 0; });
        {
            std::unique_lock lock(mutex);
            committerWaiting = true;
            wake.wait(lock, [&] { return !items.empty() || closing; });
            committerWaiting = false;
            if (items.empty()) break;
            if (starved) {
                ++idleWaits;
                idleNs += elapsedNs(idleStart);
            }
            commit = stopped ? nullptr : &items.front().commit;
            isDraw = items.front().draw;
            epoch = items.front().collectEpoch;
            writes = &items.front().writes;
            kind = items.front().kind;
            order = items.front().order;
        }
        if (SharedCollectEpoch()) GuestMemory::AdoptCollectEpoch(epoch);
        const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        std::exception_ptr fatal;
        bool failed = false;
        bool shutdown = false;
        try {
            if (VerifiesOrder() && !committed.Accept(order)) throw std::runtime_error("draw pipeline: the committer reached an item its thread queued before one already committed (APS5_VERIFY_PIPELINE_ORDER)");
            if (commit != nullptr) (*commit)();
            if (!isDraw) breakProofsAfter(kind, *writes);
        } catch (const std::exception&) {
            failed = true;
            fatal = std::current_exception();
        } catch (const ProcessShutdown&) {
            shutdown = true;
            fatal = std::current_exception();
        } catch (...) {
            fatal = std::current_exception();
        }
        const auto spent = timed ? elapsedNs(start) : 0u;
        if (fatal != nullptr && !stopped) {
            stopped = true;
            if (!shutdown) Driver::Get().ReportFailure(fatal);
        }
        std::lock_guard lock(mutex);
        countWrites(items.front().writes, false);
        const auto kindIndex = static_cast<std::size_t>(items.front().kind);
        kindCommits[kindIndex].fetch_add(1, std::memory_order_relaxed);
        kindNs[kindIndex].fetch_add(spent, std::memory_order_relaxed);
        retired.push_back(std::move(items.front()));
        items.pop_front();
        retiredItems.fetch_add(1, std::memory_order_release);
        ++commits;
        commitNs += spent;
        if (failed) ++commitErrors;
        if (fatal != nullptr && failure == nullptr) failure = fatal;
        outstanding.fetch_sub(1, std::memory_order_release);
        if (idleWaiters != 0 && (items.empty() || items.size() <= resumeAt)) idle.notify_all();
    }
    runningCommitters.fetch_sub(1, std::memory_order_acq_rel);
}

void DrawPipeline::Stop() {
    std::thread committer;
    {
        std::lock_guard lock(mutex);
        closing = true;
        committer = std::move(thread);
    }
    wake.notify_all();
    if (!committer.joinable()) return;
    if (committer.get_id() == std::this_thread::get_id()) committer.detach();
    else committer.join();
}

void DrawPipeline::ReleaseRetired() {
    std::vector<Item> done;
    std::lock_guard lock(mutex);
    std::swap(done, retired);
}

void DrawPipeline::Drain(DrainReason reason, std::uint32_t opcode) {
    drainRequests[std::min<std::uint32_t>(opcode, 0x100)].fetch_add(1, std::memory_order_relaxed);
    reasonRequests[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    const bool finisherBusy = DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread();
    if ((finisherBusy || Busy()) && GuestMemory::GpuMutex().HeldByThisThread()) throw std::runtime_error("draw pipeline drain under the GPU lock (reason " + std::to_string(static_cast<unsigned>(reason)) + ", opcode " + std::to_string(opcode) + ")");
    if (finisherBusy) DrawFinisher::Get().Drain();
    if (!Busy()) return;
    const auto start = std::chrono::steady_clock::now();
    SpinUntil(HandoffSpinWindow() * 2, [&] { return !Busy(); });
    std::unique_lock lock(mutex);
    ++idleWaiters;
    idle.wait(lock, [&] { return items.empty(); });
    --idleWaiters;
    const auto index = static_cast<std::size_t>(reason);
    ++drains[index];
    ++drainOpcodes[std::min<std::uint32_t>(opcode, 0x100)];
    drainWaitNs[index] += elapsedNs(start);
    rethrowFailure();
}

std::uint64_t DrawPipeline::DrainRequestsByOpcode(std::uint32_t opcode) {
    return Queue0().drainRequests[std::min<std::uint32_t>(opcode, 0x100)].load(std::memory_order_relaxed);
}

bool DrawPipeline::HeldReads(std::uint64_t address, std::size_t bytes) {
    static const bool unordered = std::getenv("APS5_NO_QUEUED_DRAW_READ_ORDER") != nullptr;
    if (bytes == 0 || unordered) return false;
    const auto end = address + bytes;
    auto& pipeline = Queue0();
    std::lock_guard lock(pipeline.mutex);
    heldReadQueries.Add(1);
    for (const auto& item : pipeline.items) {
        for (const auto& [begin, limit] : item.reads) {
            if (address >= limit || begin >= end) continue;
            heldReadHits.Add(1);
            return true;
        }
    }
    return false;
}

std::uint64_t DrawPipeline::HeldReadQueries() {
    return heldReadQueries.Sum();
}

void DrawPipeline::ReleaseRunningReads() {
    std::lock_guard lock(mutex);
    if (!items.empty()) items.front().reads.clear();
}

std::uint64_t DrawPipeline::DrainRequestsByReason(DrainReason reason) {
    return Queue0().reasonRequests[static_cast<std::size_t>(reason)].load(std::memory_order_relaxed);
}

std::optional<std::uint64_t> DrawPipeline::PendingLabel(std::uint64_t address, std::size_t bytes) {
    if (bytes == 0 || bytes > 8) return std::nullopt;
    if (DrawFinisher::Overlap()) {
        std::uint64_t held = 0;
        const auto lookup = DrawFinisher::Get().PendingLabel(address, bytes, held);
        if (lookup == DrawFinisher::LabelLookup::Value) return held;
        if (lookup == DrawFinisher::LabelLookup::Unknown) return std::nullopt;
    }
    const auto end = address + bytes;
    std::lock_guard lock(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        const bool overlaps = std::any_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; });
        if (!overlaps) continue;
        if (item->labelBytes.empty() || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return std::nullopt;
        std::uint64_t value = 0;
        std::memcpy(&value, item->labelBytes.data() + (address - item->labelAddress), bytes);
        return value;
    }
    return std::nullopt;
}

DrawPipeline::QueuedWrite DrawPipeline::StoredBytes(std::uint64_t address, std::size_t bytes, std::span<std::byte> out) {
    const auto end = address + bytes;
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlaps(address, end)) return QueuedWrite::Unknown;
    auto answer = QueuedWrite::None;
    std::lock_guard lock(mutex);
    for (auto item = items.rbegin(); item != items.rend(); ++item) {
        if (std::none_of(item->writes.begin(), item->writes.end(), [&](const Range& range) { return address < range.second && range.first < end; })) continue;
        if (!item->store) return QueuedWrite::Unknown;
        if (out.empty()) {
            answer = QueuedWrite::Stored;
            continue;
        }
        if (out.size() != bytes || address < item->labelAddress || end > item->labelAddress + item->labelBytes.size()) return QueuedWrite::Unknown;
        std::memcpy(out.data(), item->labelBytes.data() + (address - item->labelAddress), bytes);
        return QueuedWrite::Stored;
    }
    return answer;
}

void DrawPipeline::countWrites(const std::vector<Range>& writes, bool added) {
    for (const auto& range : writes) {
        const auto found = std::find_if(queuedWrites.begin(), queuedWrites.end(), [&](const auto& entry) { return entry.first == range; });
        if (added) {
            if (found != queuedWrites.end()) ++found->second;
            else queuedWrites.emplace_back(range, 1u);
        } else if (found != queuedWrites.end() && --found->second == 0) {
            *found = queuedWrites.back();
            queuedWrites.pop_back();
        }
    }
}

DrawPipeline::Watch DrawPipeline::WatchWrites() {
    if (!Busy()) return {Retired(), 0, {}};
    std::lock_guard lock(mutex);
    Watch watch{retiredItems.load(std::memory_order_relaxed), items.size(), {}};
    for (const auto& item : items) watch.writes.insert(watch.writes.end(), item.writes.begin(), item.writes.end());
    return watch;
}

bool DrawPipeline::WrittenSince(const Watch& watch, const std::vector<Range>& reads) {
    const auto overlaps = [&reads](const std::vector<Range>& writes) {
        return std::any_of(reads.begin(), reads.end(), [&writes](const Range& read) {
            return std::any_of(writes.begin(), writes.end(), [&read](const Range& write) { return read.first < write.second && write.first < read.second; });
        });
    };
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(reads)) return true;
    std::lock_guard lock(mutex);
    if (std::any_of(items.begin(), items.end(), [&](const Item& item) { return overlaps(item.writes); })) return true;
    const auto retiredNow = retiredItems.load(std::memory_order_relaxed);
    if (retiredNow == watch.retired) return false;
    return retiredNow - watch.retired > watch.items || overlaps(watch.writes);
}

bool DrawPipeline::Busy() const {
    return outstanding.load(std::memory_order_acquire) != 0 || (finisherProducer && DrawFinisher::Busy() && DrawFinisher::Overlap());
}

bool DrawPipeline::OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(ranges)) return true;
    std::lock_guard lock(mutex);
    for (const auto& [range, count] : queuedWrites) {
        for (const auto& [address, end] : ranges) {
            if (address < range.second && range.first < end) return true;
        }
    }
    return false;
}

bool DrawPipeline::Overlaps(std::uint64_t address, std::size_t bytes) {
    const auto end = address + bytes;
    if (DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlaps(address, end)) return true;
    std::lock_guard lock(mutex);
    for (const auto& [range, count] : queuedWrites) {
        if (address < range.second && range.first < end) return true;
    }
    return false;
}

}
