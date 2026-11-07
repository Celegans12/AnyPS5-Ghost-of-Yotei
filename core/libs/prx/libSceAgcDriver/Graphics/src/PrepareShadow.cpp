#include "prx/libSceAgcDriver/Graphics/include/PrepareShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PrepareStamp.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace AgcDriver::Graphics::PrepareShadow {

namespace {

struct Capture {
    GuestMemory::PrepareStamp stamp;
    std::uint64_t generation = 0;
    std::shared_ptr<const Recorder::WriteRanges> pending;
};

struct Shared {
    std::mutex mutex;
    Capture enqueued;
    Capture prepared;
    bool hasPrepared = false;
    bool committing = false;
};

Capture take() {
    Capture capture;
    capture.stamp = GuestMemory::PrepareStamp::Take();
    capture.generation = GuestMemory::TrackerGeneration();
    capture.pending = Recorder::PendingWriteSnapshot();
    return capture;
}

thread_local const Capture* current = nullptr;
thread_local std::uint32_t currentSite = 0;
thread_local bool noted = false;

std::atomic<std::uint64_t> commits{0}, validated{0}, unlocked{0}, noSet{0}, preparerStamped{0};
std::atomic<std::int64_t> reportNs{0};

void report() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = reportNs.load(std::memory_order_relaxed);
    if (last == 0) {
        reportNs.store(now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000LL || !reportNs.compare_exchange_strong(last, now)) return;
    AgcDriver::ProfilePrint_nid_no_patch("[prepare-shadow] 10 s: %llu draw commits, %llu validated (%llu preparer-stamped), %llu without a draw-set proof, %llu proofs outside the record lock\n",
        static_cast<unsigned long long>(commits.exchange(0)), static_cast<unsigned long long>(validated.exchange(0)), static_cast<unsigned long long>(preparerStamped.exchange(0)), static_cast<unsigned long long>(noSet.exchange(0)), static_cast<unsigned long long>(unlocked.exchange(0)));
}

bool pendingOverlapUnchanged(std::uint64_t address, std::uint64_t bytes, std::uint64_t fact) {
    const auto snapshot = Recorder::PendingWriteSnapshot();
    return Recorder::SnapshotOverlaps(snapshot.get(), address, static_cast<std::size_t>(bytes)) == (fact != 0);
}

}

bool Enabled() {
    static const bool enabled = std::getenv("APS5_V3E_PREPARE_SHADOW") != nullptr;
    return enabled;
}

void Wrap(std::function<void()>& commit, std::function<void()>& prepare) {
    auto shared = std::make_shared<Shared>();
    shared->enqueued = take();
    if (prepare) {
        prepare = [shared, inner = std::move(prepare)] {
            {
                std::lock_guard lock(shared->mutex);
                if (!shared->committing) {
                    shared->prepared = take();
                    shared->hasPrepared = true;
                }
            }
            inner();
        };
    }
    commit = [shared, inner = std::move(commit)] {
        const Capture* capture = &shared->enqueued;
        std::uint32_t site = 0;
        {
            std::lock_guard lock(shared->mutex);
            shared->committing = true;
            if (shared->hasPrepared) {
                capture = &shared->prepared;
                site = 2;
            }
        }
        commits.fetch_add(1, std::memory_order_relaxed);
        const auto* previous = std::exchange(current, capture);
        const auto previousSite = std::exchange(currentSite, site);
        const bool previousNoted = std::exchange(noted, false);
        struct Restore {
            const Capture* previous;
            std::uint32_t site;
            bool noted;
            ~Restore() {
                if (!PrepareShadow::noted) noSet.fetch_add(1, std::memory_order_relaxed);
                current = previous;
                currentSite = site;
                PrepareShadow::noted = noted;
                report();
            }
        } restore{previous, previousSite, previousNoted};
        inner();
    };
}

bool Active() {
    return current != nullptr && !noted;
}

void NoteDrawSet(const std::function<void(const std::function<void(std::uint64_t, std::uint64_t)>&)>& ranges) {
    if (!Active()) return;
    noted = true;
    if (!GuestMemory::GpuMutex().HeldByThisThread()) {
        unlocked.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    static std::once_flag registered;
    std::call_once(registered, [] { GuestMemory::SetRegistryCheck(GuestMemory::Registry::PendingWrites, pendingOverlapUnchanged); });
    const auto* capture = current;
    GuestMemory::ReadSet reads;
    ranges([&](std::uint64_t address, std::uint64_t bytes) {
        if (bytes == 0) return;
        reads.AddGeneration(address, bytes, capture->generation);
        reads.AddFact(GuestMemory::Registry::PendingWrites, address, bytes, Recorder::SnapshotOverlaps(capture->pending.get(), address, static_cast<std::size_t>(bytes)) ? 1 : 0);
    });
    const auto result = GuestMemory::Validate(capture->stamp, reads);
    GuestMemory::NoteValidation(result, currentSite);
    validated.fetch_add(1, std::memory_order_relaxed);
    if (currentSite == 2) preparerStamped.fetch_add(1, std::memory_order_relaxed);
}

}
