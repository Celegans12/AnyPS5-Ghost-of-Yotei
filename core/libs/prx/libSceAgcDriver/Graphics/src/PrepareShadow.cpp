#include "prx/libSceAgcDriver/Graphics/include/PrepareShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ImageFlags.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockOrder.hpp"
#include "prx/libSceAgcDriver/Execution/include/PrepareStamp.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>

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
    // v2, written by the preparer between the slot's Begin and Finish, read by the commit after its Take.
    bool lookedUp = false;
    const void* preparedSet = nullptr;
    GuestMemory::ReadSet reads;
};

Capture take() {
    Capture capture;
    capture.stamp = GuestMemory::PrepareStamp::Take();
    capture.generation = GuestMemory::TrackerGeneration();
    capture.pending = Recorder::PendingWriteSnapshot();
    return capture;
}

thread_local const Capture* current = nullptr;
thread_local Shared* currentShared = nullptr;
thread_local std::uint32_t currentSite = 0;
thread_local bool noted = false;
thread_local Shared* preparing = nullptr;

std::atomic<std::uint64_t> commits{0}, validated{0}, unlocked{0}, noSet{0}, preparerStamped{0};
// v2 outcomes per commit with a draw-set proof.
std::atomic<std::uint64_t> notPrepared{0}, noCachedSet{0}, otherSet{0}, unlockedReads{0}, tokens{0}, generations{0}, unarmed{0};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(GuestMemory::Validation::Count)> results{};
std::atomic<std::int64_t> reportNs{0};

unsigned long long take(std::atomic<std::uint64_t>& value) {
    return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed));
}

void report() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = reportNs.load(std::memory_order_relaxed);
    if (last == 0) {
        reportNs.store(now, std::memory_order_relaxed);
        return;
    }
    if (now - last < 10'000'000'000LL || !reportNs.compare_exchange_strong(last, now)) return;
    AgcDriver::ProfilePrint_nid_no_patch("[prepare-shadow] 10 s: %llu draw commits, %llu validated (%llu preparer-stamped), %llu without a draw-set proof, %llu proofs outside the record lock\n",
        take(commits), take(validated), take(preparerStamped), take(noSet), take(unlocked));
    if (Mode() != 2) return;
    std::array<unsigned long long, static_cast<std::size_t>(GuestMemory::Validation::Count)> counts{};
    for (std::size_t i = 0; i < counts.size(); ++i) counts[i] = take(results[i]);
    AgcDriver::ProfilePrint_nid_no_patch("[prepare-shadow2] 10 s: validated against the prepare-time set: valid %llu, unvouched %llu, content changed %llu, fact changed %llu, no check %llu, mapping changed %llu; re-prepare without validation: not prepared ahead %llu, no cached set at prepare %llu, another set committed %llu; set reads without the resource-set lock %llu; ranges by generation %llu, by token %llu, unarmed %llu\n",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], take(notPrepared), take(noCachedSet), take(otherSet), take(unlockedReads), take(generations), take(tokens), take(unarmed));
}

bool pendingOverlapUnchanged(std::uint64_t address, std::uint64_t bytes, std::uint64_t fact) {
    const auto snapshot = Recorder::PendingWriteSnapshot();
    return Recorder::SnapshotOverlaps(snapshot.get(), address, static_cast<std::size_t>(bytes)) == (fact != 0);
}

void registerPendingCheck() {
    static std::once_flag registered;
    std::call_once(registered, [] { GuestMemory::SetRegistryCheck(GuestMemory::Registry::PendingWrites, pendingOverlapUnchanged); });
}

}

int Mode() {
    static const int mode = [] {
        const char* value = std::getenv("APS5_V3E_PREPARE_SHADOW");
        if (value == nullptr) return 0;
        return std::strcmp(value, "2") == 0 ? 2 : 1;
    }();
    return mode;
}

bool Enabled() {
    return Mode() != 0;
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
            const auto* previous = std::exchange(preparing, shared->hasPrepared ? shared.get() : nullptr);
            struct Restore {
                Shared* previous;
                ~Restore() { preparing = previous; }
            } restore{const_cast<Shared*>(previous)};
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
        auto* previousShared = std::exchange(currentShared, shared.get());
        const auto previousSite = std::exchange(currentSite, site);
        const bool previousNoted = std::exchange(noted, false);
        struct Restore {
            const Capture* previous;
            Shared* shared;
            std::uint32_t site;
            bool noted;
            ~Restore() {
                if (!PrepareShadow::noted) noSet.fetch_add(1, std::memory_order_relaxed);
                current = previous;
                currentShared = shared;
                currentSite = site;
                PrepareShadow::noted = noted;
                report();
            }
        } restore{previous, previousShared, previousSite, previousNoted};
        inner();
    };
}

bool Active() {
    return current != nullptr && !noted;
}

bool Preparing() {
    return Mode() == 2 && preparing != nullptr;
}

void NotePreparedSet(const void* set, const Ranges& ranges) {
    auto* shared = preparing;
    if (Mode() != 2 || shared == nullptr) return;
    shared->lookedUp = true;
    if (set == nullptr) return;
    registerPendingCheck();
    const auto generation = shared->prepared.generation;
    const auto snapshot = Recorder::PendingWriteSnapshot();
    std::shared_lock<LockOrder::CacheLock> setLock(LockOrder::CacheLockFor(LockOrder::Rank::ResourceSet), std::defer_lock);
    if (RuntimeSwitches::Set(RuntimeSwitch::V3eLockSplit)) setLock.lock();
    else unlockedReads.fetch_add(1, std::memory_order_relaxed);
    ranges([&](std::uint64_t address, std::uint64_t bytes) {
        if (bytes == 0) return;
        if (GuestMemory::CollectWritesUncached(address, static_cast<std::size_t>(bytes)) != 0) {
            shared->reads.AddGeneration(address, bytes, generation);
            generations.fetch_add(1, std::memory_order_relaxed);
        } else {
            const auto token = ImageFlags::Arm(address, static_cast<std::size_t>(bytes));
            shared->reads.AddToken(address, bytes, token);
            (token != 0 ? tokens : unarmed).fetch_add(1, std::memory_order_relaxed);
        }
        const std::pair<std::uint64_t, std::uint64_t> range{address, address + bytes};
        shared->reads.AddFact(GuestMemory::Registry::PendingWrites, address, bytes, Recorder::SnapshotOverlaps(snapshot.get(), address, static_cast<std::size_t>(bytes)) ? 1 : 0);
        shared->reads.AddFact(GuestMemory::Registry::StorageResults, address, bytes, StorageTexture::AnyPendingOverlaps({&range, 1}) ? 1 : 0);
        shared->reads.AddFact(GuestMemory::Registry::Shadows, address, bytes, AnyShadowedOverlaps(address, static_cast<std::size_t>(bytes)) ? 1 : 0);
    });
    shared->preparedSet = set;
}

void NoteDrawSet(const void* set, const Ranges& ranges) {
    if (!Active()) return;
    noted = true;
    if (!GuestMemory::GpuMutex().HeldByThisThread()) {
        unlocked.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    registerPendingCheck();
    validated.fetch_add(1, std::memory_order_relaxed);
    if (currentSite == 2) preparerStamped.fetch_add(1, std::memory_order_relaxed);
    if (Mode() == 2) {
        const auto* shared = currentShared;
        if (currentSite != 2 || shared == nullptr || !shared->lookedUp) notPrepared.fetch_add(1, std::memory_order_relaxed);
        else if (shared->preparedSet == nullptr) noCachedSet.fetch_add(1, std::memory_order_relaxed);
        else if (shared->preparedSet != set) otherSet.fetch_add(1, std::memory_order_relaxed);
        else {
            const auto result = GuestMemory::Validate(current->stamp, shared->reads);
            results[static_cast<std::size_t>(result)].fetch_add(1, std::memory_order_relaxed);
            GuestMemory::NoteValidation(result, 2);
        }
        return;
    }
    const auto* capture = current;
    GuestMemory::ReadSet reads;
    ranges([&](std::uint64_t address, std::uint64_t bytes) {
        if (bytes == 0) return;
        reads.AddGeneration(address, bytes, capture->generation);
        reads.AddFact(GuestMemory::Registry::PendingWrites, address, bytes, Recorder::SnapshotOverlaps(capture->pending.get(), address, static_cast<std::size_t>(bytes)) ? 1 : 0);
    });
    const auto result = GuestMemory::Validate(capture->stamp, reads);
    GuestMemory::NoteValidation(result, currentSite);
}

}
