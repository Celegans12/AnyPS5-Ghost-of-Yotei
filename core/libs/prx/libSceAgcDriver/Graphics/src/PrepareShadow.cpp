#include "prx/libSceAgcDriver/Graphics/include/PrepareShadow.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ImageFlags.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockOrder.hpp"
#include "prx/libSceAgcDriver/Execution/include/PrepareStamp.hpp"
#include "prx/libSceAgcDriver/Execution/include/PrepareUnwired.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libc/include/GuestDirty.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
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
    // Whether the read set is built (census v2, or phase (ii) on at prepare) and phase (ii)'s switch at prepare.
    bool building = false;
    bool proof = false;
    ShaderResources::Ahead ahead = ShaderResources::Ahead::Other;
    // Why the read set is unvouched, if it is (an index into causeNames).
    std::uint8_t unvouchedCause = 0xff;
};

// Unvouched causes: GuestDirty's refusal reason + 1 (0: Arm returned 0 without one), then the unwired registries.
constexpr const char* causeNames[] = {"no reason", "epoch", "in flight", "guards", "block", "uncommitted", "private not writable", "other memory", "view", "tracker off", "copied writes", "lease", "unwatched since prepare"};
constexpr std::size_t CauseCount = std::size(causeNames);
constexpr std::uint8_t CauseCopied = 10, CauseLeased = 11, CauseUnwatched = 12;

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
thread_local bool recheckPending = false;

std::atomic<std::uint64_t> commits{0}, validated{0}, unlocked{0}, noSet{0}, preparerStamped{0};
// v2 outcomes per commit with a draw-set proof.
std::atomic<std::uint64_t> notPrepared{0}, noCachedSet{0}, otherSet{0}, unlockedReads{0}, tokens{0}, generations{0}, unarmed{0};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(GuestMemory::Validation::Count)> results{};
// Unarmed ranges by refusal reason, and unvouched commits by their first cause.
std::array<std::atomic<std::uint64_t>, CauseCount> unarmedBy{}, unvouchedBy{};
std::atomic<std::uint64_t> copiedRanges{0}, leasedSets{0};
// Phase (ii): commits that asked (TakePrepared), verdicts at prepare, outcomes at the commit.
std::atomic<std::uint64_t> noView{0}, proofAsks{0}, proofSwitchMoved{0}, proofSerialMoved{0}, proofServed{0}, proofVerified{0}, proofVerifyFailed{0};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ShaderResources::Ahead::Count)> aheadBy{};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(Fallback::Count)> fallbackBy{};
std::atomic<std::int64_t> reportNs{0};

unsigned long long take(std::atomic<std::uint64_t>& value) {
    return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed));
}

void reportCauses() {
    char unarmedText[512] = {}, unvouchedText[512] = {};
    std::size_t at = 0, atV = 0;
    for (std::size_t i = 0; i < CauseCount; ++i) {
        const auto u = take(unarmedBy[i]);
        const auto v = take(unvouchedBy[i]);
        if (u != 0 && at < sizeof(unarmedText)) at += static_cast<std::size_t>(std::snprintf(unarmedText + at, sizeof(unarmedText) - at, " %s %llu", causeNames[i], u));
        if (v != 0 && atV < sizeof(unvouchedText)) atV += static_cast<std::size_t>(std::snprintf(unvouchedText + atV, sizeof(unvouchedText) - atV, " %s %llu", causeNames[i], v));
    }
    AgcDriver::ProfilePrint_nid_no_patch("[prepare-causes] 10 s: unarmed ranges by refusal:%s; unvouched commits by first cause:%s; copied-write ranges %llu, leased sets %llu\n", unarmedText, unvouchedText, take(copiedRanges), take(leasedSets));
}

void reportProof() {
    char aheadText[512] = {};
    std::size_t at = 0;
    for (std::size_t i = 0; i < aheadBy.size(); ++i) {
        const auto n = take(aheadBy[i]);
        if (n != 0 && at < sizeof(aheadText)) at += static_cast<std::size_t>(std::snprintf(aheadText + at, sizeof(aheadText) - at, " %s %llu", ShaderResources::AheadName(static_cast<ShaderResources::Ahead>(i)), n));
    }
    std::array<unsigned long long, static_cast<std::size_t>(Fallback::Count)> fb{};
    for (std::size_t i = 0; i < fb.size(); ++i) fb[i] = take(fallbackBy[i]);
    AgcDriver::ProfilePrint_nid_no_patch("[prepare-proof] 10 s: %llu draw-set proofs asked, %llu served from the prepared verdict; prepared verdicts of the validated sets:%s; no published view %llu; not served: switch moved %llu, commit re-check moved %llu evicted %llu depth %llu imports %llu untracked %llu pending %llu (pending re-checked after a serial move %llu); verify %llu checked, %llu differed\n",
        take(proofAsks), take(proofServed), aheadText, take(noView), take(proofSwitchMoved), fb[1], fb[2], fb[3], fb[4], fb[5], fb[6], take(proofSerialMoved), take(proofVerified), take(proofVerifyFailed));
}

void report() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto last = reportNs.load(std::memory_order_relaxed);
    if (last == 0) {
        reportNs.store(now, std::memory_order_relaxed);
        return;
    }
    // APS5_V3E_PREPARE_REPORT_MS (default 10000): the report interval; replay rows use a short one.
    static const long long interval = [] {
        const char* text = std::getenv("APS5_V3E_PREPARE_REPORT_MS");
        return (text != nullptr ? std::atoll(text) : 10000LL) * 1'000'000LL;
    }();
    if (now - last < interval || !reportNs.compare_exchange_strong(last, now)) return;
    AgcDriver::ProfilePrint_nid_no_patch("[prepare-shadow] 10 s: %llu draw commits, %llu validated (%llu preparer-stamped), %llu without a draw-set proof, %llu proofs outside the record lock\n",
        take(commits), take(validated), take(preparerStamped), take(noSet), take(unlocked));
    if (RuntimeSwitches::Set(RuntimeSwitch::V3ePrepareProof) || Mode() == 2) {
        reportProof();
        if (Mode() != 2) reportCauses();
    }
    if (Mode() != 2) return;
    std::array<unsigned long long, static_cast<std::size_t>(GuestMemory::Validation::Count)> counts{};
    for (std::size_t i = 0; i < counts.size(); ++i) counts[i] = take(results[i]);
    AgcDriver::ProfilePrint_nid_no_patch("[prepare-shadow2] 10 s: validated against the prepare-time set: valid %llu, unvouched %llu, content changed %llu, fact changed %llu, no check %llu, mapping changed %llu; re-prepare without validation: not prepared ahead %llu, no cached set at prepare %llu, another set committed %llu; set reads without the resource-set lock %llu; ranges by generation %llu, by token %llu, unarmed %llu\n",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], take(notPrepared), take(noCachedSet), take(otherSet), take(unlockedReads), take(generations), take(tokens), take(unarmed));
    reportCauses();
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

bool Wrapping() {
    static const bool wrapping = [] {
        if (Enabled() || std::getenv("APS5_V3E_PREPARE_PROOF") != nullptr) return true;
        const char* ab = std::getenv("APS5_AB");
        return ab != nullptr && std::strstr(ab, "APS5_V3E_PREPARE_PROOF") != nullptr;
    }();
    return wrapping;
}

bool ProofOn() {
    return Wrapping() && RuntimeSwitches::Set(RuntimeSwitch::V3ePrepareProof) && RuntimeSwitches::Set(RuntimeSwitch::V3eLockSplit);
}

bool PublishOnly() {
    return RuntimeSwitches::Set(RuntimeSwitch::V3ePpPublishOnly);
}

bool NoPublish() {
    return RuntimeSwitches::Set(RuntimeSwitch::V3ePpNoPublish);
}

bool VerifyProof() {
    static const bool verify = [] {
        const char* value = std::getenv("APS5_V3E_PREPARE_PROOF_VERIFY");
        return value != nullptr && std::strcmp(value, "throw") == 0;
    }();
    return verify;
}

void Wrap(std::function<void()>& commit, std::function<void()>& prepare) {
    auto shared = std::make_shared<Shared>();
    if (Mode() != 0) shared->enqueued = take();
    if (prepare) {
        prepare = [shared, inner = std::move(prepare)] {
            {
                std::lock_guard lock(shared->mutex);
                if (!shared->committing) {
                    shared->proof = RuntimeSwitches::Set(RuntimeSwitch::V3ePrepareProof);
                    shared->building = Mode() == 2 || shared->proof;
                    if (Mode() != 0 || shared->building) {
                        shared->prepared = take();
                        shared->hasPrepared = true;
                    }
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
    return preparing != nullptr && preparing->building;
}

void NotePreparedSet(const ShaderResources* set, std::span<const CompiledShader> shaders) {
    auto* shared = preparing;
    if (shared == nullptr || !shared->building) return;
    shared->lookedUp = true;
    if (set == nullptr) return;
    registerPendingCheck();
    const auto generation = shared->prepared.generation;
    const auto snapshot = Recorder::PendingWriteSnapshot();
    // Phase (ii) reads the set's published view and takes no cache lock (a commit's stage hold would wait on it);
    // census v2 alone reads the set under the shared resource-set lock.
    // Census v2 also reads the published view (and vouches token-first) when views are published
    // (APS5_V3E_PP_PUBLISH_ONLY), so the census takes no cache lock and adds no tracker-lock waits.
    const bool viewed = shared->proof || PublishOnly();
    const auto view = viewed ? set->LoadProofView() : nullptr;
    if (viewed && view == nullptr) {
        noView.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::shared_lock<LockOrder::CacheLock> setLock(LockOrder::CacheLockFor(LockOrder::Rank::ResourceSet), std::defer_lock);
    if (view != nullptr) {
    } else if (RuntimeSwitches::Set(RuntimeSwitch::V3eLockSplit)) setLock.lock();
    else unlockedReads.fetch_add(1, std::memory_order_relaxed);
    const auto cause = [&](std::uint8_t why) {
        if (shared->unvouchedCause == 0xff) shared->unvouchedCause = why;
    };
    if (GuestMemory::AddUnwiredEntry(shared->reads, 0, 1, false, view != nullptr ? ShaderResources::ViewHoldsLease(*view) : set->HoldsLease())) {
        leasedSets.fetch_add(1, std::memory_order_relaxed);
        cause(CauseLeased);
    }
    const auto addRange = [&](std::uint64_t address, std::uint64_t bytes, bool copied) {
        if (bytes == 0) return;
        if (GuestMemory::AddUnwiredEntry(shared->reads, address, bytes, copied, false)) {
            copiedRanges.fetch_add(1, std::memory_order_relaxed);
            cause(CauseCopied);
        }
        // Phase (ii) vouches by a dirty token first: validating it is lock-free, while a generation entry costs an
        // uncached write-watch collect under the tracker lock at the commit (and here), which the commits then wait on
        // (v3e-ab11: record-lock holders waited +1.1 s per 10 s on the tracker). A refused arm falls back to a generation.
        std::uint64_t early = 0;
        if (view != nullptr) {
            static_cast<void>(GuestArena::GuestDirtyTakeLastRefusal_nid_postfix());
            early = ImageFlags::Arm(address, static_cast<std::size_t>(bytes));
        }
        if (early != 0) {
            shared->reads.AddToken(address, bytes, early);
            tokens.fetch_add(1, std::memory_order_relaxed);
        } else if (GuestMemory::CollectWritesUncached(address, static_cast<std::size_t>(bytes)) != 0) {
            shared->reads.AddGeneration(address, bytes, generation);
            generations.fetch_add(1, std::memory_order_relaxed);
        } else {
            static_cast<void>(GuestArena::GuestDirtyTakeLastRefusal_nid_postfix());
            const auto token = ImageFlags::Arm(address, static_cast<std::size_t>(bytes));
            shared->reads.AddToken(address, bytes, token);
            (token != 0 ? tokens : unarmed).fetch_add(1, std::memory_order_relaxed);
            if (token == 0) {
                const auto reason = std::min<std::size_t>(GuestArena::GuestDirtyTakeLastRefusal_nid_postfix(), CauseCopied - 1);
                unarmedBy[reason].fetch_add(1, std::memory_order_relaxed);
                cause(static_cast<std::uint8_t>(reason));
            }
        }
        const std::pair<std::uint64_t, std::uint64_t> range{address, address + bytes};
        shared->reads.AddFact(GuestMemory::Registry::PendingWrites, address, bytes, Recorder::SnapshotOverlaps(snapshot.get(), address, static_cast<std::size_t>(bytes)) ? 1 : 0);
        // Phase (ii) re-runs the proof's own pending and shadow checks at the commit when these serials moved
        // (RecheckPending), with the proof's except semantics; the coarse overlap facts are census v2's.
        if (view != nullptr) return;
        shared->reads.AddFact(GuestMemory::Registry::StorageResults, address, bytes, StorageTexture::AnyPendingOverlaps({&range, 1}) ? 1 : 0);
        shared->reads.AddFact(GuestMemory::Registry::Shadows, address, bytes, AnyShadowedOverlaps(address, static_cast<std::size_t>(bytes)) ? 1 : 0);
    };
    if (view != nullptr) ShaderResources::ForEachViewRange(*view, addRange);
    else set->ForEachReadRange([&](std::uint64_t address, std::uint64_t bytes) { addRange(address, bytes, bytes != 0 && set->CopiedWritesOverlap(address, static_cast<std::size_t>(bytes))); });
    if (view != nullptr) shared->ahead = ShaderResources::ProveAhead(*view);
    static_cast<void>(shaders);
    shared->preparedSet = set;
}

namespace {

// Counts a validation of the prepare-time read set (census v2 and phase (ii) alike).
void countValidation(const Shared& shared, GuestMemory::Validation result) {
    results[static_cast<std::size_t>(result)].fetch_add(1, std::memory_order_relaxed);
    GuestMemory::NoteValidation(result, 2);
    if (result == GuestMemory::Validation::Unvouched) unvouchedBy[shared.unvouchedCause < CauseCount ? shared.unvouchedCause : CauseUnwatched].fetch_add(1, std::memory_order_relaxed);
}

}

bool TakePrepared(const void* set) {
    recheckPending = false;
    if (current == nullptr || noted) return false;
    noted = true;
    if (!GuestMemory::GpuMutex().HeldByThisThread()) {
        unlocked.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    registerPendingCheck();
    proofAsks.fetch_add(1, std::memory_order_relaxed);
    validated.fetch_add(1, std::memory_order_relaxed);
    const auto* shared = currentShared;
    if (currentSite != 2 || shared == nullptr || !shared->lookedUp) {
        notPrepared.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    preparerStamped.fetch_add(1, std::memory_order_relaxed);
    if (shared->preparedSet == nullptr) {
        noCachedSet.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (shared->preparedSet != set) {
        otherSet.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::array<bool, static_cast<std::size_t>(GuestMemory::Registry::Count)> moved{};
    const auto result = GuestMemory::Validate(current->stamp, shared->reads, &moved);
    countValidation(*shared, result);
    if (result != GuestMemory::Validation::Valid) return false;
    aheadBy[static_cast<std::size_t>(shared->ahead)].fetch_add(1, std::memory_order_relaxed);
    if (!shared->proof) {
        proofSwitchMoved.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (shared->ahead != ShaderResources::Ahead::Pass) return false;
    recheckPending = moved[static_cast<std::size_t>(GuestMemory::Registry::StorageResults)] || moved[static_cast<std::size_t>(GuestMemory::Registry::Shadows)];
    if (recheckPending) proofSerialMoved.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool RecheckPending() {
    return recheckPending;
}

void NoteFallback(Fallback reason) {
    fallbackBy[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void NoteServed(bool served) {
    if (served) proofServed.fetch_add(1, std::memory_order_relaxed);
}

void NoteVerify(bool same) {
    (same ? proofVerified : proofVerifyFailed).fetch_add(1, std::memory_order_relaxed);
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
            countValidation(*shared, result);
            if (result == GuestMemory::Validation::Valid) aheadBy[static_cast<std::size_t>(shared->ahead)].fetch_add(1, std::memory_order_relaxed);
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
