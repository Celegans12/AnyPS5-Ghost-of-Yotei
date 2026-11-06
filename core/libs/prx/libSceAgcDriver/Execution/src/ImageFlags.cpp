#include "prx/libSceAgcDriver/Execution/include/ImageFlags.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::ImageFlags {

namespace {

bool flag(const char* name) {
    const char* text = std::getenv(name);
    return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0;
}

std::uint64_t unarmableArm(std::uint64_t, std::size_t) { return 0; }
bool neverClean(std::uint64_t, std::size_t, std::uint64_t) { return false; }
bool neverCleanAll(std::span<const DirtyQuery>) { return false; }
void noMark(std::uint64_t, std::size_t) {}
bool notEnabled() { return false; }

constexpr Backend Unarmable{&unarmableArm, &neverClean, &neverCleanAll, &noMark, &notEnabled};

std::uint64_t trackerArm(std::uint64_t address, std::size_t bytes) { return GuestMemory::DirtyEnabled() ? GuestMemory::DirtyArm(address, bytes) : 0; }
bool trackerClean(std::uint64_t address, std::size_t bytes, std::uint64_t token) { return GuestMemory::DirtyCleanSince(address, bytes, token); }
bool trackerCleanAll(std::span<const DirtyQuery> queries) {
    static_assert(sizeof(DirtyQuery) == sizeof(GuestMemory::DirtyQuery));
    return GuestMemory::DirtyCleanSinceAll(std::span(reinterpret_cast<const GuestMemory::DirtyQuery*>(queries.data()), queries.size()));
}
void trackerMark(std::uint64_t address, std::size_t bytes) { GuestMemory::DirtyMark(address, bytes); }
bool trackerEnabled() { return GuestMemory::DirtyEnabled(); }

constexpr Backend Tracker{&trackerArm, &trackerClean, &trackerCleanAll, &trackerMark, &trackerEnabled};

std::atomic<const Backend*> backend{&Tracker};

// Runtime switches (APS5_AB): every turn-on starts a new epoch, carried in a token's top 16 bits, so a token armed in
// an earlier "on" phase (whose cached object may have moved while the flags were off) is never clean again.
constexpr unsigned EpochShift = 48;
constexpr std::uint64_t TokenMask = (std::uint64_t{1} << EpochShift) - 1;
std::atomic<std::uint64_t> epoch{1};
std::uint64_t epochTag() { return (epoch.load(std::memory_order_acquire) & 0xffffu) << EpochShift; }
bool follow(Graphics::RuntimeSwitch id, std::atomic<bool>& last) {
    const bool on = Graphics::RuntimeSwitches::Set(id);
    if (on) {
        if (!last.load(std::memory_order_relaxed) && !last.exchange(true, std::memory_order_acq_rel)) epoch.fetch_add(1, std::memory_order_acq_rel);
    } else if (last.load(std::memory_order_relaxed)) {
        last.store(false, std::memory_order_relaxed);
    }
    return on;
}

struct Counts {
    std::array<std::atomic<std::uint64_t>, 3> verdicts{};
    std::atomic<std::uint64_t> proofs{0}, served{0}, arms{0}, armed{0}, early{0}, storageClean{0}, storageFull{0}, gateOther{0}, gateCpu{0}, gateUnarmed{0};
};
Counts counts;
std::array<std::uint64_t, 13> reported{};

// C4-buffers, storage skips and no-token classes: [bufferflags] / [storageflags] lines.
constexpr std::size_t BufferUses = static_cast<std::size_t>(BufferUse::Count);
constexpr std::size_t Skips = static_cast<std::size_t>(StorageSkip::Count);
constexpr std::size_t UnwatchedClasses = static_cast<std::size_t>(Unwatched::Count);
// per use: clean, missing or dirty, armed, gated
constexpr std::size_t BufferFields = 4;
constexpr std::size_t ExtraCount = BufferUses * BufferFields + 2 + Skips + UnwatchedClasses;
std::array<std::atomic<std::uint64_t>, ExtraCount> extra{};
std::array<std::uint64_t, ExtraCount> extraReported{};
constexpr std::size_t DropsAt = BufferUses * BufferFields;
constexpr std::size_t SkipsAt = DropsAt + 2;
constexpr std::size_t UnwatchedAt = SkipsAt + Skips;

void reportExtraLocked(const char* label) {
    std::array<std::uint64_t, ExtraCount> d{};
    for (std::size_t i = 0; i < ExtraCount; ++i) {
        const auto now = extra[i].load(std::memory_order_relaxed);
        d[i] = now - extraReported[i];
        extraReported[i] = now;
    }
    const auto u = [&](std::size_t i) { return static_cast<unsigned long long>(d[i]); };
    if (BufferEnabled() || InputEnabled()) {
        AgcDriver::ProfilePrint_nid_no_patch("[bufferflags] %s: bindings clean %llu / not clean %llu (armed %llu, gated %llu); inputs clean %llu / not clean %llu (armed %llu, gated %llu); memo drops %llu (%llu entries)%s\n", label, u(0), u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(DropsAt), u(DropsAt + 1), BufferVerify() ? "; verify on" : "");
    }
    if (Enabled()) {
        AgcDriver::ProfilePrint_nid_no_patch("[storageflags] %s: not flagged: scoped %llu, dcc %llu, keys %llu; no token after a full refresh: kept at unwatch %llu, imported %llu, view %llu, not writable %llu, other private %llu, arm refused %llu%s\n", label, u(SkipsAt), u(SkipsAt + 1), u(SkipsAt + 2), u(UnwatchedAt), u(UnwatchedAt + 1), u(UnwatchedAt + 2), u(UnwatchedAt + 3), u(UnwatchedAt + 4), u(UnwatchedAt + 5), TargetEnabled() ? "; target flags on" : StorageEnabled() ? "" : "; storage flags off");
    }
}
std::mutex reportMutex;
std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();

void reportLocked(const char* label) {
    const std::array<std::uint64_t, 13> now{counts.verdicts[0].load(), counts.verdicts[1].load(), counts.verdicts[2].load(), counts.proofs.load(), counts.served.load(), counts.arms.load(), counts.armed.load(), counts.early.load(), counts.storageClean.load(), counts.storageFull.load(), counts.gateOther.load(), counts.gateCpu.load(), counts.gateUnarmed.load()};
    std::array<std::uint64_t, 13> d{};
    for (std::size_t i = 0; i < now.size(); ++i) {
        d[i] = now[i] - reported[i];
        reported[i] = now[i];
    }
    AgcDriver::ProfilePrint_nid_no_patch("[imageflags] %s: storage refreshes clean %llu / full %llu (gates: other pending %llu, cpu pending %llu, no token %llu); early clean hits %llu; snapshot hits checked %llu: not armed %llu, dirty %llu, clean %llu; proofs run %llu, served %llu; arms %llu (armed %llu)%s\n", label, static_cast<unsigned long long>(d[8]), static_cast<unsigned long long>(d[9]), static_cast<unsigned long long>(d[10]), static_cast<unsigned long long>(d[11]), static_cast<unsigned long long>(d[12]), static_cast<unsigned long long>(d[7]),static_cast<unsigned long long>(d[0] + d[1] + d[2]), static_cast<unsigned long long>(d[0]), static_cast<unsigned long long>(d[1]), static_cast<unsigned long long>(d[2]), static_cast<unsigned long long>(d[3]), static_cast<unsigned long long>(d[4]), static_cast<unsigned long long>(d[5]), static_cast<unsigned long long>(d[6]), Verify() ? "; verify on" : "");
}

}

bool Enabled() {
    static std::atomic<bool> last{false};
    return follow(Graphics::RuntimeSwitch::V3eImageFlags, last);
}

bool Verify() {
    static const bool verify = flag("APS5_V3E_IMAGE_FLAGS_VERIFY");
    return verify;
}

bool BufferEnabled() {
    static std::atomic<bool> last{false};
    return follow(Graphics::RuntimeSwitch::V3eBindingFlags, last);
}

bool InputEnabled() {
    static std::atomic<bool> last{false};
    return follow(Graphics::RuntimeSwitch::V3eInputFlags, last);
}

bool BufferVerify() {
    static const bool verify = flag("APS5_V3E_BUFFER_FLAGS_VERIFY");
    return verify;
}

bool StorageEnabled() {
    static const bool enabled = flag("APS5_V3E_STORAGE_FLAGS");
    return enabled && Enabled();
}

bool TargetEnabled() {
    static const bool enabled = flag("APS5_V3E_TARGET_FLAGS");
    return enabled && StorageEnabled();
}

void CountBuffer(BufferUse use, bool clean, bool armed) {
    const auto at = static_cast<std::size_t>(use) * BufferFields;
    extra[at + (clean ? 0 : 1)].fetch_add(1, std::memory_order_relaxed);
    if (armed) extra[at + 2].fetch_add(1, std::memory_order_relaxed);
}

void CountBufferGate(BufferUse use) {
    extra[static_cast<std::size_t>(use) * BufferFields + 3].fetch_add(1, std::memory_order_relaxed);
}

void CountBufferDrop(std::size_t entries) {
    extra[DropsAt].fetch_add(1, std::memory_order_relaxed);
    extra[DropsAt + 1].fetch_add(entries, std::memory_order_relaxed);
}

void CountStorageSkip(StorageSkip skip) {
    extra[SkipsAt + static_cast<std::size_t>(skip)].fetch_add(1, std::memory_order_relaxed);
}

void CountUnwatched(Unwatched reason) {
    extra[UnwatchedAt + static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void ThrowBufferDisagreement(std::uint64_t address, std::size_t bytes, const char* what) {
    char text[224];
    std::snprintf(text, sizeof(text), "AGC graphics: buffer flags call 0x%llx (+0x%zx) clean while today's path finds %s (APS5_V3E_BUFFER_FLAGS_VERIFY)", static_cast<unsigned long long>(address), bytes, what);
    throw std::runtime_error(text);
}

void UseBackend(const Backend* use) {
    backend.store(use != nullptr ? use : &Unarmable, std::memory_order_release);
}

void NoteServed(std::uint64_t address, std::size_t bytes) {
    if (backend.load(std::memory_order_acquire) == &Tracker) GuestMemory::DirtyVerifyUse(address, bytes);
}

std::uint64_t Arm(std::uint64_t address, std::size_t bytes) {
    const auto token = backend.load(std::memory_order_acquire)->arm(address, bytes);
    counts.arms.fetch_add(1, std::memory_order_relaxed);
    if (token == 0 || token > TokenMask) return 0;
    counts.armed.fetch_add(1, std::memory_order_relaxed);
    return epochTag() | token;
}

bool CleanSince(std::uint64_t address, std::size_t bytes, std::uint64_t token) {
    if (token == 0 || (token & ~TokenMask) != epochTag()) return false;
    return backend.load(std::memory_order_acquire)->cleanSince(address, bytes, token & TokenMask);
}

bool CleanSinceAll(std::span<const DirtyQuery> queries) {
    if (queries.empty()) return false;
    const auto tag = epochTag();
    thread_local std::vector<DirtyQuery> untagged;
    untagged.clear();
    for (const auto& query : queries) {
        if (query.token == 0 || (query.token & ~TokenMask) != tag) return false;
        untagged.push_back({query.address, query.bytes, query.token & TokenMask});
    }
    return backend.load(std::memory_order_acquire)->cleanSinceAll(untagged);
}

void CountStorageGate(bool otherPending, bool cpuPending, bool unarmed) {
    if (otherPending) counts.gateOther.fetch_add(1, std::memory_order_relaxed);
    if (cpuPending) counts.gateCpu.fetch_add(1, std::memory_order_relaxed);
    if (unarmed) counts.gateUnarmed.fetch_add(1, std::memory_order_relaxed);
}

void CountStorage(bool served) {
    (served ? counts.storageClean : counts.storageFull).fetch_add(1, std::memory_order_relaxed);
}

void CountEarly(bool served) {
    if (served) counts.early.fetch_add(1, std::memory_order_relaxed);
}

void Count(Verdict verdict, bool proofAsked, bool served) {
    counts.verdicts[static_cast<std::size_t>(verdict)].fetch_add(1, std::memory_order_relaxed);
    if (proofAsked) counts.proofs.fetch_add(1, std::memory_order_relaxed);
    if (served) counts.served.fetch_add(1, std::memory_order_relaxed);
}

void ThrowDisagreement(std::uint64_t address, std::size_t bytes) {
    char text[192];
    std::snprintf(text, sizeof(text), "AGC graphics: image flags call the snapshot of 0x%llx (+0x%zx) clean while its proof finds it changed (APS5_V3E_IMAGE_FLAGS_VERIFY)", static_cast<unsigned long long>(address), bytes);
    throw std::runtime_error(text);
}

void Close(const char* label) {
    if (!Enabled() && !BufferEnabled() && !InputEnabled()) return;
    std::lock_guard lock(reportMutex);
    if (Enabled()) reportLocked(label);
    reportExtraLocked(label);
}

void MaybeReport() {
    if ((!Enabled() && !BufferEnabled() && !InputEnabled()) || ProfileWindows::FrameWindows()) return;
    std::unique_lock lock(reportMutex, std::try_to_lock);
    if (!lock.owns_lock() || !ProfileWindows::ClockDue(lastReport)) return;
    if (Enabled()) reportLocked("10s");
    reportExtraLocked("10s");
}

}
