#include "prx/libSceAgcDriver/Execution/include/PrepareStamp.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/LockOrder.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace AgcDriver::GuestMemory {

namespace {

constexpr std::size_t RegistryCount = static_cast<std::size_t>(Registry::Count);
constexpr std::size_t ValidationCount = static_cast<std::size_t>(Validation::Count);
constexpr std::size_t Sites = 4;

std::array<std::atomic<std::uint64_t>, RegistryCount> serials{};

void observeGuestLock(unsigned rank, unsigned event) {
    if (!LockOrder::CheckEnabled()) return;
    const auto ranked = static_cast<LockOrder::Rank>(rank);
    switch (event) {
        case GuestArena::GuestLockBefore: LockOrder::BeforeAcquire(ranked); break;
        case GuestArena::GuestLockHeld:
        case GuestArena::GuestLockHeldUnchecked: LockOrder::NoteHeld(ranked); break;
        case GuestArena::GuestLockReleased: LockOrder::NoteReleased(ranked); break;
        default: break;
    }
}

const bool guestLocksObserved = [] {
    GuestArena::GuestArenaSetLockObserver_nid_postfix(&observeGuestLock);
    return true;
}();
std::array<std::atomic<RegistryCheck>, RegistryCount> checks{};

struct Census {
    std::array<std::array<std::atomic<std::uint64_t>, ValidationCount>, Sites> results{};
    std::array<std::atomic<std::uint64_t>, RegistryCount> moved{};
    std::atomic<std::uint64_t> contents{0}, facts{0}, validations{0}, verified{0}, violations{0}, falseFailures{0};
    std::atomic<std::chrono::steady_clock::rep> next{0};
};

Census& census() {
    static Census value;
    return value;
}

void report() {
    auto& c = census();
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto due = c.next.load(std::memory_order_relaxed);
    if (now < due || !c.next.compare_exchange_strong(due, now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::seconds(10)).count(), std::memory_order_relaxed)) return;
    if (due == 0) return;
    static constexpr const char* siteNames[Sites] = {"draw", "dispatch", "other", "test"};
    const auto take = [](std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed)); };
    const auto validations = take(c.validations);
    const auto contents = take(c.contents);
    const auto facts = take(c.facts);
    std::array<unsigned long long, RegistryCount> moved{};
    for (std::size_t r = 0; r < RegistryCount; ++r) moved[r] = take(c.moved[r]);
    for (std::size_t s = 0; s < Sites; ++s) {
        std::array<unsigned long long, ValidationCount> counts{};
        unsigned long long total = 0;
        for (std::size_t v = 0; v < ValidationCount; ++v) total += counts[v] = take(c.results[s][v]);
        if (total == 0) continue;
        AgcDriver::ProfilePrint_nid_no_patch("[v3m-validate] 10 s %s: %llu validations, valid %llu (%.1f %%), unvouched %llu, content changed %llu, fact changed %llu, no check %llu, mapping changed %llu\n", siteNames[s], total, counts[0], 100.0 * static_cast<double>(counts[0]) / static_cast<double>(total), counts[1], counts[2], counts[3], counts[4], counts[5]);
    }
    const auto verified = take(c.verified), violations = take(c.violations), falseFailures = take(c.falseFailures);
    if (verified != 0) AgcDriver::ProfilePrint_nid_no_patch("[v3m-validate] 10 s verify: %llu commits re-prepared to check Validate, VIOLATIONS %llu (valid but the re-prepare differed), %llu failed although the re-prepare matched\n", verified, violations, falseFailures);
    if (validations == 0) return;
    AgcDriver::ProfilePrint_nid_no_patch("[v3m-validate] 10 s: %.1f content and %.1f fact entries per validation; serial moved since prepare (per validation, %%): pending writes %.1f, labels %.1f, copied writers %.1f, storage results %.1f, eviction %.1f, shadows %.1f, imports %.1f, leases %.1f, residency %.1f\n", static_cast<double>(contents) / static_cast<double>(validations), static_cast<double>(facts) / static_cast<double>(validations),
        100.0 * moved[0] / validations, 100.0 * moved[1] / validations, 100.0 * moved[2] / validations, 100.0 * moved[3] / validations, 100.0 * moved[4] / validations, 100.0 * moved[5] / validations, 100.0 * moved[6] / validations, 100.0 * moved[7] / validations, 100.0 * moved[8] / validations);
}

}

const char* RegistryName(Registry registry) {
    static constexpr const char* names[RegistryCount] = {"pending writes", "labels", "copied writers", "storage results", "eviction", "shadows", "imports", "leases", "residency"};
    const auto index = static_cast<std::size_t>(registry);
    return index < RegistryCount ? names[index] : "?";
}

void BumpRegistry(Registry registry) {
    const auto index = static_cast<std::size_t>(registry);
    if (index < RegistryCount) serials[index].fetch_add(1, std::memory_order_acq_rel);
}

std::uint64_t RegistrySerial(Registry registry) {
    const auto index = static_cast<std::size_t>(registry);
    if (index >= RegistryCount) return 0;
    return serials[index].load(std::memory_order_acquire);
}

bool PrepareStampEnabled() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::V3mPrepareStamp);
}

PrepareStamp PrepareStamp::Take() {
    PrepareStamp stamp;
    for (std::size_t r = 0; r < RegistryCount; ++r) stamp.serials[r] = RegistrySerial(static_cast<Registry>(r));
    stamp.mapping = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    stamp.forget = ForgetSerial();
    stamp.unwatch = UnwatchSerial();
    return stamp;
}

void SetRegistryCheck(Registry registry, RegistryCheck check) {
    const auto index = static_cast<std::size_t>(registry);
    if (index < RegistryCount) checks[index].store(check, std::memory_order_release);
}

void ReadSet::AddToken(std::uint64_t address, std::uint64_t bytes, std::uint64_t token) {
    contents.push_back({address, bytes, token, true});
}

void ReadSet::AddGeneration(std::uint64_t address, std::uint64_t bytes, std::uint64_t generation) {
    contents.push_back({address, bytes, generation, false});
}

void ReadSet::AddFact(Registry registry, std::uint64_t address, std::uint64_t bytes, std::uint64_t fact) {
    facts.push_back({address, bytes, fact, registry});
}

void ReadSet::Clear() {
    contents.clear();
    facts.clear();
}

bool ReadSet::Empty() const {
    return contents.empty() && facts.empty();
}

const char* ValidationName(Validation result) {
    static constexpr const char* names[ValidationCount] = {"valid", "unvouched", "content changed", "fact changed", "no check", "mapping changed"};
    const auto index = static_cast<std::size_t>(result);
    return index < ValidationCount ? names[index] : "?";
}

Validation Validate(const PrepareStamp& stamp, const ReadSet& reads, std::array<bool, RegistryCount>* moved) {
    auto& c = census();
    c.validations.fetch_add(1, std::memory_order_relaxed);
    c.contents.fetch_add(reads.Contents().size(), std::memory_order_relaxed);
    c.facts.fetch_add(reads.Facts().size(), std::memory_order_relaxed);
    std::array<bool, RegistryCount> changed{};
    for (std::size_t r = 0; r < RegistryCount; ++r) {
        changed[r] = RegistrySerial(static_cast<Registry>(r)) != stamp.serials[r];
        if (changed[r]) c.moved[r].fetch_add(1, std::memory_order_relaxed);
    }
    if (moved != nullptr) *moved = changed;
    for (const auto& content : reads.Contents()) {
        if (content.value == 0 || content.bytes == 0) return Validation::Unvouched;
    }
    if (GuestAllocations::GuestAllocationsGeneration_nid_postfix() != stamp.mapping) return Validation::MappingChanged;
    for (const auto& content : reads.Contents()) {
        if (content.token) {
            if (!DirtyCleanSince(content.address, static_cast<std::size_t>(content.bytes), content.value)) return Validation::ContentChanged;
        } else {
            if (CollectWritesUncached(content.address, static_cast<std::size_t>(content.bytes)) == 0) return Validation::Unvouched;
            if (!UnchangedSince(content.address, static_cast<std::size_t>(content.bytes), content.value)) return Validation::ContentChanged;
        }
    }
    for (const auto& fact : reads.Facts()) {
        const auto index = static_cast<std::size_t>(fact.registry);
        if (index >= RegistryCount) return Validation::NoCheck;
        if (!changed[index]) continue;
        const auto check = checks[index].load(std::memory_order_acquire);
        if (check == nullptr) return Validation::NoCheck;
        if (!check(fact.address, fact.bytes, fact.fact)) return Validation::FactChanged;
    }
    return Validation::Valid;
}

int PrepareVerifyMode() {
    static const int mode = [] {
        const char* value = std::getenv("APS5_V3M_PREPARE_VERIFY");
        if (value == nullptr) return 0;
        return std::string_view(value) == "throw" ? 2 : 1;
    }();
    return mode;
}

void NoteVerify(Validation result, bool reprepareMatched, std::uint32_t site) {
    auto& c = census();
    c.verified.fetch_add(1, std::memory_order_relaxed);
    if (result == Validation::Valid && !reprepareMatched) {
        c.violations.fetch_add(1, std::memory_order_relaxed);
        static std::atomic<std::uint32_t> printed{0};
        if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[v3m-validate] verify: site %u validated but its re-prepare differed\n", site);
        if (PrepareVerifyMode() == 2) throw std::runtime_error("APS5_V3M_PREPARE_VERIFY=throw: Validate passed a commit whose re-prepare differed");
    } else if (result != Validation::Valid && reprepareMatched) {
        c.falseFailures.fetch_add(1, std::memory_order_relaxed);
    }
    report();
}

void NoteValidation(Validation result, std::uint32_t site) {
    auto& c = census();
    const auto s = site < Sites ? site : Sites - 1;
    const auto v = static_cast<std::size_t>(result);
    if (v < ValidationCount) c.results[s][v].fetch_add(1, std::memory_order_relaxed);
    report();
}

}
