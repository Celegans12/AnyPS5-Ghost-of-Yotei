#include "prx/libSceAgcDriver/Execution/include/ImageFlags.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

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

std::atomic<const Backend*> backend{&Unarmable};

struct Counts {
    std::array<std::atomic<std::uint64_t>, 3> verdicts{};
    std::atomic<std::uint64_t> proofs{0}, served{0}, arms{0}, armed{0};
};
Counts counts;
std::array<std::uint64_t, 7> reported{};
std::mutex reportMutex;
std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();

void reportLocked(const char* label) {
    const std::array<std::uint64_t, 7> now{counts.verdicts[0].load(), counts.verdicts[1].load(), counts.verdicts[2].load(), counts.proofs.load(), counts.served.load(), counts.arms.load(), counts.armed.load()};
    std::array<std::uint64_t, 7> d{};
    for (std::size_t i = 0; i < now.size(); ++i) {
        d[i] = now[i] - reported[i];
        reported[i] = now[i];
    }
    AgcDriver::ProfilePrint_nid_no_patch("[imageflags] %s: snapshot hits checked %llu: not armed %llu, dirty %llu, clean %llu; proofs run %llu, served %llu; arms %llu (armed %llu)%s\n", label, static_cast<unsigned long long>(d[0] + d[1] + d[2]), static_cast<unsigned long long>(d[0]), static_cast<unsigned long long>(d[1]), static_cast<unsigned long long>(d[2]), static_cast<unsigned long long>(d[3]), static_cast<unsigned long long>(d[4]), static_cast<unsigned long long>(d[5]), static_cast<unsigned long long>(d[6]), Verify() ? "; verify on" : "");
}

}

bool Enabled() {
    static const bool enabled = flag("APS5_V3E_IMAGE_FLAGS");
    return enabled;
}

bool Verify() {
    static const bool verify = flag("APS5_V3E_IMAGE_FLAGS_VERIFY");
    return verify;
}

void UseBackend(const Backend* use) {
    backend.store(use != nullptr ? use : &Unarmable, std::memory_order_release);
}

std::uint64_t Arm(std::uint64_t address, std::size_t bytes) {
    const auto token = backend.load(std::memory_order_acquire)->arm(address, bytes);
    counts.arms.fetch_add(1, std::memory_order_relaxed);
    if (token != 0) counts.armed.fetch_add(1, std::memory_order_relaxed);
    return token;
}

bool CleanSince(std::uint64_t address, std::size_t bytes, std::uint64_t token) {
    if (token == 0) return false;
    return backend.load(std::memory_order_acquire)->cleanSince(address, bytes, token);
}

bool CleanSinceAll(std::span<const DirtyQuery> queries) {
    if (queries.empty()) return false;
    return backend.load(std::memory_order_acquire)->cleanSinceAll(queries);
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
    if (!Enabled()) return;
    std::lock_guard lock(reportMutex);
    reportLocked(label);
}

void MaybeReport() {
    if (!Enabled() || ProfileWindows::FrameWindows()) return;
    std::unique_lock lock(reportMutex, std::try_to_lock);
    if (!lock.owns_lock() || !ProfileWindows::ClockDue(lastReport)) return;
    reportLocked("10s");
}

}
