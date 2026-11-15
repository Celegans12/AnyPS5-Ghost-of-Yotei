#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/MovedRead.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace AgcDriver::DriverDetail {

namespace {

std::atomic<std::uint64_t> guardedReads{0};
std::atomic<std::uint64_t> checkedReads{0};

bool guardedMovedReads() {
    static const bool enabled = std::getenv("APS5_NO_GUARDED_MOVED_READS") == nullptr;
    return enabled;
}

}

GuestMemory::Compare CopyMoved(std::uint64_t address, std::span<std::byte> out) {
#ifdef _WIN32
    if (guardedMovedReads() && !out.empty() && address != 0 && out.size() <= std::numeric_limits<std::uintptr_t>::max() - address && GuestArena::GuestArenaCopyOwned_nid_no_patch(static_cast<std::uintptr_t>(address), out.data(), out.size())) {
        guardedReads.fetch_add(1, std::memory_order_relaxed);
        return GuestMemory::Compare::Equal;
    }
#endif
    checkedReads.fetch_add(1, std::memory_order_relaxed);
    return GuestMemory::CopyMapped(address, out);
}

MovedReadCounts MovedReads() {
    return {guardedReads.load(std::memory_order_relaxed), checkedReads.load(std::memory_order_relaxed)};
}

std::string MovedReadsReport() {
    static std::mutex reportMutex;
    static MovedReadCounts reported;
    const auto now = MovedReads();
    std::lock_guard lock(reportMutex);
    char line[192];
    std::snprintf(line, sizeof(line), "[moved-reads] (10 s): %llu runs at moved or unproved addresses copied under the read guard, %llu through the checked system copy\n", static_cast<unsigned long long>(now.guarded - reported.guarded), static_cast<unsigned long long>(now.checked - reported.checked));
    reported = now;
    return line;
}

}
