#include "prx/libSceAgcDriver/Submit/include/Acb.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include "prx/libkernel/Time/include/Time.hpp"

// Diagnostic (APS5_TRACE_EOP=1 with APS5_TRACE_WAITS=1, v3 exec 11-07): who submits compute work: per queue, submitting
// thread and guest call site, and the thread's last kernel wait before the submit (kind, site, how long, and the gap
// from its end to the submit); [acb-submits] every 10 s.
namespace AgcDriver::DriverDetail {
// PacketMemory.cpp (APS5_TRACE_EOP).
void TraceFrameMarks(std::int64_t* flipNs, std::int64_t* eop0Ns, std::uint64_t* flips);
}

namespace {
bool traceSubmits() {
    static const bool enabled = std::getenv("APS5_TRACE_EOP") != nullptr;
    return enabled;
}
struct SubmitSite { std::uint64_t count = 0, waitedNs = 0, gapNs = 0, maxGapNs = 0, noWait = 0; };
// [frame-chain]: per queue, its first submit after each queue-0 flip the worker reached: the time from that flip and
// from the last queue-0 EOP packet before it, and how many flips passed since the queue's previous first submit.
struct ChainQueue { std::uint64_t firsts = 0, flipNs = 0, minFlipNs = ~0ull, maxFlipNs = 0, eopNs = 0, minEopNs = ~0ull, maxEopNs = 0, skipped = 0, lastFlips = 0; };
std::map<std::uint32_t, ChainQueue> chainQueues;
std::mutex submitMutex;
std::map<std::tuple<std::uint32_t, unsigned long, const void*, const char*, const void*>, SubmitSite> submitSites;
std::chrono::steady_clock::time_point submitReport = std::chrono::steady_clock::now();
void traceSubmit(std::uint32_t queue, const void* caller) {
    if (!traceSubmits()) return;
    const char* kind = nullptr;
    const void* waitSite = nullptr;
    std::uint64_t waited = 0, ended = 0;
    const bool any = KernelLastWait_nid_postfix(&kind, &waitSite, &waited, &ended);
    const auto now = std::chrono::steady_clock::now();
    const auto nowNs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    std::lock_guard lock(submitMutex);
    {
        std::int64_t flipAt = 0, eopAt = 0;
        std::uint64_t flips = 0;
        AgcDriver::DriverDetail::TraceFrameMarks(&flipAt, &eopAt, &flips);
        auto& chain = chainQueues[queue];
        if (flips != 0 && flips != chain.lastFlips) {
            if (chain.lastFlips != 0 && flips > chain.lastFlips + 1) chain.skipped += flips - chain.lastFlips - 1;
            chain.lastFlips = flips;
            ++chain.firsts;
            const auto nowSigned = static_cast<std::int64_t>(nowNs);
            const auto fromFlip = static_cast<std::uint64_t>(std::max<std::int64_t>(0, nowSigned - flipAt));
            chain.flipNs += fromFlip;
            chain.minFlipNs = std::min(chain.minFlipNs, fromFlip);
            chain.maxFlipNs = std::max(chain.maxFlipNs, fromFlip);
            if (eopAt != 0) {
                const auto fromEop = static_cast<std::uint64_t>(std::max<std::int64_t>(0, nowSigned - eopAt));
                chain.eopNs += fromEop;
                chain.minEopNs = std::min(chain.minEopNs, fromEop);
                chain.maxEopNs = std::max(chain.maxEopNs, fromEop);
            }
        }
    }
    auto& site = submitSites[{queue, KernelTraceThreadId_nid_postfix(), caller, any ? kind : "none", any ? waitSite : nullptr}];
    ++site.count;
    if (any) {
        site.waitedNs += waited;
        const auto gap = nowNs > ended ? nowNs - ended : 0;
        site.gapNs += gap;
        site.maxGapNs = std::max(site.maxGapNs, gap);
    } else {
        ++site.noWait;
    }
    if (now - submitReport < std::chrono::seconds(10)) return;
    submitReport = now;
    std::string line = "[acb-submits] 10 s (queue tid @submit-site: count; last wait before it kind@site avg ms, gap from its end to the submit avg / max ms):";
    for (const auto& [key, s] : submitSites) {
        char text[256];
        const auto n = static_cast<double>(s.count);
        std::snprintf(text, sizeof(text), " | 0x%x tid %lu @%p: %llu; %s@%p %.1f ms, gap %.1f / %.1f ms", std::get<0>(key), std::get<1>(key), std::get<2>(key), static_cast<unsigned long long>(s.count), std::get<3>(key), std::get<4>(key), s.waitedNs / 1e6 / n, s.gapNs / 1e6 / n, s.maxGapNs / 1e6);
        line += text;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    submitSites.clear();
    std::string chainLine = "[frame-chain] 10 s, per queue, its first submit after each queue-0 flip the worker reached (count; from that flip avg / min / max ms; from the last queue-0 EOP packet avg / min / max ms; flips with no submit in between):";
    for (auto& [q, c] : chainQueues) {
        if (c.firsts == 0) continue;
        char text[256];
        const auto n = static_cast<double>(c.firsts);
        std::snprintf(text, sizeof(text), " | 0x%x %llu; %.1f / %.1f / %.1f; %.1f / %.1f / %.1f; %llu", q, static_cast<unsigned long long>(c.firsts), c.flipNs / 1e6 / n, c.minFlipNs / 1e6, c.maxFlipNs / 1e6, c.eopNs / 1e6 / n, c.minEopNs == ~0ull ? 0.0 : c.minEopNs / 1e6, c.maxEopNs / 1e6, static_cast<unsigned long long>(c.skipped));
        chainLine += text;
        const auto keep = c.lastFlips;
        c = {};
        c.lastFlips = keep;
    }
    std::fprintf(stderr, "%s\n", chainLine.c_str());
}
}

namespace AgcDriver {
void TraceGuestSubmit(std::uint32_t queue, const void* caller) { traceSubmit(queue, caller); }
}
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

extern "C" {

int APS5_VABI sceAgcDriverSubmitAcb(uint32_t queue, const Packet* packet) {
    if (queue < 0x20 || queue >= 0x58) {
        throw std::runtime_error(std::string(__func__) + ": unsupported compute queue");
    }
    traceSubmit(queue, __builtin_return_address(0));
    AgcDriver::Submit(packet, queue);
    return 0;
}

int APS5_VABI sceAgcDriverSubmitMultiAcbs(uint32_t queue, uint32_t* const* acbs, const uint32_t* sizes_in_dwords, uint32_t count) {
    if (count == 0) return 0;
    if (queue < 0x20 || queue >= 0x58) {
        throw std::runtime_error(std::string(__func__) + ": unsupported compute queue");
    }
    traceSubmit(queue, __builtin_return_address(0));
    return AgcDriver::SubmitMany(acbs, sizes_in_dwords, count, queue);
}

}
