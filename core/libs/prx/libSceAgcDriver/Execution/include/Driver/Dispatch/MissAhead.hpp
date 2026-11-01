#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_MISSAHEAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCH_MISSAHEAD_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace AgcDriver::DriverDetail {

class MissAheadPlan {
public:
    enum class Skip { Cached, Duplicate, Unprepared, Unregistered, CaptureFailed, Listed, Kernel, Failed, Indirect };

    struct Counts {
        std::size_t visited = 0, started = 0, cached = 0, duplicate = 0, unprepared = 0, unregistered = 0, captureFailed = 0, reached = 0, listed = 0, kernel = 0, failed = 0, indirect = 0;
    };

    MissAheadPlan(std::size_t buildCap, std::size_t visitCap) : buildCap(buildCap), visitCap(visitCap) {}

    bool BuildsFull() const { return counts.started >= buildCap; }

    bool Visit() {
        if (counts.visited >= visitCap) return false;
        ++counts.visited;
        return true;
    }

    bool Admit(std::uint64_t key) { return keys.insert(key).second; }

    void Started() { ++counts.started; }

    void Reach() { ++counts.reached; }

    void Note(Skip reason) {
        switch (reason) {
        case Skip::Cached: ++counts.cached; break;
        case Skip::Duplicate: ++counts.duplicate; break;
        case Skip::Unprepared: ++counts.unprepared; break;
        case Skip::Unregistered: ++counts.unregistered; break;
        case Skip::CaptureFailed: ++counts.captureFailed; break;
        case Skip::Listed: ++counts.listed; break;
        case Skip::Kernel: ++counts.kernel; break;
        case Skip::Failed: ++counts.failed; break;
        case Skip::Indirect: ++counts.indirect; break;
        }
    }

    const Counts& Totals() const { return counts; }

private:
    std::size_t buildCap;
    std::size_t visitCap;
    Counts counts;
    std::set<std::uint64_t> keys;
};

class MissAheadRange {
public:
    bool Covered(std::uint64_t submission, std::uint32_t queue, std::size_t from) const {
        return recorded && submission == lastSubmission && queue == lastQueue && from < lastEnd;
    }

    void Record(std::uint64_t submission, std::uint32_t queue, std::size_t end) {
        recorded = true;
        lastSubmission = submission;
        lastQueue = queue;
        lastEnd = end;
    }

private:
    bool recorded = false;
    std::uint64_t lastSubmission = 0;
    std::uint32_t lastQueue = 0;
    std::size_t lastEnd = 0;
};

class MissAheadKnown {
public:
    void Insert(std::uint64_t key) {
        std::lock_guard lock(mutex);
        if (keys.size() >= Retained) keys.clear();
        keys.insert(key);
    }

    bool Contains(std::uint64_t key) const {
        std::lock_guard lock(mutex);
        return keys.contains(key);
    }

    static constexpr std::size_t Retained = 16384;

private:
    mutable std::mutex mutex;
    std::unordered_set<std::uint64_t> keys;
};

class MissAheadLedger {
public:
    struct Totals {
        std::uint64_t started = 0, hit = 0, waited = 0;
        std::uint64_t Unused() const { return started - hit - waited; }
    };

    void Started(std::uint64_t key) {
        std::lock_guard lock(mutex);
        if (pending.size() >= Retained) pending.clear();
        pending[key] = false;
        ++totals.started;
        any.store(true, std::memory_order_relaxed);
    }

    void Finished(std::uint64_t key) {
        std::lock_guard lock(mutex);
        if (const auto found = pending.find(key); found != pending.end()) found->second = true;
    }

    bool Used(std::uint64_t key) {
        if (!any.load(std::memory_order_relaxed)) return false;
        std::lock_guard lock(mutex);
        const auto found = pending.find(key);
        if (found == pending.end()) return false;
        if (found->second) ++totals.hit;
        else ++totals.waited;
        pending.erase(found);
        return true;
    }

    bool Active() const { return any.load(std::memory_order_relaxed); }

    Totals Snapshot() const {
        std::lock_guard lock(mutex);
        return totals;
    }

    static constexpr std::size_t Retained = 4096;

private:
    mutable std::mutex mutex;
    std::unordered_map<std::uint64_t, bool> pending;
    Totals totals;
    std::atomic<bool> any{false};
};

}

#endif
