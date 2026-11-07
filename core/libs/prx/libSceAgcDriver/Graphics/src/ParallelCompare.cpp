#include "prx/libSceAgcDriver/Graphics/include/ParallelCompare.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {
namespace {

std::atomic<unsigned> runningHelpers{0};

class ComparePool {
public:
    explicit ComparePool(unsigned count) {
        stopping = count == 0;
        for (unsigned i = 0; i < count; ++i) {
            workers.emplace_back([this] { work(); });
            runningHelpers.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    bool Run(std::size_t count, HelperJob job, void* context) {
        std::unique_lock caller(callers, std::try_to_lock);
        if (!caller.owns_lock()) return false;
        {
            std::unique_lock lock(mutex);
            if (stopping) return false;
            finished.wait(lock, [&] { return working == 0; });
            jobCount = count;
            jobRun = job;
            jobContext = context;
            failure = nullptr;
            next.store(0, std::memory_order_relaxed);
            done.store(0, std::memory_order_relaxed);
            ++generation;
        }
        wake.notify_all();
        drain(count, job, context);
        std::unique_lock lock(mutex);
        finished.wait(lock, [&] { return working == 0 && done.load(std::memory_order_acquire) == count; });
        jobCount = 0;
        jobRun = nullptr;
        jobContext = nullptr;
        if (failure != nullptr) std::rethrow_exception(std::exchange(failure, nullptr));
        return true;
    }

    void Stop() {
        std::vector<std::thread> joined;
        {
            std::lock_guard caller(callers);
            std::lock_guard lock(mutex);
            stopping = true;
            joined.swap(workers);
        }
        wake.notify_all();
        for (auto& worker : joined) {
            if (worker.get_id() == std::this_thread::get_id()) worker.detach();
            else worker.join();
        }
    }

private:
    // Every index taken counts as done, also one whose job threw: Run waits for the count and rethrows
    // the first failure on the calling thread.
    void drain(std::size_t count, HelperJob job, void* context) {
        std::size_t ran = 0;
        for (auto index = next.fetch_add(1, std::memory_order_relaxed); index < count; index = next.fetch_add(1, std::memory_order_relaxed)) {
            try {
                job(context, index);
            } catch (...) {
                std::lock_guard lock(mutex);
                if (failure == nullptr) failure = std::current_exception();
            }
            ++ran;
        }
        if (ran != 0) done.fetch_add(ran, std::memory_order_acq_rel);
    }

    void work() {
        std::uint64_t seen = 0;
        std::unique_lock lock(mutex);
        for (;;) {
            wake.wait(lock, [&] { return generation != seen || stopping; });
            if (stopping) break;
            seen = generation;
            const auto count = jobCount;
            const auto job = jobRun;
            const auto context = jobContext;
            ++working;
            lock.unlock();
            if (job != nullptr) drain(count, job, context);
            lock.lock();
            --working;
            if (working == 0) finished.notify_all();
        }
        runningHelpers.fetch_sub(1, std::memory_order_acq_rel);
    }

    std::mutex callers;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable finished;
    std::uint64_t generation = 0;
    unsigned working = 0;
    std::size_t jobCount = 0;
    HelperJob jobRun = nullptr;
    void* jobContext = nullptr;
    std::exception_ptr failure;
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::vector<std::thread> workers;
    bool stopping = false;
};

std::mutex& poolLifecycle() {
    static std::mutex lifecycle;
    return lifecycle;
}

bool& poolStopped() {
    static bool stopped = false;
    return stopped;
}

std::atomic<ComparePool*>& createdPool() {
    static std::atomic<ComparePool*> created{nullptr};
    return created;
}

ComparePool* sharedPool() {
    static ComparePool* const pool = [] {
        std::lock_guard lifecycle(poolLifecycle());
        auto* created = new ComparePool(poolStopped() ? 0u : CompareHelpers());
        createdPool().store(created, std::memory_order_release);
        return created;
    }();
    return pool;
}

}

unsigned CompareHelpers() {
    static const unsigned helpers = std::clamp(std::thread::hardware_concurrency() / 8u, 1u, 4u);
    return helpers;
}

bool RunOnCompareHelpers(std::size_t count, HelperJob job, void* context) {
    if (count < 2) return false;
    return sharedPool()->Run(count, job, context);
}

void CompareSpansFrom(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal, std::size_t parallelBytes) {
    struct Job {
        std::span<const CompareSpan> spans;
        std::span<std::uint8_t> equal;
    } job{spans, equal};
    const auto compare = [](void* context, std::size_t index) {
        const auto& state = *static_cast<const Job*>(context);
        const auto& span = state.spans[index];
        state.equal[index] = std::memcmp(span.first, span.second, span.bytes) == 0 ? 1 : 0;
    };
    std::size_t total = 0;
    for (const auto& span : spans) total += span.bytes;
    if (spans.size() > 1 && total >= parallelBytes && RunOnCompareHelpers(spans.size(), compare, &job)) return;
    for (std::size_t index = 0; index < spans.size(); ++index) compare(&job, index);
}

void StopCompareHelpers() {
    ComparePool* pool = nullptr;
    {
        std::lock_guard lifecycle(poolLifecycle());
        poolStopped() = true;
        pool = createdPool().load(std::memory_order_acquire);
    }
    if (pool != nullptr) pool->Stop();
}

unsigned CompareHelpersRunning() {
    return runningHelpers.load(std::memory_order_acquire);
}

void CompareSpans(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal) {
    CompareSpansFrom(spans, equal, std::size_t{1} << 20u);
}

}
