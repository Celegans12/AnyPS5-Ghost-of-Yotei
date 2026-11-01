#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_REGISTRATIONPREBUILDQUEUE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_SHADERS_REGISTRATIONPREBUILDQUEUE_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace AgcDriver::DriverDetail {

class RegistrationPrebuildQueue {
public:
    struct Settings {
        bool waitForPresent = true;
        unsigned concurrency = 2;
        std::chrono::milliseconds fallback{120000};
        std::uint64_t budgetBytes = 0;
        std::chrono::milliseconds summary{10000};
        std::chrono::milliseconds demandQuiet{0};
        std::chrono::milliseconds stallLift{0};
        std::uint64_t liftOnlyWords = 0;
    };

    struct Hooks {
        std::function<bool()> demandActive;
        std::function<std::uint64_t()> usedBytes;
        std::function<void(std::size_t, std::uint64_t, std::uint64_t)> deferred;
        std::function<std::chrono::milliseconds()> demandIdleFor;
    };

    using Submitter = std::function<void(std::function<void()>)>;
    using ReleaseLog = std::function<void(std::size_t, const char*)>;

    static Settings SettingsFromEnvironment() {
        Settings settings;
        settings.concurrency = 0;
        if (const char* start = std::getenv("APS5_COMPUTE_REGISTRATION_START")) settings.waitForPresent = std::strcmp(start, "immediate") != 0;
        if (const char* delay = std::getenv("APS5_COMPUTE_REGISTRATION_FALLBACK_MS")) settings.fallback = std::chrono::milliseconds(std::strtoull(delay, nullptr, 10));
        if (const char* limit = std::getenv("APS5_COMPUTE_REGISTRATION_CONCURRENCY")) settings.concurrency = std::max(1u, static_cast<unsigned>(std::strtoul(limit, nullptr, 10)));
        if (const char* quiet = std::getenv("APS5_COMPUTE_REGISTRATION_DEMAND_QUIET_MS")) settings.demandQuiet = std::chrono::milliseconds(std::strtoull(quiet, nullptr, 10));
        if (const char* stall = std::getenv("APS5_COMPUTE_REGISTRATION_QUIET_LIFT_STALL_MS")) settings.stallLift = std::chrono::milliseconds(std::strtoull(stall, nullptr, 10));
        if (const char* words = std::getenv("APS5_COMPUTE_REGISTRATION_LIFT_ONLY_WORDS")) settings.liftOnlyWords = std::strtoull(words, nullptr, 10);
        return settings;
    }

    RegistrationPrebuildQueue(Settings settings, Submitter submit, ReleaseLog log = {}, Hooks hooks = {}) : settings(settings), submit(std::move(submit)), log(std::move(log)), hooks(std::move(hooks)), opened(!settings.waitForPresent) {}

    RegistrationPrebuildQueue(const RegistrationPrebuildQueue&) = delete;
    RegistrationPrebuildQueue& operator=(const RegistrationPrebuildQueue&) = delete;

    ~RegistrationPrebuildQueue() {
        Stop();
    }

    void Enqueue(std::uint64_t key, std::uint64_t cost, std::function<void()> task, std::uint64_t words = 0) {
        {
            std::lock_guard lock(mutex);
            if (stopped) return;
            pending.push_back({key, cost, std::move(task), words});
            if (!opened.load() && !timer.joinable()) timer = std::thread([this] { Fallback(); });
        }
        Pump();
    }

    void Open(const char* reason) {
        if (opened.load()) return;
        std::size_t queued = 0;
        {
            std::lock_guard lock(mutex);
            if (opened.exchange(true)) return;
            queued = pending.size();
        }
        wake.notify_all();
        if (log) log(queued, reason);
        Pump();
    }

    bool Promote(std::uint64_t key) {
        std::lock_guard lock(mutex);
        const auto found = std::find_if(pending.begin(), pending.end(), [&](const Item& item) { return item.key == key; });
        if (found == pending.end()) return false;
        pending.erase(found);
        return true;
    }

    void Poke() {
        Pump();
    }

    bool Presented(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now()) {
        if (settings.stallLift.count() == 0 || quietLifted.load()) return false;
        const auto ticks = now.time_since_epoch().count();
        const auto previous = lastPresent.exchange(ticks);
        if (previous == 0 || std::chrono::steady_clock::duration(ticks - previous) < settings.stallLift) return false;
        if (quietLifted.exchange(true)) return false;
        Pump();
        return true;
    }

    bool Busy() const {
        std::lock_guard lock(mutex);
        return !pending.empty() || running != 0;
    }

    bool Opened() const {
        return opened.load();
    }

    void Stop() {
        std::thread joined[2];
        {
            std::lock_guard lock(mutex);
            stopped = true;
            pending.clear();
            joined[0].swap(timer);
            joined[1].swap(retry);
        }
        wake.notify_all();
        for (auto& thread : joined) {
            if (thread.joinable() && thread.get_id() != std::this_thread::get_id()) thread.join();
            else if (thread.joinable()) thread.detach();
        }
    }

    std::size_t Queued() const {
        std::lock_guard lock(mutex);
        return pending.size();
    }

    unsigned PeakRunning() const {
        return peak.load();
    }

private:
    void Fallback() {
        std::unique_lock lock(mutex);
        const bool woken = wake.wait_for(lock, settings.fallback, [&] { return opened.load() || stopped; });
        lock.unlock();
        if (!woken) Open("fallback");
    }

    void ArmRetry(std::chrono::steady_clock::time_point at) {
        if (!retryAt || at < *retryAt) retryAt = at;
        if (!retry.joinable()) retry = std::thread([this] { RetryLoop(); });
        wake.notify_all();
    }

    void RetryLoop() {
        std::unique_lock lock(mutex);
        while (!stopped) {
            if (!retryAt) {
                wake.wait(lock);
                continue;
            }
            const auto at = *retryAt;
            if (wake.wait_until(lock, at, [&] { return stopped || !retryAt || *retryAt != at; })) continue;
            retryAt.reset();
            lock.unlock();
            Pump();
            lock.lock();
        }
    }

    struct Item {
        std::uint64_t key = 0;
        std::uint64_t cost = 0;
        std::function<void()> task;
        std::uint64_t words = 0;
    };

    struct Summary {
        std::size_t queued = 0;
        std::uint64_t demand = 0;
        std::uint64_t memory = 0;
    };

    void Pump() {
        if (!opened.load()) return;
        for (;;) {
            Item item;
            std::optional<Summary> summary;
            bool started = false;
            {
                std::lock_guard lock(mutex);
                if (stopped || pending.empty() || (settings.concurrency != 0 && running >= settings.concurrency)) return;
                bool blocked = false;
                const bool holding = settings.demandQuiet.count() != 0 && !quietLifted.load();
                const auto startable = [&](const Item& queued) { return !holding || settings.stallLift.count() == 0 || settings.liftOnlyWords == 0 || queued.words < settings.liftOnlyWords; };
                auto chosen = pending.begin();
                const auto found = std::find_if(pending.begin(), pending.end(), startable);
                if (found != pending.end()) chosen = found;
                const auto& candidate = *chosen;
                if (hooks.demandActive && hooks.demandActive()) {
                    ++deferredForDemand;
                    blocked = true;
                } else if (!startable(candidate)) {
                    ++deferredForDemand;
                    blocked = true;
                } else if (holding && hooks.demandIdleFor && hooks.demandIdleFor() < settings.demandQuiet) {
                    ++deferredForDemand;
                    blocked = true;
                    ArmRetry(std::chrono::steady_clock::now() + (settings.demandQuiet - hooks.demandIdleFor()));
                } else if (running != 0 && settings.budgetBytes != 0) {
                    const auto used = std::max(runningBytes, hooks.usedBytes ? hooks.usedBytes() : std::uint64_t{0});
                    if (used + candidate.cost > settings.budgetBytes) {
                        ++deferredForMemory;
                        blocked = true;
                    }
                }
                if (blocked) {
                    const auto now = std::chrono::steady_clock::now();
                    if (!summaryAt) summaryAt = now;
                    if (now - *summaryAt >= settings.summary) {
                        summary = Summary{pending.size(), deferredForDemand, deferredForMemory};
                        deferredForDemand = 0;
                        deferredForMemory = 0;
                        summaryAt = now;
                    }
                } else {
                    item = std::move(*chosen);
                    pending.erase(chosen);
                    ++running;
                    runningBytes += item.cost;
                    unsigned now = running;
                    unsigned seen = peak.load();
                    while (now > seen && !peak.compare_exchange_weak(seen, now)) {}
                    started = true;
                }
            }
            if (summary && hooks.deferred) hooks.deferred(summary->queued, summary->demand, summary->memory);
            if (!started) return;
            submit([this, cost = item.cost, task = std::move(item.task)] {
                task();
                {
                    std::lock_guard lock(mutex);
                    --running;
                    runningBytes -= std::min(runningBytes, cost);
                }
                Pump();
            });
        }
    }

    Settings settings;
    Submitter submit;
    ReleaseLog log;
    Hooks hooks;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::deque<Item> pending;
    std::atomic<bool> opened;
    std::atomic<bool> quietLifted{false};
    std::atomic<std::chrono::steady_clock::rep> lastPresent{0};
    std::atomic<unsigned> peak{0};
    std::thread timer;
    std::thread retry;
    std::optional<std::chrono::steady_clock::time_point> retryAt;
    unsigned running = 0;
    std::uint64_t runningBytes = 0;
    std::uint64_t deferredForDemand = 0;
    std::uint64_t deferredForMemory = 0;
    std::optional<std::chrono::steady_clock::time_point> summaryAt;
    bool stopped = false;
};

}

#endif
