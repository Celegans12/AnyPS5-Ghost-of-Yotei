#include "prx/libSceAgcDriver/Graphics/include/InflightBuilds.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using AgcDriver::Graphics::InflightBuilds;

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::shared_future<void> Pending(std::promise<void>& promise) {
    return promise.get_future().share();
}

void firstClaimOwns() {
    InflightBuilds builds;
    std::promise<void> promise;
    const auto claim = builds.Acquire(7, Pending(promise));
    Require(claim.owner && !claim.existing.valid(), "the first claim does not own the key");
    Require(builds.Find(7).valid() && builds.Size() == 1, "the claimed key is not in flight");
    Require(!builds.Find(8).valid(), "an unclaimed key is in flight");
}

void secondClaimAttaches() {
    InflightBuilds builds;
    std::promise<void> first;
    std::promise<void> second;
    Require(builds.Acquire(7, Pending(first)).owner, "the first claim does not own the key");
    const auto claim = builds.Acquire(7, Pending(second));
    Require(!claim.owner && claim.existing.valid(), "a duplicate claim does not attach");
    Require(claim.existing.wait_for(std::chrono::seconds(0)) == std::future_status::timeout, "the attached future is ready early");
    first.set_value();
    Require(claim.existing.wait_for(std::chrono::seconds(0)) == std::future_status::ready, "the attached future does not follow the owner");
    Require(builds.Size() == 1, "a duplicate claim added an entry");
}

void releaseAllowsRebuild() {
    InflightBuilds builds;
    std::promise<void> first;
    std::promise<void> second;
    Require(builds.Acquire(7, Pending(first)).owner, "the first claim does not own the key");
    builds.Release(7);
    Require(builds.Size() == 0 && !builds.Find(7).valid(), "a released key is still in flight");
    Require(builds.Acquire(7, Pending(second)).owner, "a released key cannot be claimed again");
    builds.Release(9);
    Require(builds.Size() == 1, "releasing an unknown key changed the map");
}

void failedBuildReleases() {
    InflightBuilds builds;
    std::packaged_task<void()> job([&builds] {
        try {
            throw std::runtime_error("build failed");
        } catch (...) {
        }
        builds.Release(5);
    });
    const auto future = job.get_future().share();
    Require(builds.Acquire(5, future).owner, "the claim does not own the key");
    std::thread worker([&job] { job(); });
    future.wait();
    worker.join();
    Require(builds.Size() == 0, "a failed build leaves its key in flight");
}

void oneOwnerUnderContention() {
    InflightBuilds builds;
    std::promise<void> promise;
    const auto future = Pending(promise);
    std::atomic<int> owners{0};
    std::atomic<int> attached{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            const auto claim = builds.Acquire(42, future);
            (claim.owner ? owners : attached).fetch_add(1);
        });
    }
    for (auto& thread : threads) thread.join();
    Require(owners.load() == 1 && attached.load() == 15, "contended claims do not give exactly one owner");
}

}

int main() {
    try {
        firstClaimOwns();
        secondClaimAttaches();
        releaseAllowsRebuild();
        failedBuildReleases();
        oneOwnerUnderContention();
        std::puts("inflight builds tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "inflight builds test failed: %s\n", error.what());
        return 1;
    }
}
