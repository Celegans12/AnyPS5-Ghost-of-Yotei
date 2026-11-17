#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INFLIGHTBUILDS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_INFLIGHTBUILDS_HPP

#include <cstdint>
#include <future>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace AgcDriver::Graphics {

class InflightBuilds {
public:
    struct Claim {
        bool owner = false;
        std::shared_future<void> existing;
    };

    Claim Acquire(std::uint64_t key, const std::shared_future<void>& build) {
        std::lock_guard lock(mutex_);
        const auto [it, inserted] = builds_.try_emplace(key, build);
        if (inserted) return {true, {}};
        return {false, it->second};
    }

    std::shared_future<void> Find(std::uint64_t key) const {
        std::lock_guard lock(mutex_);
        const auto it = builds_.find(key);
        return it != builds_.end() ? it->second : std::shared_future<void>{};
    }

    void Release(std::uint64_t key) {
        std::lock_guard lock(mutex_);
        builds_.erase(key);
    }

    std::size_t Size() const {
        std::lock_guard lock(mutex_);
        return builds_.size();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, std::shared_future<void>> builds_;
};

}

#endif
