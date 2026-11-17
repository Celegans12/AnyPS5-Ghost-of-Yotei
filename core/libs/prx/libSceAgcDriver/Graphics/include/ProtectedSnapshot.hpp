#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PROTECTEDSNAPSHOT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PROTECTEDSNAPSHOT_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

namespace Hazards {

constexpr std::size_t Slots = 8;
constexpr std::size_t RetireBatch = 64;

struct alignas(64) Record {
    std::array<std::atomic<const void*>, Slots> slots{};
    std::atomic<bool> taken{false};
    Record* next = nullptr;
    std::size_t depth = 0;
};

Record& Mine();
void Protected(std::vector<const void*>& out);
[[noreturn]] void TooDeep();
std::size_t RecordsForTest();

class Guard {
public:
    Guard() : record(Mine()), level(record.depth) {
        if (level >= Slots) TooDeep();
        record.depth = level + 1;
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    ~Guard() {
        record.slots[level].store(nullptr, std::memory_order_release);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        record.depth = level;
    }
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;

    template<typename TNode>
    const TNode* Protect(const std::atomic<const TNode*>& source) {
        auto& slot = record.slots[level];
        const TNode* node = source.load(std::memory_order_relaxed);
        for (;;) {
            slot.store(node, std::memory_order_seq_cst);
            const TNode* again = source.load(std::memory_order_seq_cst);
            if (again == node) return node;
            node = again;
        }
    }

private:
    Record& record;
    std::size_t level;
};

}

template<typename T>
class ProtectedSnapshot {
public:
    ProtectedSnapshot() = default;
    ~ProtectedSnapshot() {
        delete current.load(std::memory_order_acquire);
        for (const auto* node : retired) delete node;
    }
    ProtectedSnapshot(const ProtectedSnapshot&) = delete;
    ProtectedSnapshot& operator=(const ProtectedSnapshot&) = delete;

    std::shared_ptr<const T> Load() const {
        Hazards::Guard guard;
        const auto* node = guard.Protect(current);
        return node != nullptr ? node->value : nullptr;
    }

    template<typename TVisit>
    decltype(auto) Read(TVisit&& visit) const {
        Hazards::Guard guard;
        const auto* node = guard.Protect(current);
        return std::forward<TVisit>(visit)(node != nullptr ? node->value.get() : nullptr);
    }

    void Store(std::shared_ptr<const T> value) {
        const Node* fresh = value != nullptr ? new Node{std::move(value)} : nullptr;
        if (const Node* previous = current.exchange(fresh, std::memory_order_seq_cst); previous != nullptr) retire(previous);
    }

    std::size_t RetainedForTest() const {
        std::lock_guard lock(retiring);
        return retired.size();
    }

private:
    struct Node {
        std::shared_ptr<const T> value;
    };

    void retire(const Node* node) {
        std::vector<const Node*> freed;
        {
            std::lock_guard lock(retiring);
            retired.push_back(node);
            if (retired.size() < Hazards::RetireBatch) return;
            Hazards::Protected(scanned);
            const auto kept = std::partition(retired.begin(), retired.end(), [&](const Node* candidate) { return std::binary_search(scanned.begin(), scanned.end(), static_cast<const void*>(candidate)); });
            freed.assign(kept, retired.end());
            retired.erase(kept, retired.end());
        }
        for (const auto* each : freed) delete each;
    }

    std::atomic<const Node*> current{nullptr};
    mutable std::mutex retiring;
    std::vector<const Node*> retired;
    std::vector<const void*> scanned;
};

}

#endif
