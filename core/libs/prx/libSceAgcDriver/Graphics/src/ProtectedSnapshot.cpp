#include "prx/libSceAgcDriver/Graphics/include/ProtectedSnapshot.hpp"
#include <cstdio>
#include <cstdlib>

namespace AgcDriver::Graphics::Hazards {

namespace {

std::atomic<Record*> head{nullptr};

struct ThreadRecord {
    Record* record = nullptr;
    bool ownerGone = false;
};

constinit thread_local ThreadRecord mine{};

struct Owner {
    Record* record = nullptr;
    Owner() = default;
    Owner(const Owner&) = delete;
    Owner& operator=(const Owner&) = delete;
    ~Owner() {
        if (record == nullptr) return;
        for (auto& slot : record->slots) slot.store(nullptr, std::memory_order_relaxed);
        record->depth = 0;
        mine.record = nullptr;
        mine.ownerGone = true;
        record->taken.store(false, std::memory_order_release);
    }
};

Record* Take() {
    for (auto* record = head.load(std::memory_order_acquire); record != nullptr; record = record->next) {
        bool expected = false;
        if (!record->taken.load(std::memory_order_relaxed) && record->taken.compare_exchange_strong(expected, true, std::memory_order_acquire, std::memory_order_relaxed)) return record;
    }
    auto* record = new Record();
    record->taken.store(true, std::memory_order_relaxed);
    record->next = head.load(std::memory_order_relaxed);
    while (!head.compare_exchange_weak(record->next, record, std::memory_order_release, std::memory_order_relaxed)) {}
    return record;
}

}

Record& Mine() {
    if (auto* record = mine.record; record != nullptr) [[likely]] return *record;
    auto* record = Take();
    mine.record = record;
    if (!mine.ownerGone) {
        thread_local Owner owner;
        owner.record = record;
    }
    return *record;
}

void Protected(std::vector<const void*>& out) {
    out.clear();
    for (auto* record = head.load(std::memory_order_acquire); record != nullptr; record = record->next) {
        for (const auto& slot : record->slots) {
            if (const auto* value = slot.load(std::memory_order_seq_cst); value != nullptr) out.push_back(value);
        }
    }
    std::sort(out.begin(), out.end());
}

void TooDeep() {
    std::fprintf(stderr, "[snapshot] more than %zu protected snapshot reads are nested on one thread (signal handlers entering the driver inside a read); stopping\n", Slots);
    std::abort();
}

std::size_t RecordsForTest() {
    std::size_t count = 0;
    for (auto* record = head.load(std::memory_order_acquire); record != nullptr; record = record->next) ++count;
    return count;
}

}
