#include "prx/libSceAgcDriver/Core/include/OpHost.hpp"
#include "prx/libSceAgcDriver/Core/include/Op.hpp"
#include "prx/libSceAgcDriver/Core/include/OpQueue.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/FlatTransport.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>

namespace AgcDriver::Core {

namespace {

using Ledger = DriverDetail::WideLedger;

std::uintptr_t self() {
    static thread_local char mark = 0;
    return reinterpret_cast<std::uintptr_t>(&mark);
}

struct Meta {
    std::uint64_t gates = 0;
    std::uint64_t collectEpoch = 0;
};

}

struct OpHost::State {
    explicit State(Settings settings)
        : depth(std::max<std::size_t>(settings.depth, 2)),
          verifies(settings.verifies),
          turns(settings.turns),
          queue(OpQueue::Settings{depth, std::max<std::size_t>(depth / 4, 1), 0, settings.verifies && !settings.turns}),
          ledger(depth, settings.verifies, settings.counts),
          mask(std::bit_ceil(depth) - 1),
          metas(std::make_unique<Meta[]>(mask + 1)) {}

    const std::size_t depth;
    const bool verifies;
    const bool turns;
    OpQueue queue;
    Ledger ledger;
    const std::uint64_t mask;
    const std::unique_ptr<Meta[]> metas;
    std::atomic<std::uintptr_t> producer{0};
    std::atomic<bool> closed{false};

    alignas(64) std::atomic<std::uint64_t> posted{0};
    std::uint64_t gates = 0;
    std::atomic<std::uint64_t> gatesQueued{0};
    std::atomic<std::uint64_t> roomWaits{0};
    std::atomic<std::uint64_t> wakes{0};

    alignas(64) std::atomic<std::uint64_t> passed{0};
    alignas(64) std::atomic<bool> sleeping{false};
};

OpHost::OpHost(Settings settings) : state(std::make_unique<State>(settings)) {}

OpHost::~OpHost() = default;

bool OpHost::Post(const Op& op, std::span<const Range> writes, std::span<const Range> reads, std::uint64_t collectEpoch) {
    auto& host = *state;
    if (!host.turns) {
        const auto me = self();
        auto owner = host.producer.load(std::memory_order_relaxed);
        if (owner != me) {
            if (owner != 0 || !host.producer.compare_exchange_strong(owner, me, std::memory_order_acq_rel)) throw std::logic_error("op host: a second thread posts operations (APS5_CORE_OPS)");
        }
    }
    Op* slot = host.queue.Stage();
    while (slot == nullptr) {
        if (host.queue.Closed() || host.closed.load(std::memory_order_acquire)) return false;
        host.roomWaits.store(host.roomWaits.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        host.queue.AwaitRoom();
        slot = host.queue.Stage();
    }
    const auto position = host.posted.load(std::memory_order_relaxed);
    *slot = op;
    host.metas[position & host.mask] = Meta{host.gates, collectEpoch};
    Ledger::Entry entry;
    entry.writes = writes;
    entry.reads = reads;
    host.ledger.Publish(entry);
    host.queue.Publish();
    host.posted.store(position + 1, std::memory_order_seq_cst);
    return true;
}

bool OpHost::Producing() const {
    return state->producer.load(std::memory_order_relaxed) == self();
}

OpHost::Gate OpHost::Queue() {
    auto& host = *state;
    const auto number = ++host.gates;
    host.gatesQueued.store(number, std::memory_order_relaxed);
    return {number, host.posted.load(std::memory_order_relaxed)};
}

std::uint64_t OpHost::Posted() const {
    return state->posted.load(std::memory_order_seq_cst);
}

std::uint64_t OpHost::Retired() const {
    return state->ledger.Retired();
}

std::size_t OpHost::Pending(std::uint64_t before) const {
    const auto& host = *state;
    const auto first = host.ledger.Retired();
    const auto last = std::min(host.ledger.Published(), before);
    return last > first ? static_cast<std::size_t>(last - first) : 0;
}

bool OpHost::Overlaps(std::span<const Range> ranges, std::uint64_t before) const {
    const auto& host = *state;
    if (std::min(host.ledger.Published(), before) <= host.ledger.Retired()) return false;
    return host.ledger.Overlaps(ranges, before);
}

bool OpHost::OverlapsSince(std::uint64_t first, std::span<const Range> ranges, std::uint64_t before) const {
    const auto& host = *state;
    if (std::min(host.ledger.Published(), before) <= first) return false;
    return host.ledger.OverlapsSince(first, ranges, before);
}

bool OpHost::Reads(std::uint64_t address, std::uint64_t end) const {
    const auto& host = *state;
    if (host.ledger.Published() == host.ledger.Retired()) return false;
    return host.ledger.Reads(address, end);
}

void OpHost::AwaitRetired(std::uint64_t position) {
    state->queue.AwaitRetired(position);
}

bool OpHost::Runnable() const {
    const auto& host = *state;
    const auto head = host.queue.Retired();
    if (host.posted.load(std::memory_order_seq_cst) == head) return false;
    return host.metas[head & host.mask].gates <= host.passed.load(std::memory_order_seq_cst);
}

const Op* OpHost::Front(std::uint64_t& collectEpoch) {
    auto& host = *state;
    const auto head = host.queue.Retired();
    if (host.posted.load(std::memory_order_seq_cst) == head) return nullptr;
    const auto& meta = host.metas[head & host.mask];
    if (meta.gates > host.passed.load(std::memory_order_relaxed)) return nullptr;
    const auto taken = host.queue.TryTake(1);
    if (taken.empty()) return nullptr;
    collectEpoch = meta.collectEpoch;
    return taken.data();
}

void OpHost::ReleaseReads() {
    state->ledger.ReleaseReads();
}

void OpHost::Retire() {
    auto& host = *state;
    host.ledger.Retire();
    host.queue.Retire(1);
}

void OpHost::Check(Gate gate) const {
    const auto& host = *state;
    if (!host.verifies) return;
    if (gate.number != host.passed.load(std::memory_order_relaxed) + 1) throw std::logic_error("op host: an element of the old path was committed out of the order its thread queued it in (APS5_VERIFY_CORE_OPS)");
    if (host.queue.Retired() < gate.ops) throw std::logic_error("op host: an element of the old path was committed before an operation posted ahead of it (APS5_VERIFY_CORE_OPS)");
}

void OpHost::Pass(Gate gate) {
    auto& host = *state;
    const auto seen = host.passed.load(std::memory_order_relaxed);
    if (seen < gate.number) host.passed.store(gate.number, std::memory_order_seq_cst);
}

void OpHost::Sleep(bool sleeping) {
    state->sleeping.store(sleeping, std::memory_order_seq_cst);
}

bool OpHost::Sleeps() const {
    auto& host = *state;
    if (!host.sleeping.load(std::memory_order_seq_cst)) return false;
    host.wakes.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void OpHost::Open() {
    state->passed.store(Every, std::memory_order_seq_cst);
}

void OpHost::Close() {
    auto& host = *state;
    host.closed.store(true, std::memory_order_release);
    host.queue.Close();
}

bool OpHost::Closed() const {
    return state->closed.load(std::memory_order_acquire);
}

OpHost::Counts OpHost::Counted() const {
    const auto& host = *state;
    const auto ledger = host.ledger.Counted();
    Counts counts;
    counts.posted = host.posted.load(std::memory_order_relaxed);
    counts.retired = host.queue.Retired();
    counts.gates = host.gatesQueued.load(std::memory_order_relaxed);
    counts.passed = host.passed.load(std::memory_order_relaxed);
    counts.roomWaits = host.roomWaits.load(std::memory_order_relaxed);
    counts.wakes = host.wakes.load(std::memory_order_relaxed);
    counts.wide = ledger.wide;
    counts.queries = ledger.queries;
    counts.scanned = ledger.scanned;
    return counts;
}

}
