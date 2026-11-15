#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::DriverDetail::DrawPipeline;
using Kind = DrawPipeline::CommitKind;
using Site = DrawFinisher::DrainSite;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Gate {
    std::promise<void> opener;
    std::shared_future<void> opened = opener.get_future().share();
    bool open = false;
    void Open() {
        if (!std::exchange(open, true)) opener.set_value();
    }
    ~Gate() { Open(); }
};

struct OpenOnExit {
    Gate& gate;
    ~OpenOnExit() { gate.Open(); }
};

struct Commits {
    std::mutex mutex;
    std::vector<int> order;
    void Note(int index) {
        std::lock_guard lock(mutex);
        order.push_back(index);
    }
    std::vector<int> Taken() {
        std::lock_guard lock(mutex);
        return order;
    }
};

void Hold(Gate& gate) {
    DrawFinisher::Pending nothing;
    nothing.unknown = false;
    DrawFinisher::Get().Enqueue([opened = gate.opened] { opened.wait(); }, std::move(nothing));
}

void Settle() {
    DrawFinisher::Get().Drain();
    DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
}

void ItemsCommitInTheOrderQueued(bool relays) {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    constexpr int Submissions = 3;
    const auto drained = DrawFinisher::DrainCount(Site::Item, true);
    const auto relayed = DrawFinisher::DrainCount(Site::Item, false);
    Gate gate;
    Commits commits;
    int queued = 0;
    auto producer = std::async(std::launch::async, [&] {
        Hold(gate);
        const auto draw = [&] {
            finisher.Enqueue([&commits, &pipeline, index = queued++] {
                pipeline.Enqueue(Kind::Draw, [&commits, index] { commits.Note(index); }, {}, 0, {}, {}, true);
            });
        };
        const auto item = [&](Kind kind) {
            pipeline.Enqueue(kind, [&commits, index = queued++] { commits.Note(index); }, {});
        };
        for (int submission = 0; submission < Submissions; ++submission) {
            draw();
            item(Kind::DmaWrite);
            draw();
            item(Kind::Completion);
        }
    });
    const OpenOnExit opening{gate};
    if (relays) {
        Require(producer.wait_for(std::chrono::seconds(20)) == std::future_status::ready, "queueing an ordered item behind a busy draw finisher waited for the finisher");
        Require(commits.Taken().empty(), "an item queued behind a held draw finisher job was committed before that job ran");
    } else {
        Require(producer.wait_for(std::chrono::milliseconds(300)) == std::future_status::timeout, "an ordered item was queued behind a busy draw finisher without the relay");
    }
    gate.Open();
    producer.get();
    Settle();
    const auto order = commits.Taken();
    Require(order.size() == static_cast<std::size_t>(queued), "the pipeline committed " + std::to_string(order.size()) + " of " + std::to_string(queued) + " items");
    for (std::size_t index = 0; index < order.size(); ++index) {
        Require(order[index] == static_cast<int>(index), "commit " + std::to_string(index) + " ran item " + std::to_string(order[index]));
    }
    if (relays) {
        Require(DrawFinisher::DrainCount(Site::Item, true) == drained, "a relayed item was counted as a drain of the draw finisher");
        Require(DrawFinisher::DrainCount(Site::Item, false) == relayed + 2u * Submissions, "the relayed items were not counted");
    } else {
        Require(DrawFinisher::DrainCount(Site::Item, true) > drained, "the drain of the draw finisher before an ordered item was not counted");
        Require(DrawFinisher::DrainCount(Site::Item, false) == relayed, "an item was counted as relayed without the relay");
    }
}

void ItemsThatReadWaitForTheFinisher() {
    auto& pipeline = DrawPipeline::Queue0();
    alignas(8) static std::array<std::uint32_t, 4> source{};
    alignas(8) static std::array<std::uint32_t, 4> destination{};
    const auto from = reinterpret_cast<std::uintptr_t>(source.data());
    const auto to = reinterpret_cast<std::uintptr_t>(destination.data());
    Gate gate;
    Commits commits;
    auto producer = std::async(std::launch::async, [&] {
        Hold(gate);
        pipeline.Enqueue(Kind::DmaCopy, [&commits] { commits.Note(0); }, {{to, to + sizeof(destination)}}, 0, {}, {{from, from + sizeof(source)}});
    });
    const OpenOnExit opening{gate};
    Require(producer.wait_for(std::chrono::milliseconds(300)) == std::future_status::timeout, "an item that reads guest memory was queued behind a busy draw finisher, where its reads hold no flush back");
    gate.Open();
    producer.get();
    Settle();
    Require(commits.Taken().size() == 1, "the item that reads guest memory was not committed");
}

void RelayedWritesStayVisible() {
    auto& pipeline = DrawPipeline::Queue0();
    alignas(8) static std::array<std::uint32_t, 4> stored{};
    alignas(8) static std::array<std::uint32_t, 4> elsewhere{};
    const auto address = reinterpret_cast<std::uintptr_t>(stored.data());
    const std::uint32_t value = 0x1234abcdu;
    std::vector<std::byte> bytes(sizeof(value));
    std::memcpy(bytes.data(), &value, sizeof(value));
    Gate gate;
    Hold(gate);
    const OpenOnExit opening{gate};
    pipeline.EnqueueStore([] {}, address, bytes);
    Require(pipeline.Busy(), "the pipeline reports no work while the draw finisher holds a relayed item");
    Require(pipeline.Overlaps(address, sizeof(value)), "a relayed item's write range is not reported while the draw finisher holds it");
    Require(!pipeline.Overlaps(reinterpret_cast<std::uintptr_t>(elsewhere.data()), sizeof(elsewhere)), "a relayed item is reported over a range it does not write");
    const auto label = pipeline.PendingLabel(address, sizeof(value));
    Require(label.has_value() && *label == value, "the bytes a relayed store carries are not found while the draw finisher holds it");
    Require(pipeline.StoredBytes(address, sizeof(value), {}) != DrawPipeline::QueuedWrite::None, "a relayed store is not reported as a queued write");
    gate.Open();
    Settle();
    Require(!pipeline.Overlaps(address, sizeof(value)), "a committed relayed item still reports its write range");
}

void FailedJobSkipsTheItemsBehindIt() {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    Gate gate;
    Commits commits;
    Hold(gate);
    const OpenOnExit opening{gate};
    finisher.Enqueue([] { throw std::runtime_error("finisher job failed"); });
    pipeline.Enqueue(Kind::Seal, [&commits] { commits.Note(0); }, {});
    gate.Open();
    bool raised = false;
    try {
        finisher.Drain();
    } catch (const std::runtime_error& error) {
        raised = std::string(error.what()) == "finisher job failed";
    }
    Require(raised, "a failed draw finisher job did not reach the thread that drains the finisher");
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(commits.Taken().empty(), "an item relayed behind a failed draw finisher job was committed");
    finisher.Enqueue([&commits] { commits.Note(1); });
    Settle();
    Require(commits.Taken() == std::vector<int>{1}, "the draw finisher did not run a job queued after its failure was taken");
}

void CommitOrderRefusesAnItemBehindALaterOne() {
    DrawPipeline::CommitOrder order;
    Require(order.Accept({1, 1}) && order.Accept({1, 2}) && order.Accept({1, 2}) && order.Accept({2, 1}) && order.Accept({1, 5}), "the commit order check refused items in the order their threads queued them");
    Require(!order.Accept({1, 4}), "the commit order check accepted an item behind one its thread queued later");
    Require(order.Accept({2, 1}) && order.Accept({1, 5}) && !order.Accept({2, 0}), "the commit order check does not keep one sequence per queuing thread");
}

}

int main() {
    try {
        if (DrawPipeline::Depth() == 0) return 77;
        const bool relays = DrawFinisher::Relays();
        CommitOrderRefusesAnItemBehindALaterOne();
        ItemsCommitInTheOrderQueued(relays);
        ItemsThatReadWaitForTheFinisher();
        if (relays) {
            RelayedWritesStayVisible();
            FailedJobSkipsTheItemsBehindIt();
        }
        std::cout << "ordered items behind the draw finisher " << (relays ? "relayed" : "waited") << " in order\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
