#include "CoreDecoderStreams.hpp"
#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Core/include/DecoderRing.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace AgcDriver;
using namespace AgcDriver::Core;
using namespace DecoderTests;

namespace {

int failures = 0;
std::atomic<int> threadFailures{0};

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

void ExpectInThread(bool condition, const char* what) {
    if (condition) return;
    if (threadFailures.fetch_add(1) < 20) std::fprintf(stderr, "FAIL: %s\n", what);
}

DecodedOp opOf(std::uint64_t host, bool alone = false) {
    DecodedOp op;
    std::memset(static_cast<void*>(&op), 0, sizeof(DecodedOp));
    op.kind = DecodedKind::Draw;
    op.host = host;
    op.flags = alone ? DecodedAlone : 0u;
    return op;
}

struct Emitted {
    std::vector<std::uint64_t> orders;
};

void record(void* context, const DecodedOp& op) {
    static_cast<Emitted*>(context)->orders.push_back(op.order);
}

void testOrderedRetire() {
    DecoderRing ring({16, 4, true});
    Emitted emitted;
    for (std::uint64_t index = 0; index < 10; ++index) Expect(ring.Post(opOf(index * 3)) == index, "operations are numbered in posting order");
    Expect(ring.Posted() == 10 && ring.Retired() == 0, "ten operations are posted, none retired");
    const auto first = ring.TryClaim(4);
    const auto second = ring.TryClaim(4);
    const auto third = ring.TryClaim(4);
    const auto none = ring.TryClaim(4);
    Expect(first.first == 0 && first.count == 4 && second.first == 4 && second.count == 4 && third.first == 8 && third.count == 2 && none.count == 0, "batches are runs of consecutive operations");
    Expect(ring.At(5).host == 15 && ring.At(5).order == 5, "an operation is read back by its number");
    ring.Complete(second, &emitted, &record);
    Expect(emitted.orders.empty() && ring.Retired() == 0, "a batch that is not at the head does not leave");
    ring.Complete(third, &emitted, &record);
    Expect(emitted.orders.empty(), "nor does a later one");
    ring.Complete(first, &emitted, &record);
    Expect(ring.Retired() == 10 && emitted.orders.size() == 10, "completing the head lets every finished batch leave");
    for (std::size_t index = 0; index < emitted.orders.size(); ++index) Expect(emitted.orders[index] == index, "operations leave in posting order");
    const auto counts = ring.Counted();
    Expect(counts.posted == 10 && counts.retired == 10 && counts.claims == 3 && counts.claimed == 10 && counts.retireRuns == 1, "the ring counts its claims and its retire runs");
}

void testAlone() {
    DecoderRing ring({32, 4, true});
    Emitted emitted;
    for (std::uint64_t index = 0; index < 9; ++index) static_cast<void>(ring.Post(opOf(index, index == 3 || index == 4)));
    const auto before = ring.TryClaim(8);
    Expect(before.first == 0 && before.count == 3, "a batch ends before an operation that runs alone");
    Expect(ring.TryClaim(8).count == 0, "an operation that runs alone waits for everything before it to leave");
    ring.Complete(before, &emitted, &record);
    const auto alone = ring.TryClaim(8);
    Expect(alone.first == 3 && alone.count == 1, "it is then claimed by itself");
    Expect(ring.TryClaim(8).count == 0, "nothing starts beside it");
    ring.Complete(alone, &emitted, &record);
    const auto second = ring.TryClaim(8);
    Expect(second.first == 4 && second.count == 1, "a second one follows by itself");
    ring.Complete(second, &emitted, &record);
    const auto rest = ring.TryClaim(8);
    Expect(rest.first == 5 && rest.count == 4, "the operations after it are claimed together again");
    ring.Complete(rest, &emitted, &record);
    Expect(ring.Retired() == 9 && emitted.orders.size() == 9 && ring.Counted().alone == 2, "all nine operations left, two of them alone");
}

void testRoomAndClose() {
    DecoderRing ring({4, 1, true});
    Emitted emitted;
    for (std::uint64_t index = 0; index < 4; ++index) static_cast<void>(ring.Post(opOf(index)));
    std::atomic<std::uint64_t> fifth{DecoderRing::Shut - 1};
    std::thread producer([&] { fifth.store(ring.Post(opOf(4))); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Expect(fifth.load() == DecoderRing::Shut - 1, "a full ring holds the producer");
    const auto batch = ring.Claim(2);
    ring.Complete(batch, &emitted, &record);
    producer.join();
    Expect(fifth.load() == 4 && ring.Counted().roomWaits == 1, "the producer posts once an operation has left");
    std::thread joiner([&] { ring.Join(); });
    auto rest = ring.Claim(8);
    Expect(rest.first == 2 && rest.count == 3, "the remaining operations are claimed");
    ring.Complete(rest, &emitted, &record);
    joiner.join();
    Expect(ring.Retired() == 5, "a join returns once everything posted has left");
    std::atomic<std::uint32_t> empty{7};
    std::thread sleeper([&] { empty.store(ring.Claim(4).count); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Expect(empty.load() == 7, "a consumer waits on an empty ring");
    ring.Close();
    sleeper.join();
    Expect(empty.load() == 0 && ring.Closed(), "closing the ring releases the consumer with nothing");
    Expect(ring.Post(opOf(9)) == DecoderRing::Shut, "a closed ring refuses a post");
}

void testVerifies() {
    DecoderRing ring({16, 4, true});
    for (std::uint64_t index = 0; index < 6; ++index) static_cast<void>(ring.Post(opOf(index)));
    const auto batch = ring.TryClaim(2);
    bool threw = false;
    try {
        ring.Complete({4, 2}, nullptr, nullptr);
    } catch (const std::logic_error&) {
        threw = true;
    }
    Expect(threw, "completing operations that were not claimed is refused");
    ring.Complete(batch, nullptr, nullptr);
    Expect(ring.Retired() == 2, "a batch completes without an emitter");
    const auto next = ring.TryClaim(2);
    ring.Complete(next, nullptr, nullptr);
    threw = false;
    try {
        ring.Complete(next, nullptr, nullptr);
    } catch (const std::logic_error&) {
        threw = true;
    }
    Expect(threw, "completing a batch again after it left is refused");
}

struct Shared {
    DecoderRing* ring = nullptr;
    std::atomic<std::uint64_t> rested{0};
    std::atomic<std::uint64_t> flying{0};
    std::atomic<std::uint64_t> aloneFlying{0};
    std::atomic<std::uint64_t> next{0};
    std::atomic<std::uint64_t> emitters{0};
    std::atomic<std::uint64_t> processed{0};
};

void emitChecked(void* context, const DecodedOp& op) {
    auto& shared = *static_cast<Shared*>(context);
    ExpectInThread(shared.emitters.fetch_add(1) == 0, "two threads emit at once");
    ExpectInThread(shared.next.load() == op.order, "an operation leaves out of order");
    ExpectInThread(op.host == op.order * 7 + 1, "an operation left with another's content");
    shared.next.store(op.order + 1);
    shared.emitters.fetch_sub(1);
}

void consume(Shared& shared, std::uint64_t seed) {
    Random random{seed};
    auto& ring = *shared.ring;
    for (;;) {
        const auto batch = ring.Claim(1 + random.Below(16));
        if (batch.count == 0) return;
        shared.flying.fetch_add(batch.count);
        for (std::uint32_t index = 0; index < batch.count; ++index) {
            const auto& op = ring.At(batch.first + index);
            ExpectInThread(op.order == batch.first + index, "a claimed operation does not carry its number");
            if ((op.flags & DecodedAlone) != 0) {
                ExpectInThread(batch.count == 1, "an operation that runs alone is claimed in a batch");
                shared.aloneFlying.fetch_add(1);
                ExpectInThread(shared.flying.load() == 1, "an operation that runs alone runs beside another");
                ExpectInThread(ring.Retired() == op.order, "an operation that runs alone starts before the ones before it left");
                if (random.OneIn(4)) std::this_thread::yield();
                ExpectInThread(shared.flying.load() == 1, "an operation started beside one that runs alone");
                shared.aloneFlying.fetch_sub(1);
            } else {
                ExpectInThread(shared.aloneFlying.load() == 0, "an operation runs beside one that runs alone");
            }
            if (random.OneIn(64)) std::this_thread::yield();
        }
        shared.processed.fetch_add(batch.count);
        shared.flying.fetch_sub(batch.count);
        ring.Complete(batch, &shared, &emitChecked);
        if (random.OneIn(200)) ring.Rest(&shared, [](void* context) {
            auto& rest = *static_cast<Shared*>(context);
            ExpectInThread(rest.flying.load() == 0, "an operation is in flight while the ring rests");
            ExpectInThread(rest.emitters.load() == 0, "operations leave while the ring rests");
            std::this_thread::yield();
            ExpectInThread(rest.flying.load() == 0, "an operation started while the ring rests");
            rest.rested.fetch_add(1);
        });
    }
}

void testConcurrent() {
    constexpr std::uint64_t Operations = 200000;
    DecoderRing ring({256, 8, true});
    Shared shared;
    shared.ring = &ring;
    std::vector<std::thread> consumers;
    for (std::uint64_t index = 0; index < 4; ++index) consumers.emplace_back([&shared, index] { consume(shared, 1000 + index); });
    Random random{42};
    for (std::uint64_t index = 0; index < Operations; ++index) {
        const auto order = ring.Post(opOf(index * 7 + 1, random.OneIn(97)));
        if (order != index) {
            Expect(false, "the producer's operations are numbered in order");
            break;
        }
        if (random.OneIn(1000)) ring.Flush();
        if (random.OneIn(5000)) ring.Join();
    }
    ring.Join();
    Expect(ring.Retired() == Operations && shared.next.load() == Operations && shared.processed.load() == Operations, "every operation was processed once and left in order");
    ring.Close();
    for (auto& consumer : consumers) consumer.join();
    const auto counts = ring.Counted();
    Expect(counts.alone != 0 && counts.claims < Operations, "operations ran alone and others in batches");
    Expect(shared.rested.load() != 0 && counts.rests == shared.rested.load(), "the ring rested with nothing in flight");
}

struct Pipeline {
    DecoderRing* ring = nullptr;
    std::atomic<std::uint64_t> next{0};
    std::atomic<std::uint64_t> draws{0};
};

std::uint64_t postDraw(void* context, const DecodedOp& op) {
    auto& pipeline = *static_cast<Pipeline*>(context);
    auto posted = op;
    posted.host = HashState(*op.state.shader, *op.state.context, DecodedUserConfig(op.state)) ^ op.state.indexBase ^ (static_cast<std::uint64_t>(op.state.instanceCount) << 40u);
    return pipeline.ring->Post(posted);
}

DecoderHeavy postHeavy(void*, std::size_t cursor, std::span<const std::uint32_t> packet, QueueState& live) {
    return {cursor + packet.size(), ReferenceHeavy(live, packet), false};
}

std::uint64_t ringRetired(void* context) {
    return static_cast<Pipeline*>(context)->ring->Retired();
}

void ringAwait(void* context, std::uint64_t count) {
    static_cast<Pipeline*>(context)->ring->AwaitRetired(count);
}

void emitDraw(void* context, const DecodedOp& op) {
    auto& pipeline = *static_cast<Pipeline*>(context);
    ExpectInThread(pipeline.next.load() == op.order, "a draw leaves out of order");
    pipeline.next.store(op.order + 1);
}

void resolve(Pipeline& pipeline, std::uint64_t seed) {
    Random random{seed};
    DecodedView view;
    auto& ring = *pipeline.ring;
    for (;;) {
        const auto batch = ring.Claim(1 + random.Below(12));
        if (batch.count == 0) return;
        for (std::uint32_t index = 0; index < batch.count; ++index) {
            const auto& op = ring.At(batch.first + index);
            ApplyDecoded(op.state, view);
            const auto hash = HashState(view.queue.shader, view.queue.context, view.queue.userConfig) ^ view.queue.indexBase ^ (static_cast<std::uint64_t>(view.queue.instanceCount) << 40u);
            ExpectInThread(hash == op.host, "the state a resolver reads differs from the state the decoder posted");
            if (random.OneIn(32)) std::this_thread::yield();
        }
        pipeline.draws.fetch_add(batch.count);
        ring.Complete(batch, &pipeline, &emitDraw);
    }
}

void testDecoderThroughRing() {
    DecoderRing ring({64, 8, true});
    Pipeline pipeline;
    pipeline.ring = &ring;
    std::vector<std::thread> resolvers;
    for (std::uint64_t index = 0; index < 3; ++index) resolvers.emplace_back([&pipeline, index] { resolve(pipeline, 77 + index); });
    Decoder decoder({8}, {&pipeline, &postDraw, &postHeavy, &ringRetired, &ringAwait});
    QueueState live;
    std::uint64_t draws = 0;
    for (std::uint64_t seed = 300; seed < 312; ++seed) {
        const auto stream = RandomStream(seed, 6000);
        Decoder::Input input;
        input.commands = stream.words;
        input.limit = stream.words.size();
        input.lists = &stream.lists;
        const auto result = decoder.Decode(live, input);
        Expect(!result.stopped && result.cursor == stream.words.size(), "a stream decodes through the ring");
        draws += stream.draws;
        ring.Join();
    }
    Expect(ring.Retired() == draws && pipeline.draws.load() == draws && pipeline.next.load() == draws, "every draw was resolved once and left in order");
    Expect(decoder.Counted().snapshotWaits != 0, "the decoder waited for resolvers before reusing a snapshot");
    ring.Close();
    for (auto& resolver : resolvers) resolver.join();
}

}

int main() {
    testOrderedRetire();
    testAlone();
    testRoomAndClose();
    testVerifies();
    testConcurrent();
    testDecoderThroughRing();
    failures += threadFailures.load();
    if (failures != 0) {
        std::fprintf(stderr, "%d decoder ring checks failed\n", failures);
        return 1;
    }
    std::printf("decoder ring tests passed\n");
    return 0;
}
