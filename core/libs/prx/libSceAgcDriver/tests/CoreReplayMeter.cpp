#include "prx/libSceAgcDriver/Core/include/ReplayMeter.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

constexpr std::uint64_t PageBytes = 4096;
constexpr std::uint64_t BlockBytes = 65536;

bool overlaps(std::uint64_t a, std::uint64_t aBytes, std::uint64_t b, std::uint64_t bBytes, std::uint64_t grain) {
    return a / grain <= (b + bBytes - 1) / grain && b / grain <= (a + aBytes - 1) / grain;
}

class FakeProof final : public ReplayProof {
public:
    struct Stored {
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint64_t at;
    };

    std::uint64_t generation = 1;
    std::vector<Stored> stores;
    std::vector<ReplayRange> unwatched;
    bool live = false;
    bool tokensBroken = false;
    std::uint64_t pageAsks = 0;
    std::uint64_t blockAsks = 0;

    void Store(std::uint64_t address, std::uint64_t bytes) {
        stores.push_back({address, bytes, ++generation});
    }

    std::uint64_t Generation() override {
        return generation;
    }

    bool PagesUnchanged(std::span<const ReplayQuery> queries) override {
        ++pageAsks;
        for (const auto& query : queries) {
            if (query.since == 0 || !watched(query.address, query.bytes)) return false;
            for (const auto& stored : stores) {
                if (stored.at > query.since && overlaps(stored.address, stored.bytes, query.address, query.bytes, PageBytes)) return false;
            }
        }
        return true;
    }

    bool BlockWritten(const ReplayQuery& query) override {
        ++blockAsks;
        if (!watched(query.address, query.bytes)) return false;
        for (const auto& stored : stores) {
            if (stored.at > query.since && overlaps(stored.address, stored.bytes, query.address, query.bytes, BlockBytes)) return true;
        }
        return false;
    }

    bool TokensLive() override {
        return live;
    }

    ReplayToken TokenFor(const ReplayRange& range) override {
        return {range.address ^ range.bytes, generation};
    }

    bool StillCurrent(const ReplayRange& range, const ReplayToken& token) override {
        if (tokensBroken || token.content != (range.address ^ range.bytes)) return false;
        for (const auto& stored : stores) {
            if (stored.at > token.context && overlaps(stored.address, stored.bytes, range.address, range.bytes, BlockBytes)) return false;
        }
        return true;
    }

private:
    bool watched(std::uint64_t address, std::uint64_t bytes) const {
        for (const auto& range : unwatched) {
            if (overlaps(range.address, range.bytes, address, bytes, 1)) return false;
        }
        return true;
    }
};

constexpr std::uint64_t IndexAddress = 0x10000000;
constexpr std::uint64_t VertexAddress = 0x20000000;
constexpr std::uint64_t DccAddress = 0x30000000;
constexpr std::uint64_t HtileAddress = 0x30100000;

ReplayDraw Draw(std::uint64_t plan = 7, std::uint64_t variant = 0x1000) {
    ReplayDraw draw;
    draw.plan = plan;
    draw.digest = {plan * 11u, plan * 13u};
    draw.variant = variant;
    draw.recipe = variant + 8u;
    draw.inputs = 0xa1;
    draw.targets = 0xb2;
    draw.parameters = 0xc3;
    AddReplayRange(draw, IndexAddress, 3072);
    AddReplayRange(draw, VertexAddress, 8192);
    AddReplayMetadata(draw, DccAddress, 2048);
    AddReplayMetadata(draw, HtileAddress, 16384);
    return draw;
}

std::uint64_t Cause(const ReplayMeter& meter, ReplayCause cause) {
    return meter.Counted().causes[static_cast<std::size_t>(cause)];
}

std::uint64_t Miss(const ReplayMeter& meter, ReplayTokenMiss miss) {
    return meter.Counted().tokenMisses[static_cast<std::size_t>(miss)];
}

ReplayCause Run(ReplayMeter& meter, FakeProof& proof, const ReplayDraw& draw, bool fallback = false) {
    const bool filled = meter.Filled(draw, proof);
    return meter.Commit(draw, fallback, filled, proof);
}

void TestSameKey() {
    ReplayMeter meter;
    FakeProof proof;
    const auto draw = Draw();
    Expect(Run(meter, proof, draw) == ReplayCause::New, "the first draw of a key is new");
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "the same key with untouched inputs is replayable");
    proof.Store(0x50000000, 64);
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "a store elsewhere does not break the proof");
    Expect(meter.Counted().draws == 3, "draws counted");
    Expect(meter.Counted().sameKey == 2, "same key counted");
    Expect(Cause(meter, ReplayCause::Replayed) == 2, "replayed counted");
    Expect(Cause(meter, ReplayCause::New) == 1, "new counted");
}

void TestKeyCauses() {
    ReplayMeter meter;
    FakeProof proof;
    const auto draw = Draw();
    static_cast<void>(Run(meter, proof, draw));
    auto taught = draw;
    taught.variant = 0x2000;
    Expect(Run(meter, proof, taught) == ReplayCause::Recipe, "a variant taught again for the same plan and images is a recipe miss");
    auto recipe = taught;
    recipe.recipe = 0x9999;
    Expect(Run(meter, proof, recipe) == ReplayCause::Recipe, "another recipe is a recipe miss");
    auto input = recipe;
    input.inputs = 0xa2;
    Expect(Run(meter, proof, input) == ReplayCause::Input, "another input range is an input miss");
    auto target = input;
    target.targets = 0xb3;
    Expect(Run(meter, proof, target) == ReplayCause::Target, "another target is a target miss");
    auto parameters = target;
    parameters.parameters = 0xc4;
    Expect(Run(meter, proof, parameters) == ReplayCause::Parameters, "other draw parameters are a parameters miss");
    auto both = parameters;
    both.inputs = 0xa3;
    both.parameters = 0xc5;
    Expect(Run(meter, proof, both) == ReplayCause::Input, "the first difference in order names the miss");
    Expect(Run(meter, proof, both) == ReplayCause::Replayed, "the new key is replayable at its next draw");
    Expect(Run(meter, proof, Draw(8)) == ReplayCause::New, "another plan is new");
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "a key still in its slot is found again after other keys");
}

void TestFallback() {
    ReplayMeter meter;
    FakeProof proof;
    const auto draw = Draw();
    static_cast<void>(Run(meter, proof, draw));
    Expect(Run(meter, proof, draw, true) == ReplayCause::Fallback, "a draw the light back end refused is a fallback");
    Expect(Cause(meter, ReplayCause::Fallback) == 1, "fallback counted");
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "the key is replayable again after the fallback");
}

void TestFill() {
    ReplayMeter meter;
    FakeProof proof;
    const auto draw = Draw();
    Expect(!meter.Filled(draw, proof), "a target never drawn has no fill to report");
    static_cast<void>(Run(meter, proof, draw));
    proof.Store(DccAddress, 2048);
    Expect(meter.Filled(draw, proof), "a store over the keys of a target since its last draw is a fill");
    Expect(Run(meter, proof, draw) == ReplayCause::Fill, "a draw after a keys fill on its target is a fill miss");
    Expect(meter.Counted().afterFill == 1, "draws after a fill counted");
    Expect(!meter.Filled(draw, proof), "the next draw on the target is not after a fill");
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "the draw after the first one on a filled target is replayable");
    proof.Store(HtileAddress + 4096, 512);
    Expect(Run(meter, proof, draw) == ReplayCause::Fill, "a store over the HTILE of the depth target is a fill");
    proof.Store(DccAddress + 2048 + PageBytes, 64);
    Expect(!meter.Filled(draw, proof), "a store in the same 64 KiB block but another page is not a fill");
    auto other = Draw(9, 0x3000);
    static_cast<void>(Run(meter, proof, other));
    proof.Store(DccAddress, 64);
    static_cast<void>(Run(meter, proof, other));
    Expect(!meter.Filled(draw, proof), "the fill was seen by the draw before on the same target");
    Expect(meter.Counted().afterFill == 3, "every first draw after a fill counted");
}

void TestTokenCauses() {
    ReplayMeter meter;
    FakeProof proof;
    const auto draw = Draw();
    static_cast<void>(Run(meter, proof, draw));
    proof.Store(VertexAddress + 100, 16);
    Expect(Run(meter, proof, draw) == ReplayCause::Token, "a store into an input range is a token miss");
    Expect(Miss(meter, ReplayTokenMiss::Written) == 1, "a store into the range is counted as written");
    proof.Store(VertexAddress + 8192 + PageBytes, 16);
    Expect(Run(meter, proof, draw) == ReplayCause::Token, "a store into the 64 KiB block of an input is a token miss");
    Expect(Miss(meter, ReplayTokenMiss::Grain) == 1, "a store outside the range in its block is counted as grain");
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "with no store since, the key is replayable");
    proof.unwatched.push_back({IndexAddress, 16});
    Expect(Run(meter, proof, draw) == ReplayCause::Token, "an input the tracker does not watch is a token miss");
    Expect(Miss(meter, ReplayTokenMiss::Unwatched) == 1, "an unwatched input counted");
    proof.unwatched.clear();
    auto many = Draw(12, 0x4000);
    for (std::uint64_t index = 0; index < ReplayMaxInputs; ++index) AddReplayRange(many, 0x60000000 + index * 0x100000, 256);
    Expect(!many.rangesComplete, "more input ranges than the draw holds are marked incomplete");
    static_cast<void>(Run(meter, proof, many));
    Expect(Run(meter, proof, many) == ReplayCause::Token, "a draw with too many input ranges is a token miss");
    Expect(Miss(meter, ReplayTokenMiss::Ranges) == 1, "too many ranges counted");
    auto empty = Draw(13, 0x5000);
    empty.rangeCount = 0;
    static_cast<void>(Run(meter, proof, empty));
    Expect(Run(meter, proof, empty) == ReplayCause::Replayed, "a draw without input ranges needs no input proof");
    Expect(Cause(meter, ReplayCause::Token) == 4, "token misses counted");
}

void TestFillBeforeToken() {
    ReplayMeter meter;
    FakeProof proof;
    const auto draw = Draw();
    static_cast<void>(Run(meter, proof, draw));
    proof.Store(DccAddress, 64);
    proof.Store(IndexAddress, 64);
    Expect(Run(meter, proof, draw) == ReplayCause::Fill, "a fill is named before a token miss");
    Expect(Miss(meter, ReplayTokenMiss::Written) == 0, "the token breakdown covers token misses only");
}

void TestCollision() {
    ReplayMeter meter(0, 16, 12);
    FakeProof proof;
    const auto a = Draw(20, 0x6000);
    const auto b = Draw(21, 0x7000);
    Expect(Run(meter, proof, a) == ReplayCause::New, "first key in the one-slot table is new");
    Expect(Run(meter, proof, b) == ReplayCause::New, "second key in the one-slot table is new");
    Expect(Run(meter, proof, a) == ReplayCause::Evicted, "a key pushed out of its slot by another is an eviction");
    Expect(Run(meter, proof, a) == ReplayCause::Replayed, "the key is found again once stored back");
    ReplayMeter tiny(0, 0, 0);
    static_cast<void>(Run(tiny, proof, a));
    static_cast<void>(Run(tiny, proof, b));
    Expect(Run(tiny, proof, a) == ReplayCause::Evicted, "an eviction is named even when the variant slot collided too");
    ReplayMeter variantKnows(0, 16, 12);
    const auto c = Draw(22, 0x8000);
    auto d = Draw(23, 0x9000);
    static_cast<void>(Run(variantKnows, proof, a));
    static_cast<void>(Run(variantKnows, proof, b));
    static_cast<void>(Run(variantKnows, proof, c));
    static_cast<void>(Run(variantKnows, proof, d));
    Expect(Run(variantKnows, proof, a) == ReplayCause::Evicted, "the last draw of the variant names an eviction the eviction table forgot");
    Expect(Cause(variantKnows, ReplayCause::Evicted) == 1, "evictions counted");
}

void TestContentToken() {
    ReplayMeter meter;
    FakeProof proof;
    proof.live = true;
    const auto draw = Draw();
    static_cast<void>(Run(meter, proof, draw));
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "a still current token proves the inputs");
    Expect(meter.Counted().tokensTrue == 1, "still current tokens counted");
    proof.Store(VertexAddress, 16);
    Expect(Run(meter, proof, draw) == ReplayCause::Token, "a token no longer current is a token miss");
    Expect(meter.Counted().tokensFalse == 1, "tokens no longer current counted");
    Expect(Miss(meter, ReplayTokenMiss::Written) == 1, "the tracker names why the token failed");
    proof.tokensBroken = true;
    Expect(Run(meter, proof, draw) == ReplayCause::Token, "a false token with clean inputs is still a token miss");
    Expect(Miss(meter, ReplayTokenMiss::TokenOnly) == 1, "a miss only the token sees counted");
    proof.tokensBroken = false;
    Expect(Run(meter, proof, draw) == ReplayCause::Replayed, "a token taken again after the draw holds");
    auto taught = draw;
    taught.variant = 0x1234;
    Expect(Run(meter, proof, taught) == ReplayCause::Recipe, "a key miss does not ask the token");
    Expect(meter.Counted().tokensTrue == 2 && meter.Counted().tokensFalse == 2, "tokens asked on same keys only");
}

void TestRanges() {
    ReplayDraw draw;
    AddReplayRange(draw, 0, 64);
    AddReplayRange(draw, 0x1000, 0);
    Expect(draw.rangeCount == 0, "empty ranges are ignored");
    AddReplayRange(draw, 0x1000, 0x100);
    AddReplayRange(draw, 0x1080, 0x100);
    Expect(draw.rangeCount == 1 && draw.ranges[0].address == 0x1000 && draw.ranges[0].bytes == 0x180, "overlapping ranges merge");
    AddReplayRange(draw, 0x0f00, 0x100);
    Expect(draw.rangeCount == 1 && draw.ranges[0].address == 0x0f00 && draw.ranges[0].bytes == 0x280, "touching ranges merge");
    AddReplayRange(draw, 0x9000, 0x10);
    Expect(draw.rangeCount == 2, "a separate range is added");
    AddReplayMetadata(draw, 0x5000, 0x40);
    AddReplayMetadata(draw, 0x5000, 0x40);
    AddReplayMetadata(draw, 0, 0x40);
    Expect(draw.metadataCount == 1, "metadata ranges are kept once");
    Expect(ReplayMix(1, 2) != ReplayMix(2, 1), "the mix depends on order");
}

void TestNames() {
    Expect(std::string(ReplayCauseName(ReplayCause::Replayed)) == "replayed", "replayed name");
    Expect(std::string(ReplayCauseName(ReplayCause::Input)) == "input range", "input name");
    Expect(std::string(ReplayCauseName(ReplayCause::Count)) == "?", "out of range cause name");
    Expect(std::string(ReplayTokenMissName(ReplayTokenMiss::Grain)) == "grain", "grain name");
    Expect(std::string(ReplayTokenMissName(ReplayTokenMiss::Count)) == "?", "out of range miss name");
}

void TestReport() {
    ReplayMeter meter;
    FakeProof proof;
    const auto draw = Draw();
    static_cast<void>(Run(meter, proof, draw));
    for (int index = 0; index < 3; ++index) static_cast<void>(Run(meter, proof, draw));
    const auto first = meter.TakeReport(10, 8, false, 5.0);
    Expect(Contains(first, "[core-replay] (cumulative) 4 draws metered at the core commit, 3 replayable (75.0 %)"), "report heads with the draws and the replayable share");
    Expect(Contains(first, "front: 10 draws looked at, 8 committed by the core (80.0 %), replayable per draw looked at 30.0 %"), "report gives the front's share");
    Expect(Contains(first, "last window 5.0 s: 3 of 4 metered draws replayable (75.0 %), front 10 draws looked at, 8 committed by the core"), "first window covers everything");
    Expect(Contains(first, "write tracker at the 64 KiB block grain (T1 off)"), "report names the proof");
    Expect(Contains(first, "meter cost 0 ns per metered draw (0 timed)"), "report gives no cost before any timed draw");
    meter.NoteCost(100);
    meter.NoteCost(200);
    Expect(Contains(meter.TakeReport(10, 8, false, 0.1), "meter cost 150 ns per metered draw (2 timed)"), "report averages the timed draws");
    static_cast<void>(Run(meter, proof, draw));
    const auto second = meter.TakeReport(14, 9, true, 5.2);
    Expect(Contains(second, "4 replayable (80.0 %)"), "cumulative share moves");
    Expect(Contains(second, "last window 5.2 s: 1 of 1 metered draws replayable (100.0 %), front 4 draws looked at, 1 committed by the core"), "window restarts at the last report");
    Expect(Contains(second, "proof: T1 content token"), "report names the token when it is live");
}

void TestSampling() {
    ReplayMeter meter(16, 16, 12, 3);
    FakeProof proof;
    std::vector<ReplayDraw> draws;
    for (std::uint64_t index = 0; index < 500; ++index) {
        auto draw = Draw(300 + index, 0x40000 + index * 64);
        draw.rangeCount = 0;
        AddReplayRange(draw, 0x70000000 + index * 0x20000, 1024);
        draws.push_back(draw);
    }
    constexpr int Rounds = 20;
    for (int round = 0; round < Rounds; ++round) {
        for (std::size_t index = 0; index < draws.size(); ++index) {
            if (index % 2 == 1) proof.Store(draws[index].ranges[0].address, 16);
            static_cast<void>(Run(meter, proof, draws[index]));
        }
    }
    const auto same = meter.Counted().sameKey;
    const auto proved = Cause(meter, ReplayCause::Replayed) + Cause(meter, ReplayCause::Token);
    Expect(same + Cause(meter, ReplayCause::Evicted) == (Rounds - 1) * draws.size(), "every later round finds its key unless its slot collided");
    Expect(proved + Cause(meter, ReplayCause::Unproved) == same, "a same-key draw is proved or left unproved");
    Expect(proved * 16 > same && proved * 4 < same, "about one same-key draw in eight is proved");
    Expect(Cause(meter, ReplayCause::Replayed) * 10 > proved * 4 && Cause(meter, ReplayCause::Replayed) * 10 < proved * 6, "the sample sees the written half");
    const auto report = meter.TakeReport(10000, 10000, false, 1.0);
    Expect(Contains(report, "input proof on 1 in 8 same-key draws"), "report names the sample");
    Expect(Contains(report, "not replayable: new 500,"), "report keeps the exact causes");
}

void TestCost() {
    ReplayMeter meter;
    FakeProof proof;
    std::vector<ReplayDraw> draws;
    for (std::uint64_t index = 0; index < 2000; ++index) draws.push_back(Draw(100 + index, 0x10000 + index * 64));
    for (const auto& draw : draws) static_cast<void>(Run(meter, proof, draw));
    const auto started = std::chrono::steady_clock::now();
    constexpr int Rounds = 100;
    for (int round = 0; round < Rounds; ++round) {
        for (const auto& draw : draws) static_cast<void>(Run(meter, proof, draw));
    }
    const auto nanoseconds = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - started).count() / (Rounds * draws.size());
    const auto evicted = Cause(meter, ReplayCause::Evicted);
    std::printf("CoreReplayMeter: %.1f ns per metered draw with the fake proof, %llu of %llu draws of 2000 keys evicted by a slot collision\n", nanoseconds, static_cast<unsigned long long>(evicted), static_cast<unsigned long long>(Rounds * draws.size()));
    Expect(Cause(meter, ReplayCause::Replayed) + evicted == Rounds * draws.size(), "every repeated key is replayable unless its slot collided");
    Expect(evicted * 10u < Rounds * draws.size(), "collisions stay rare at 2000 keys in 65 536 slots");
}

}

int main() {
    TestSameKey();
    TestKeyCauses();
    TestFallback();
    TestFill();
    TestTokenCauses();
    TestFillBeforeToken();
    TestCollision();
    TestContentToken();
    TestRanges();
    TestNames();
    TestReport();
    TestSampling();
    TestCost();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("CoreReplayMeter: ok");
    return 0;
}
