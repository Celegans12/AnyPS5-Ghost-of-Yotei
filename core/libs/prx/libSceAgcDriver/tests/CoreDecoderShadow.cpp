#include "CoreDecoderStreams.hpp"
#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Core/include/DecoderShadow.hpp"
#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>

using namespace AgcDriver;
using namespace AgcDriver::Core;
using namespace DecoderTests;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

struct Tamper {
    std::size_t draw = std::numeric_limits<std::size_t>::max();
    int kind = 0;
};

std::string follow(const Stream& stream, QueueState& queue, DecoderShadow& shadow, bool comparesRun, Tamper tamper, std::size_t& draws) {
    shadow.Begin(queue);
    bool settled = false;
    bool tampered = false;
    for (std::size_t cursor = 0; cursor < stream.words.size();) {
        const bool drawPacket = LaneOf(stream.words[cursor]) == Lane::Draw;
        if (drawPacket && settled && !tampered && draws >= tamper.draw) {
            tampered = true;
            if (tamper.kind == 0) queue.context.insert_or_assign(0x31a, 0xdeadu);
            else if (tamper.kind == 1) queue.instanceCount += 1;
            else if (tamper.kind == 2) queue.userConfig.erase(0x24a);
            else queue.runMask.bits ^= RunMask::Parameters;
        }
        try {
            shadow.Step(stream.words, cursor, stream.lists, queue, comparesRun);
        } catch (const std::runtime_error& error) {
            return error.what();
        }
        settled = LaneOf(stream.words[cursor]) != Lane::Heavy && LaneOf(stream.words[cursor]) != Lane::Skip;
        bool draw = false;
        const auto count = ReferenceStep(stream, cursor, queue, draw);
        if (draw) {
            ++draws;
            if (comparesRun) queue.runMask.Drawn();
        }
        cursor += count;
    }
    return {};
}

void testFollows() {
    for (const bool comparesRun : {true, false}) {
        DecoderShadow shadow;
        QueueState queue;
        std::size_t draws = 0;
        for (std::uint64_t seed = 500; seed < 508; ++seed) {
            const auto stream = RandomStream(seed, 3000);
            const auto difference = follow(stream, queue, shadow, comparesRun, {}, draws);
            if (!difference.empty()) std::fprintf(stderr, "%s\n", difference.c_str());
            Expect(difference.empty(), "the shadow decoder agrees with the packet path on a random stream");
        }
        const auto counts = shadow.Counted();
        Expect(counts.draws == draws && draws != 0 && counts.differing == 0, "every draw was verified");
        Expect(counts.light != 0 && counts.heavy != 0 && counts.adopted >= 8, "packets were decoded here and others left to the packet path");
        Expect(shadow.Report().find(" 0 differing") != std::string::npos, "the report names the differing count");
    }
}

void testDigests() {
    QueueState queue;
    queue.staleDigests = 0;
    queue.shaderDigest = {};
    queue.contextDigest = {};
    queue.userConfigDigest = {};
    queue.shader.clear();
    queue.context.clear();
    queue.userConfig.clear();
    queue.userConfig.insert_or_assign(0x24a, 0);
    std::size_t draws = 0;
    DecoderShadow shadow;
    const auto stream = RandomStream(600, 1500);
    Expect(follow(stream, queue, shadow, true, {}, draws).empty(), "digests kept by both paths agree");
    QueueState late;
    DecoderShadow adopting;
    Stream small;
    const std::uint32_t values[] = {7u, 8u};
    small.Set(0x69, 0x318, values);
    small.DrawAuto(3);
    small.Set(0x69, 0x320, values);
    small.DrawAuto(3);
    adopting.Begin(late);
    adopting.Step(small.words, 0, small.lists, late, true);
    bool draw = false;
    std::size_t cursor = ReferenceStep(small, 0, late, draw);
    Expect(adopting.Counted().digestsAdopted == 0, "nothing is adopted while both sides hold stale digests");
    late.staleDigests = 0;
    late.shaderDigest = {1, 2};
    late.contextDigest = {3, 4};
    late.userConfigDigest = {5, 6};
    std::string difference;
    try {
        while (cursor < small.words.size()) {
            adopting.Step(small.words, cursor, small.lists, late, true);
            cursor += ReferenceStep(small, cursor, late, draw);
            if (draw) late.runMask.Drawn();
        }
    } catch (const std::runtime_error& error) {
        difference = error.what();
    }
    Expect(difference.empty() && adopting.Counted().digestsAdopted == 1 && adopting.Counted().draws == 2, "digests the packet path refreshed are adopted and then kept in step");
}

void testCatches() {
    const char* expected[] = {"context register 0x31a", "instance count", "user configuration register 0x24a", "run mask"};
    for (int kind = 0; kind < 4; ++kind) {
        DecoderShadow shadow;
        QueueState queue;
        std::size_t draws = 0;
        const auto stream = RandomStream(700 + static_cast<std::uint64_t>(kind), 2000);
        const auto difference = follow(stream, queue, shadow, true, {40, kind}, draws);
        Expect(difference.find("APS5_VERIFY_DECODER") != std::string::npos && difference.find(expected[kind]) != std::string::npos, "a state the packet path changed behind the decoder is caught and named");
        if (difference.find(expected[kind]) == std::string::npos) std::fprintf(stderr, "  got: %s\n", difference.c_str());
        Expect(draws >= 40 && draws < 60 && shadow.Counted().differing == 1 && shadow.Counted().draws == draws, "the difference is caught at the draw it first shows at");
    }
}

}

int main() {
    testFollows();
    testDigests();
    testCatches();
    if (failures != 0) {
        std::fprintf(stderr, "%d decoder shadow checks failed\n", failures);
        return 1;
    }
    std::printf("decoder shadow tests passed\n");
    return 0;
}
