#include "prx/libSceAgcDriver/Core/include/DecoderShadow.hpp"
#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace AgcDriver::Core {

namespace {

std::string bankDifference(const char* name, const Registers& decoded, const Registers& queue) {
    std::string found;
    char text[160];
    decoded.ForEach([&](std::uint32_t offset, std::uint32_t value) {
        if (!found.empty()) return;
        if (!queue.contains(offset)) {
            std::snprintf(text, sizeof(text), "%s register 0x%x: the decoder holds 0x%x, the packet path has no value", name, offset, value);
            found = text;
        } else if (queue.at(offset) != value) {
            std::snprintf(text, sizeof(text), "%s register 0x%x: the decoder holds 0x%x, the packet path 0x%x", name, offset, value, queue.at(offset));
            found = text;
        }
    });
    if (!found.empty()) return found;
    queue.ForEach([&](std::uint32_t offset, std::uint32_t value) {
        if (!found.empty() || decoded.contains(offset)) return;
        std::snprintf(text, sizeof(text), "%s register 0x%x: the decoder has no value, the packet path holds 0x%x", name, offset, value);
        found = text;
    });
    return found;
}

}

std::string DescribeDecodedDifference(const DecodedState& decoded, const QueueState& queue) {
    if (auto found = bankDifference("shader", *decoded.shader, queue.shader); !found.empty()) return found;
    if (auto found = bankDifference("context", *decoded.context, queue.context); !found.empty()) return found;
    if (auto found = bankDifference("user configuration", *decoded.userConfig, queue.userConfig); !found.empty()) return found;
    char text[200];
    if (decoded.indexBase != queue.indexBase) {
        std::snprintf(text, sizeof(text), "index base: the decoder holds 0x%llx, the packet path 0x%llx", static_cast<unsigned long long>(decoded.indexBase), static_cast<unsigned long long>(queue.indexBase));
        return text;
    }
    if (decoded.indexType != queue.indexType) {
        std::snprintf(text, sizeof(text), "index type: the decoder holds %u, the packet path %u", decoded.indexType, queue.indexType);
        return text;
    }
    if (decoded.instanceCount != queue.instanceCount) {
        std::snprintf(text, sizeof(text), "instance count: the decoder holds %u, the packet path %u", decoded.instanceCount, queue.instanceCount);
        return text;
    }
    if ((decoded.hasIndexBufferSize != 0) != queue.indexBufferSize.has_value() || decoded.indexBufferSize != queue.indexBufferSize.value_or(0)) {
        std::snprintf(text, sizeof(text), "index buffer size: the decoder holds %u (set %u), the packet path %u (set %u)", decoded.indexBufferSize, static_cast<unsigned>(decoded.hasIndexBufferSize), queue.indexBufferSize.value_or(0), queue.indexBufferSize.has_value() ? 1u : 0u);
        return text;
    }
    if (decoded.drawIndirectBase != queue.drawIndirectBase || decoded.dispatchIndirectBase != queue.dispatchIndirectBase) return "indirect bases differ";
    if (decoded.predicationAddress != queue.predication.address || decoded.predicationOperation != queue.predication.operation || (decoded.predicationExecuteWhenSet != 0) != queue.predication.executeWhenSet) return "predication differs";
    if (decoded.staleDigests != queue.staleDigests) {
        std::snprintf(text, sizeof(text), "stale digests: the decoder holds %u, the packet path %u", static_cast<unsigned>(decoded.staleDigests), static_cast<unsigned>(queue.staleDigests));
        return text;
    }
    if ((decoded.staleDigests & 1u) == 0 && !(decoded.shaderDigest == queue.shaderDigest)) return "the shader bank digest differs while the registers agree";
    if ((decoded.staleDigests & 2u) == 0 && !(decoded.contextDigest == queue.contextDigest)) return "the context bank digest differs while the registers agree";
    if ((decoded.staleDigests & 4u) == 0 && !(decoded.userConfigDigest == queue.userConfigDigest)) return "the user configuration bank digest differs while the registers agree";
    return {};
}

struct DecoderShadow::State {
    State() : decoder({8}, {this, &State::draw, &State::heavy, nullptr, nullptr}) {}

    Decoder decoder;
    QueueState live;
    const QueueState* expected = nullptr;
    std::size_t cursor = 0;
    std::uint64_t order = 0;
    bool comparesRun = false;
    bool adopt = true;
    Counts counts;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();

    static std::uint64_t draw(void* context, const DecodedOp& op) {
        auto& shadow = *static_cast<State*>(context);
        const auto& queue = *shadow.expected;
        std::string difference;
        if (!SameDecoded(op.state, queue)) difference = DescribeDecodedDifference(op.state, queue);
        else if (shadow.comparesRun && op.state.runBits != queue.runMask.bits) difference = "run mask: the decoder holds " + std::to_string(op.state.runBits) + ", the packet path " + std::to_string(queue.runMask.bits);
        if (!difference.empty()) {
            ++shadow.counts.differing;
            char where[96];
            std::snprintf(where, sizeof(where), " (draw at word %zu, opcode 0x%x, %llu draws verified before it)", shadow.cursor, op.opcode, static_cast<unsigned long long>(shadow.counts.draws));
            throw std::runtime_error("APS5_VERIFY_DECODER: the state the decoder would post for a draw differs from the state of the packet path: " + difference + where);
        }
        ++shadow.counts.draws;
        return shadow.order++;
    }

    static DecoderHeavy heavy(void* context, std::size_t cursor, std::span<const std::uint32_t> packet, QueueState&) {
        auto& shadow = *static_cast<State*>(context);
        ++shadow.counts.heavy;
        shadow.adopt = true;
        return {cursor + packet.size(), 0, false};
    }
};

bool DecoderShadow::Enabled() {
    static const bool enabled = std::getenv("APS5_VERIFY_DECODER") != nullptr;
    return enabled;
}

DecoderShadow::DecoderShadow() : state(std::make_unique<State>()) {}

DecoderShadow::~DecoderShadow() = default;

void DecoderShadow::Begin(const QueueState&) {
    state->adopt = true;
}

void DecoderShadow::Step(std::span<const std::uint32_t> commands, std::size_t cursor, const Decoder::Lists& lists, const QueueState& queue, bool comparesRun) {
    auto& shadow = *state;
    if (cursor >= commands.size() || LaneOf(commands[cursor]) == Lane::Skip) return;
    if (shadow.adopt) {
        shadow.live = queue;
        shadow.decoder.Touch(DecoderEvery);
        shadow.adopt = false;
        ++shadow.counts.adopted;
    }
    if (shadow.live.staleDigests != queue.staleDigests && queue.staleDigests == 0) {
        shadow.live.shaderDigest = queue.shaderDigest;
        shadow.live.contextDigest = queue.contextDigest;
        shadow.live.userConfigDigest = queue.userConfigDigest;
        shadow.live.staleDigests = 0;
        ++shadow.counts.digestsAdopted;
    }
    if (!comparesRun) shadow.live.runMask = queue.runMask;
    shadow.expected = &queue;
    shadow.cursor = cursor;
    shadow.comparesRun = comparesRun;
    Decoder::Input input;
    input.commands = commands;
    input.cursor = cursor;
    input.limit = cursor + LaneWords(commands[cursor]);
    input.lists = &lists;
    const auto heavyBefore = shadow.counts.heavy;
    const auto result = shadow.decoder.Decode(shadow.live, input);
    if (result.stopped || result.cursor != input.limit) throw std::runtime_error("APS5_VERIFY_DECODER: the decoder did not consume the packet at word " + std::to_string(cursor));
    ++shadow.counts.packets;
    if (shadow.counts.heavy == heavyBefore) ++shadow.counts.light;
}

DecoderShadow::Counts DecoderShadow::Counted() const {
    return state->counts;
}

bool DecoderShadow::ReportDue() {
    const auto now = std::chrono::steady_clock::now();
    if (now - state->reported < std::chrono::seconds(5)) return false;
    state->reported = now;
    return true;
}

std::string DecoderShadow::Report() const {
    const auto& counts = state->counts;
    const auto decoded = state->decoder.Counted();
    char text[400];
    std::snprintf(text, sizeof(text), "[decoder] verify (cumulative): %llu packets followed, %llu decoded here, %llu left to the packet path; %llu draws verified, %llu differing; state adopted %llu times, digests %llu times; snapshots: shader %llu, context %llu, user configuration %llu; %llu register writes, %llu without a change",
        static_cast<unsigned long long>(counts.packets), static_cast<unsigned long long>(counts.light), static_cast<unsigned long long>(counts.heavy), static_cast<unsigned long long>(counts.draws), static_cast<unsigned long long>(counts.differing), static_cast<unsigned long long>(counts.adopted), static_cast<unsigned long long>(counts.digestsAdopted),
        static_cast<unsigned long long>(decoded.shaderSnapshots), static_cast<unsigned long long>(decoded.contextSnapshots), static_cast<unsigned long long>(decoded.userConfigSnapshots), static_cast<unsigned long long>(decoded.registers), static_cast<unsigned long long>(decoded.unchanged));
    return text;
}

}
