#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace AgcDriver::Core {

namespace {

constexpr std::uint32_t OffsetFlags = 0x70000000u;
constexpr std::uint32_t LastOffset = 0xffffu;
constexpr std::uint32_t TagMarkerOffset = 0x342u;
constexpr std::uint32_t IndexTypeOffset = 0x243u;
constexpr std::uint64_t NoOrder = std::numeric_limits<std::uint64_t>::max();

enum class Opcode : std::uint32_t {
    IndexBufferSize = 0x13,
    IndexBase = 0x26,
    NumInstances = 0x2f,
    SetShRegIndirect = 0x63,
    SetUconfigRegIndirect = 0x64,
    SetContextReg = 0x69,
    SetShReg = 0x76,
    SetUconfigReg = 0x79,
    SetContextRegIndirect = 0x9f,
};

Bank bankOf(std::uint32_t opcode) {
    switch (static_cast<Opcode>(opcode)) {
        case Opcode::SetContextReg:
        case Opcode::SetContextRegIndirect:
            return Bank::Context;
        case Opcode::SetShReg:
        case Opcode::SetShRegIndirect:
            return Bank::Shader;
        default:
            return Bank::UserConfig;
    }
}

bool validOffset(std::uint32_t raw) {
    return raw != 0xffffffffu && (raw & ~OffsetFlags) <= LastOffset;
}

struct Slot {
    Registers registers;
    std::uint64_t last = 0;
};

struct Ring {
    explicit Ring(std::size_t count) : slots(std::max<std::size_t>(count, 2)), current(slots.size() - 1) {}

    std::vector<Slot> slots;
    std::size_t current;
    std::uint64_t version = 0;
    bool dirty = true;
};

}

struct Decoder::State {
    State(Settings settings, DecoderHost host) : host(host), shader(settings.snapshots), context(settings.snapshots), userConfig(settings.snapshots) {}

    DecoderHost host;
    Ring shader;
    Ring context;
    Ring userConfig;
    const QueueState* live = nullptr;
    std::uint64_t retired = 0;
    DecoderCounts counts;

    void touch(std::uint8_t banks) {
        if ((banks & DecoderShader) != 0) shader.dirty = true;
        if ((banks & DecoderContext) != 0) context.dirty = true;
        if ((banks & DecoderUserConfig) != 0) userConfig.dirty = true;
    }

    void adopt(const QueueState& queue) {
        if (live == &queue) return;
        live = &queue;
        touch(DecoderEvery);
    }

    bool held(std::uint64_t last) {
        if (host.retired == nullptr || last <= retired) return false;
        retired = host.retired(host.context);
        return last > retired;
    }

    void take(Ring& ring, const Registers& registers, std::uint64_t& snapshots) {
        if (!ring.dirty) return;
        const auto index = (ring.current + 1) % ring.slots.size();
        auto& slot = ring.slots[index];
        if (held(slot.last)) {
            ++counts.snapshotWaits;
            if (host.awaitRetired == nullptr) throw std::logic_error("decoder: a register snapshot is still read and the host cannot wait for it");
            host.awaitRetired(host.context, slot.last);
            if (held(slot.last)) throw std::logic_error("decoder: the host returned before the operations reading a register snapshot retired");
        }
        slot.registers = registers;
        slot.last = 0;
        ring.current = index;
        ring.dirty = false;
        ++ring.version;
        ++snapshots;
    }

    bool write(QueueState& queue, Bank kind, std::uint32_t offset, std::uint32_t value) {
        auto& bank = kind == Bank::Shader ? queue.shader : kind == Bank::Context ? queue.context : queue.userConfig;
        const bool had = bank.contains(offset);
        const auto before = had ? std::as_const(bank).at(offset) : 0u;
        if (kind == Bank::UserConfig && offset == IndexTypeOffset) queue.indexType = value & 3u;
        if (had && before == value) {
            ++counts.unchanged;
            return false;
        }
        queue.runMask.Register(kind, offset);
        if ((queue.staleDigests & (1u << static_cast<unsigned>(kind))) == 0 && KeyRegister(kind, offset)) DigestWrite(kind == Bank::Shader ? queue.shaderDigest : kind == Bank::Context ? queue.contextDigest : queue.userConfigDigest, offset, had, before, value);
        bank.insert_or_assign(offset, value);
        ++counts.registers;
        return true;
    }

    void mark(Bank kind) {
        (kind == Bank::Shader ? shader : kind == Bank::Context ? context : userConfig).dirty = true;
    }

    bool direct(QueueState& queue, std::uint32_t opcode, std::span<const std::uint32_t> packet) {
        if (packet.size() < 3) return false;
        if (opcode == static_cast<std::uint32_t>(Opcode::SetUconfigReg) && packet[1] == TagMarkerOffset) return true;
        if (!validOffset(packet[1])) return false;
        const auto offset = packet[1] & ~OffsetFlags;
        if (packet.size() - 2 > static_cast<std::size_t>(LastOffset) + 1 - offset) return false;
        const auto kind = bankOf(opcode);
        bool changed = false;
        for (std::size_t index = 2; index < packet.size(); ++index) changed |= write(queue, kind, offset + static_cast<std::uint32_t>(index - 2), packet[index]);
        if (changed) mark(kind);
        return true;
    }

    bool listed(QueueState& queue, std::uint32_t opcode, std::span<const std::uint32_t> packet, std::span<const std::uint32_t> pairs) {
        if (packet.size() != 5 || pairs.size() != static_cast<std::size_t>(packet[4]) * 2) return false;
        for (std::size_t index = 0; index < pairs.size(); index += 2) {
            if (!validOffset(pairs[index])) return false;
        }
        const auto kind = bankOf(opcode);
        bool changed = false;
        for (std::size_t index = 0; index < pairs.size(); index += 2) changed |= write(queue, kind, pairs[index] & ~OffsetFlags, pairs[index + 1]);
        if (changed) mark(kind);
        return true;
    }

    bool parameter(QueueState& queue, std::uint32_t opcode, std::span<const std::uint32_t> packet) {
        switch (static_cast<Opcode>(opcode)) {
            case Opcode::IndexBufferSize:
                if (packet.size() != 2) return false;
                queue.indexBufferSize = packet[1];
                return true;
            case Opcode::IndexBase:
                if (packet.size() != 3) return false;
                queue.indexBase = packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u);
                return true;
            case Opcode::NumInstances:
                if (packet.size() != 2) return false;
                queue.instanceCount = packet[1];
                return true;
            default:
                return false;
        }
    }

    DecodedState capture(QueueState& queue) {
        adopt(queue);
        take(shader, queue.shader, counts.shaderSnapshots);
        take(context, queue.context, counts.contextSnapshots);
        take(userConfig, queue.userConfig, counts.userConfigSnapshots);
        DecodedState out{};
        out.shader = &shader.slots[shader.current].registers;
        out.context = &context.slots[context.current].registers;
        out.userConfig = &userConfig.slots[userConfig.current].registers;
        out.shaderVersion = shader.version;
        out.contextVersion = context.version;
        out.userConfigVersion = userConfig.version;
        out.indexBase = queue.indexBase;
        out.drawIndirectBase = queue.drawIndirectBase;
        out.dispatchIndirectBase = queue.dispatchIndirectBase;
        out.predicationAddress = queue.predication.address;
        out.shaderDigest = queue.shaderDigest;
        out.contextDigest = queue.contextDigest;
        out.userConfigDigest = queue.userConfigDigest;
        out.indexBufferSize = queue.indexBufferSize.value_or(0);
        out.indexType = queue.indexType;
        out.instanceCount = queue.instanceCount;
        out.predicationOperation = queue.predication.operation;
        out.runBits = queue.runMask.bits;
        out.staleDigests = queue.staleDigests;
        out.hasIndexBufferSize = queue.indexBufferSize.has_value() ? 1 : 0;
        out.predicationExecuteWhenSet = queue.predication.executeWhenSet ? 1 : 0;
        return out;
    }

    void bind(Ring& ring, const Registers* registers, std::uint64_t order) {
        auto& slot = ring.slots[ring.current];
        if (&slot.registers == registers) {
            slot.last = std::max(slot.last, order + 1);
            return;
        }
        for (auto& other : ring.slots) {
            if (&other.registers != registers) continue;
            other.last = std::max(other.last, order + 1);
            return;
        }
        throw std::logic_error("decoder: an operation is bound to a register snapshot this decoder did not take");
    }
};

Decoder::Decoder(Settings settings, DecoderHost host) : state(std::make_unique<State>(settings, host)) {}

Decoder::~Decoder() = default;

DecodedState Decoder::Capture(QueueState& live) {
    return state->capture(live);
}

void Decoder::Bind(const DecodedState& decoded, std::uint64_t order) {
    auto& held = *state;
    held.bind(held.shader, decoded.shader, order);
    held.bind(held.context, decoded.context, order);
    held.bind(held.userConfig, decoded.userConfig, order);
}

void Decoder::Touch(std::uint8_t banks) {
    state->touch(banks);
}

DecoderCounts Decoder::Counted() const {
    return state->counts;
}

Decoder::Result Decoder::Decode(QueueState& live, const Input& input) {
    auto& held = *state;
    held.adopt(live);
    const auto commands = input.commands;
    const auto stop = std::min(input.limit, commands.size());
    auto cursor = input.cursor;
    auto list = input.lists != nullptr ? input.lists->lower_bound(cursor) : Lists::const_iterator{};
    while (cursor < stop) {
        const auto header = commands[cursor];
        const auto lane = LaneOf(header);
        if (lane == Lane::Skip) {
            ++cursor;
            continue;
        }
        const auto words = LaneWords(header);
        if (words > stop - cursor) return {cursor, true};
        const auto packet = commands.subspan(cursor, words);
        const auto opcode = (header >> 8u) & 0xffu;
        bool taken = false;
        switch (lane) {
            case Lane::Register:
                taken = held.direct(live, opcode, packet);
                break;
            case Lane::RegisterList:
                if (input.lists != nullptr) {
                    while (list != input.lists->end() && list->first < cursor) ++list;
                    taken = list != input.lists->end() && list->first == cursor && held.listed(live, opcode, packet, list->second);
                }
                if (taken) ++held.counts.lists;
                else ++held.counts.listsLeft;
                break;
            case Lane::Parameter:
                taken = held.parameter(live, opcode, packet);
                if (taken) ++held.counts.parameters;
                break;
            case Lane::Draw: {
                if (held.host.draw == nullptr) return {cursor, true};
                live.runMask.Packet(opcode);
                DecodedOp op;
                std::memset(static_cast<void*>(&op), 0, sizeof(DecodedOp));
                op.kind = DecodedKind::Draw;
                op.packet = packet.data();
                op.words = static_cast<std::uint32_t>(words);
                if (words <= DecodedInlineWords) std::copy(packet.begin(), packet.end(), op.inlineWords.begin());
                op.offset = static_cast<std::uint32_t>(cursor);
                op.opcode = opcode;
                op.submission = input.submission;
                op.epoch = input.epoch;
                op.state = held.capture(live);
                live.runMask.Drawn();
                const auto order = held.host.draw(held.host.context, op);
                if (order == NoOrder) return {cursor, true};
                Bind(op.state, order);
                ++held.counts.draws;
                ++held.counts.packets;
                cursor += words;
                continue;
            }
            default:
                break;
        }
        if (taken) {
            live.runMask.Packet(opcode);
            ++held.counts.packets;
            cursor += words;
            continue;
        }
        if (held.host.heavy == nullptr) return {cursor, true};
        ++held.counts.heavy;
        const auto outcome = held.host.heavy(held.host.context, cursor, packet, live);
        held.adopt(live);
        held.touch(outcome.changed);
        if (outcome.stop) return {outcome.next, true};
        if (outcome.next <= cursor) throw std::logic_error("decoder: the host did not advance past a packet it was handed");
        if (input.lists != nullptr && outcome.next != cursor + words) list = input.lists->lower_bound(outcome.next);
        cursor = outcome.next;
    }
    return {cursor, false};
}

void ApplyDecoded(const DecodedState& decoded, DecodedView& view) {
    auto& queue = view.queue;
    if (view.shaderVersion != decoded.shaderVersion) {
        queue.shader = *decoded.shader;
        view.shaderVersion = decoded.shaderVersion;
        ++view.copies;
    }
    if (view.contextVersion != decoded.contextVersion) {
        queue.context = *decoded.context;
        view.contextVersion = decoded.contextVersion;
        ++view.copies;
    }
    if (view.userConfigVersion != decoded.userConfigVersion) {
        queue.userConfig = *decoded.userConfig;
        view.userConfigVersion = decoded.userConfigVersion;
        ++view.copies;
    }
    queue.indexBase = decoded.indexBase;
    queue.drawIndirectBase = decoded.drawIndirectBase;
    queue.dispatchIndirectBase = decoded.dispatchIndirectBase;
    if (decoded.hasIndexBufferSize != 0) queue.indexBufferSize = decoded.indexBufferSize;
    else queue.indexBufferSize.reset();
    queue.indexType = decoded.indexType;
    queue.instanceCount = decoded.instanceCount;
    queue.predication = {decoded.predicationAddress, decoded.predicationOperation, decoded.predicationExecuteWhenSet != 0};
    queue.shaderDigest = decoded.shaderDigest;
    queue.contextDigest = decoded.contextDigest;
    queue.userConfigDigest = decoded.userConfigDigest;
    queue.staleDigests = decoded.staleDigests;
    queue.runMask.bits = decoded.runBits;
}

bool SameDecoded(const DecodedState& decoded, const QueueState& queue) {
    if (decoded.indexBase != queue.indexBase || decoded.drawIndirectBase != queue.drawIndirectBase || decoded.dispatchIndirectBase != queue.dispatchIndirectBase) return false;
    if ((decoded.hasIndexBufferSize != 0) != queue.indexBufferSize.has_value() || decoded.indexBufferSize != queue.indexBufferSize.value_or(0)) return false;
    if (decoded.indexType != queue.indexType || decoded.instanceCount != queue.instanceCount) return false;
    if (decoded.predicationAddress != queue.predication.address || decoded.predicationOperation != queue.predication.operation || (decoded.predicationExecuteWhenSet != 0) != queue.predication.executeWhenSet) return false;
    if (decoded.staleDigests != queue.staleDigests) return false;
    if ((decoded.staleDigests & 1u) == 0 && !(decoded.shaderDigest == queue.shaderDigest)) return false;
    if ((decoded.staleDigests & 2u) == 0 && !(decoded.contextDigest == queue.contextDigest)) return false;
    if ((decoded.staleDigests & 4u) == 0 && !(decoded.userConfigDigest == queue.userConfigDigest)) return false;
    return *decoded.shader == queue.shader && *decoded.context == queue.context && *decoded.userConfig == queue.userConfig;
}

}
