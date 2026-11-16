#include "prx/libSceAgcDriver/Core/include/ReplayMeter.hpp"
#include <cstdio>
#include <vector>

namespace AgcDriver::Core {

namespace {

struct Identity {
    std::uint64_t plan = 0;
    std::uint64_t digest0 = 0;
    std::uint64_t digest1 = 0;
    std::uint64_t variant = 0;
    std::uint64_t recipe = 0;
    std::uint64_t inputs = 0;
    std::uint64_t targets = 0;
    std::uint64_t parameters = 0;
    bool operator==(const Identity&) const = default;
};

struct ReplaySlot {
    std::uint64_t key = 0;
    Identity identity;
    std::uint64_t generation = 0;
};

struct VariantSlot {
    std::uint64_t key = 0;
    Identity identity;
};

struct TargetSlot {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t generation = 0;
};

struct TokenSlot {
    std::uint64_t key = 0;
    std::uint32_t count = 0;
    std::array<ReplayToken, ReplayMaxInputs> tokens{};
};

enum class Inputs : std::uint8_t { Clean, Grain, Written, Unwatched, Ranges };

Identity identityOf(const ReplayDraw& draw) {
    return {draw.plan, draw.digest[0], draw.digest[1], draw.variant, draw.recipe, draw.inputs, draw.targets, draw.parameters};
}

std::uint64_t finish(std::uint64_t hash) {
    hash ^= hash >> 33u;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33u;
    hash *= 0xc4ceb9fe1a85ec53ull;
    hash ^= hash >> 33u;
    return hash | 1u;
}

std::uint64_t replayKeyOf(const Identity& identity) {
    auto hash = ReplayMix(0x243f6a8885a308d3ull, identity.plan);
    hash = ReplayMix(hash, identity.digest0);
    hash = ReplayMix(hash, identity.digest1);
    hash = ReplayMix(hash, identity.variant);
    hash = ReplayMix(hash, identity.recipe);
    hash = ReplayMix(hash, identity.inputs);
    hash = ReplayMix(hash, identity.targets);
    hash = ReplayMix(hash, identity.parameters);
    return finish(hash);
}

std::uint64_t variantKeyOf(const Identity& identity) {
    auto hash = ReplayMix(0x13198a2e03707344ull, identity.plan);
    hash = ReplayMix(hash, identity.digest0);
    hash = ReplayMix(hash, identity.digest1);
    return finish(hash);
}

bool sameVariant(const Identity& a, const Identity& b) {
    return a.plan == b.plan && a.digest0 == b.digest0 && a.digest1 == b.digest1;
}

Inputs inputsOf(const ReplayDraw& draw, std::uint64_t since, ReplayProof& proof) {
    if (!draw.rangesComplete) return Inputs::Ranges;
    if (draw.rangeCount == 0) return Inputs::Clean;
    std::array<ReplayQuery, ReplayMaxInputs> queries{};
    for (std::uint32_t index = 0; index < draw.rangeCount; ++index) queries[index] = {draw.ranges[index].address, draw.ranges[index].bytes, since};
    const std::span<const ReplayQuery> asked(queries.data(), draw.rangeCount);
    bool blocks = false;
    for (const auto& query : asked) {
        if (!proof.BlockWritten(query)) continue;
        blocks = true;
        break;
    }
    const bool pages = proof.PagesUnchanged(asked);
    if (pages) return blocks ? Inputs::Grain : Inputs::Clean;
    return blocks ? Inputs::Written : Inputs::Unwatched;
}

ReplayTokenMiss tokenMissOf(Inputs inputs) {
    switch (inputs) {
        case Inputs::Grain: return ReplayTokenMiss::Grain;
        case Inputs::Written: return ReplayTokenMiss::Written;
        case Inputs::Unwatched: return ReplayTokenMiss::Unwatched;
        case Inputs::Ranges: return ReplayTokenMiss::Ranges;
        case Inputs::Clean: break;
    }
    return ReplayTokenMiss::TokenOnly;
}

double share(std::uint64_t part, std::uint64_t whole) {
    return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

unsigned long long wide(std::uint64_t value) {
    return static_cast<unsigned long long>(value);
}

}

std::uint64_t ReplayMix(std::uint64_t hash, std::uint64_t value) {
    hash ^= value * 0xc2b2ae3d27d4eb4full + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    hash *= 0x9e3779b97f4a7c15ull;
    return hash ^ (hash >> 31u);
}

void AddReplayRange(ReplayDraw& draw, std::uint64_t address, std::uint64_t bytes) {
    if (address == 0 || bytes == 0 || bytes > ~0ull - address) return;
    for (std::uint32_t index = 0; index < draw.rangeCount; ++index) {
        auto& range = draw.ranges[index];
        if (address > range.address + range.bytes || range.address > address + bytes) continue;
        const auto begin = range.address < address ? range.address : address;
        const auto end = range.address + range.bytes > address + bytes ? range.address + range.bytes : address + bytes;
        range = {begin, end - begin};
        return;
    }
    if (draw.rangeCount == draw.ranges.size()) {
        draw.rangesComplete = false;
        return;
    }
    draw.ranges[draw.rangeCount++] = {address, bytes};
}

void AddReplayMetadata(ReplayDraw& draw, std::uint64_t address, std::uint64_t bytes) {
    if (address == 0 || bytes == 0 || bytes > ~0ull - address) return;
    for (std::uint32_t index = 0; index < draw.metadataCount; ++index) {
        if (draw.metadata[index].address == address && draw.metadata[index].bytes == bytes) return;
    }
    if (draw.metadataCount == draw.metadata.size()) return;
    draw.metadata[draw.metadataCount++] = {address, bytes};
}

const char* ReplayCauseName(ReplayCause cause) {
    switch (cause) {
        case ReplayCause::Replayed: return "replayed";
        case ReplayCause::New: return "new";
        case ReplayCause::Evicted: return "evicted";
        case ReplayCause::Recipe: return "recipe";
        case ReplayCause::Input: return "input range";
        case ReplayCause::Target: return "target";
        case ReplayCause::Parameters: return "parameters";
        case ReplayCause::Fallback: return "fallback";
        case ReplayCause::Fill: return "fill";
        case ReplayCause::Token: return "token";
        case ReplayCause::Unproved: return "unproved";
        case ReplayCause::Count: break;
    }
    return "?";
}

const char* ReplayTokenMissName(ReplayTokenMiss miss) {
    switch (miss) {
        case ReplayTokenMiss::Grain: return "grain";
        case ReplayTokenMiss::Written: return "written";
        case ReplayTokenMiss::Unwatched: return "unwatched";
        case ReplayTokenMiss::Ranges: return "ranges";
        case ReplayTokenMiss::TokenOnly: return "token only";
        case ReplayTokenMiss::Count: break;
    }
    return "?";
}

struct ReplayMeter::State {
    std::vector<ReplaySlot> replay;
    std::vector<std::uint64_t> evicted;
    std::vector<VariantSlot> variants;
    std::vector<TargetSlot> targets;
    std::vector<TokenSlot> tokens;
    std::uint64_t replayMask = 0;
    std::uint64_t variantMask = 0;
    std::uint64_t targetMask = 0;
    std::uint32_t replayBits = 0;
    std::uint32_t proofShift = 0;
    std::uint64_t proofMask = 0;
    std::uint64_t sampler = 0x9e3779b97f4a7c15ull;
    ReplayCounts counts;
    ReplayCounts reported;
    std::uint64_t reportedFrontDraws = 0;
    std::uint64_t reportedFrontCommitted = 0;
};

ReplayMeter::ReplayMeter(std::uint32_t replayBits, std::uint32_t variantBits, std::uint32_t targetBits, std::uint32_t proofShift) : state(std::make_unique<State>()) {
    state->replayBits = replayBits;
    state->proofShift = proofShift;
    state->proofMask = (std::uint64_t{1} << proofShift) - 1u;
    state->replay.resize(std::size_t{1} << replayBits);
    state->evicted.resize(std::size_t{1} << replayBits);
    state->variants.resize(std::size_t{1} << variantBits);
    state->targets.resize(std::size_t{1} << targetBits);
    state->replayMask = (std::uint64_t{1} << replayBits) - 1u;
    state->variantMask = (std::uint64_t{1} << variantBits) - 1u;
    state->targetMask = (std::uint64_t{1} << targetBits) - 1u;
}

ReplayMeter::~ReplayMeter() = default;

bool ReplayMeter::Filled(const ReplayDraw& draw, ReplayProof& proof) const {
    std::array<ReplayQuery, ReplayMaxMetadata> queries{};
    std::size_t count = 0;
    for (std::uint32_t index = 0; index < draw.metadataCount; ++index) {
        const auto& metadata = draw.metadata[index];
        const auto& slot = state->targets[finish(metadata.address) & state->targetMask];
        if (slot.address != metadata.address || slot.bytes != metadata.bytes || slot.generation == 0) continue;
        queries[count++] = {metadata.address, metadata.bytes, slot.generation};
    }
    if (count == 0) return false;
    if (proof.PagesUnchanged(std::span<const ReplayQuery>(queries.data(), count))) return false;
    for (std::size_t index = 0; index < count; ++index) {
        if (!proof.PagesUnchanged(std::span<const ReplayQuery>(&queries[index], 1)) && proof.BlockWritten(queries[index])) return true;
    }
    return false;
}

ReplayCause ReplayMeter::Commit(const ReplayDraw& draw, bool fallback, bool filled, ReplayProof& proof) {
    auto& held = *state;
    auto& counts = held.counts;
    ++counts.draws;
    if (filled) ++counts.afterFill;
    const auto now = proof.Generation();
    const bool live = proof.TokensLive();
    const auto identity = identityOf(draw);
    const auto replayKey = replayKeyOf(identity);
    const auto variantKey = variantKeyOf(identity);
    auto& slot = held.replay[replayKey & held.replayMask];
    auto& known = held.variants[variantKey & held.variantMask];
    auto& evicted = held.evicted[(replayKey >> held.replayBits) & held.replayMask];
    const bool same = slot.key == replayKey && slot.identity == identity;
    if (same) ++counts.sameKey;
    if (live && held.tokens.empty()) held.tokens.resize(held.replay.size());
    const auto sampled = [&] {
        if (held.proofMask == 0) return true;
        held.sampler ^= held.sampler << 13u;
        held.sampler ^= held.sampler >> 7u;
        held.sampler ^= held.sampler << 17u;
        return (held.sampler & held.proofMask) == 0;
    };
    ReplayCause cause = ReplayCause::New;
    if (fallback) {
        cause = ReplayCause::Fallback;
    } else if (same && filled) {
        cause = ReplayCause::Fill;
    } else if (same && !sampled()) {
        cause = ReplayCause::Unproved;
    } else if (same) {
        const auto inputs = inputsOf(draw, slot.generation, proof);
        bool current = true;
        if (live) {
            const auto& tokens = held.tokens[replayKey & held.replayMask];
            if (tokens.key != replayKey || tokens.count != draw.rangeCount || !draw.rangesComplete) current = false;
            for (std::uint32_t index = 0; current && index < draw.rangeCount; ++index) current = proof.StillCurrent(draw.ranges[index], tokens.tokens[index]);
            ++(current ? counts.tokensTrue : counts.tokensFalse);
        }
        const bool proved = live ? current : inputs == Inputs::Clean;
        if (!proved) {
            cause = ReplayCause::Token;
            ++counts.tokenMisses[static_cast<std::size_t>(tokenMissOf(inputs))];
        } else {
            cause = ReplayCause::Replayed;
        }
    } else if (evicted == replayKey) {
        cause = ReplayCause::Evicted;
    } else if (known.key == variantKey && sameVariant(known.identity, identity)) {
        if (known.identity.variant != identity.variant || known.identity.recipe != identity.recipe) cause = ReplayCause::Recipe;
        else if (known.identity.inputs != identity.inputs) cause = ReplayCause::Input;
        else if (known.identity.targets != identity.targets) cause = ReplayCause::Target;
        else if (known.identity.parameters != identity.parameters) cause = ReplayCause::Parameters;
        else cause = ReplayCause::Evicted;
    }
    ++counts.causes[static_cast<std::size_t>(cause)];
    if (slot.key != 0 && slot.key != replayKey) held.evicted[(slot.key >> held.replayBits) & held.replayMask] = slot.key;
    if (evicted == replayKey) evicted = 0;
    slot.key = replayKey;
    slot.identity = identity;
    slot.generation = now;
    known.key = variantKey;
    known.identity = identity;
    if (live) {
        auto& tokens = held.tokens[replayKey & held.replayMask];
        tokens.key = replayKey;
        tokens.count = draw.rangeCount;
        for (std::uint32_t index = 0; index < draw.rangeCount; ++index) tokens.tokens[index] = proof.TokenFor(draw.ranges[index]);
    }
    for (std::uint32_t index = 0; index < draw.metadataCount; ++index) {
        const auto& metadata = draw.metadata[index];
        held.targets[finish(metadata.address) & held.targetMask] = {metadata.address, metadata.bytes, now};
    }
    return cause;
}

void ReplayMeter::NoteCost(std::uint64_t nanoseconds) {
    state->counts.costNanoseconds += nanoseconds;
    ++state->counts.costSamples;
}

const ReplayCounts& ReplayMeter::Counted() const {
    return state->counts;
}

std::string ReplayMeter::TakeReport(std::uint64_t frontDraws, std::uint64_t frontCommitted, bool tokensLive, double windowSeconds) {
    const auto& counts = state->counts;
    const auto& before = state->reported;
    const auto cause = [](const ReplayCounts& from, ReplayCause which) { return from.causes[static_cast<std::size_t>(which)]; };
    const auto miss = [&](ReplayTokenMiss which) { return wide(counts.tokenMisses[static_cast<std::size_t>(which)]); };
    const auto estimate = [](std::uint64_t replayed, std::uint64_t token, std::uint64_t unproved) {
        const auto proved = replayed + token;
        if (proved == 0) return static_cast<double>(replayed);
        return static_cast<double>(replayed) + static_cast<double>(unproved) * static_cast<double>(replayed) / static_cast<double>(proved);
    };
    const auto replayed = cause(counts, ReplayCause::Replayed);
    const auto token = cause(counts, ReplayCause::Token);
    const auto unproved = cause(counts, ReplayCause::Unproved);
    const auto replayable = estimate(replayed, token, unproved);
    const auto tokenMissed = static_cast<double>(token + unproved) - (replayable - static_cast<double>(replayed));
    const auto windowDraws = counts.draws - before.draws;
    const auto windowReplayable = estimate(replayed - cause(before, ReplayCause::Replayed), token - cause(before, ReplayCause::Token), unproved - cause(before, ReplayCause::Unproved));
    const auto windowFront = frontDraws >= state->reportedFrontDraws ? frontDraws - state->reportedFrontDraws : 0u;
    const auto windowCommitted = frontCommitted >= state->reportedFrontCommitted ? frontCommitted - state->reportedFrontCommitted : 0u;
    const auto percent = [](double part, std::uint64_t whole) { return whole == 0 ? 0.0 : 100.0 * part / static_cast<double>(whole); };
    char text[1792];
    std::snprintf(text, sizeof(text), "[core-replay] (cumulative) %llu draws metered at the core commit, %.0f replayable (%.1f %%); front: %llu draws looked at, %llu committed by the core (%.1f %%), replayable per draw looked at %.1f %%; last window %.1f s: %.0f of %llu metered draws replayable (%.1f %%), front %llu draws looked at, %llu committed by the core; not replayable: new %llu, evicted %llu, recipe %llu, input range %llu, target %llu, parameters %llu, fallback %llu, fill %llu, token %.0f; input proof on 1 in %llu same-key draws: %llu proved, %llu replayable, %llu token misses (grain %llu, written %llu, unwatched %llu, ranges %llu, token only %llu); same key %llu; after a keys or HTILE fill %llu; proof: %s; T1 tokens %llu still current, %llu not; meter cost %.0f ns per metered draw (%llu timed)",
        wide(counts.draws), replayable, percent(replayable, counts.draws), wide(frontDraws), wide(frontCommitted), share(frontCommitted, frontDraws), percent(replayable, frontDraws),
        windowSeconds, windowReplayable, wide(windowDraws), percent(windowReplayable, windowDraws), wide(windowFront), wide(windowCommitted),
        wide(cause(counts, ReplayCause::New)), wide(cause(counts, ReplayCause::Evicted)), wide(cause(counts, ReplayCause::Recipe)), wide(cause(counts, ReplayCause::Input)), wide(cause(counts, ReplayCause::Target)), wide(cause(counts, ReplayCause::Parameters)), wide(cause(counts, ReplayCause::Fallback)), wide(cause(counts, ReplayCause::Fill)), tokenMissed,
        wide(std::uint64_t{1} << state->proofShift), wide(replayed + token), wide(replayed), wide(token), miss(ReplayTokenMiss::Grain), miss(ReplayTokenMiss::Written), miss(ReplayTokenMiss::Unwatched), miss(ReplayTokenMiss::Ranges), miss(ReplayTokenMiss::TokenOnly),
        wide(counts.sameKey), wide(counts.afterFill), tokensLive ? "T1 content token" : "write tracker at the 64 KiB block grain (T1 off)", wide(counts.tokensTrue), wide(counts.tokensFalse),
        counts.costSamples == 0 ? 0.0 : static_cast<double>(counts.costNanoseconds) / static_cast<double>(counts.costSamples), wide(counts.costSamples));
    state->reported = counts;
    state->reportedFrontDraws = frontDraws;
    state->reportedFrontCommitted = frontCommitted;
    return text;
}

}
