#include "prx/libSceAgcDriver/Core/include/Front.hpp"
#include "prx/libSceAgcDriver/Core/include/Back.hpp"
#include "prx/libSceAgcDriver/Core/include/CoarseClock.hpp"
#include "prx/libSceAgcDriver/Core/include/ResourceCache.hpp"
#include "prx/libSceAgcDriver/Core/include/RunPlan.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawRegisterKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DrawBlock.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/UnitShadow.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include "Optimization/BlockLayout.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "Optimization/SrtWalker/SrtTape.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace AgcDriver::Core {

namespace {

using Range = std::pair<std::uint64_t, std::uint64_t>;
using Role = ShaderRecompiler::DescriptorRole;

constexpr std::uint32_t NoBinding = 0xffffffffu;
constexpr std::uint64_t EveryProgram = 1;

struct Counters {
    std::atomic<std::uint64_t> draws{0}, continued{0}, plans{0}, taken{0}, takenContinued{0}, taught{0}, verified{0}, mismatches{0}, digests{0}, committed{0}, restarted{0}, walkNs{0}, frontNs{0}, backNs{0}, retries{0}, flatTaken{0}, flatSame{0}, flatNs{0}, flatBytes{0}, flatTimed{0}, flatVertexKnown{0}, flatZoned{0}, filtered{0}, released{0}, flatPostNs{0}, flatPostTimed{0}, flatPostBusy{0};
    std::array<std::atomic<std::uint64_t>, 8> sections{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FrontLeft::Count)> left{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(TeachRefusal::Count)> refused{};
};

Counters& counters() {
    static Counters held;
    return held;
}

void add(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

struct Reader {
    std::vector<Range>* ranges = nullptr;
};

bool readWord(void* user, std::uint64_t address, std::uint32_t* value) {
    auto& reader = *static_cast<Reader*>(user);
    if (!ReadGuestBytes(address, std::as_writable_bytes(std::span<std::uint32_t>(value, 1)))) return false;
    auto& ranges = *reader.ranges;
    if (!ranges.empty() && ranges.back().second == address) ranges.back().second = address + sizeof(std::uint32_t);
    else ranges.emplace_back(address, address + sizeof(std::uint32_t));
    return true;
}

bool walkStage(const PlanStage& stage, std::span<const std::uint32_t> userData, std::uint64_t shaderBase, std::vector<Range>& ranges, StageWalk& walk) {
    Reader reader{&ranges};
    ShaderRecompiler::SrtRuntime runtime;
    runtime.userData = userData;
    runtime.shaderBase = shaderBase;
    runtime.userContext = &reader;
    runtime.readMemory = &readWord;
    runtime.readSpecializationMemory = &readWord;
    return stage.tape->Run(*stage.plan, runtime, walk.results, walk.flat, walk.active);
}

void mix(std::array<std::uint64_t, 2>& digest, std::uint64_t value) {
    digest[0] = (digest[0] ^ value) * 0x9e3779b97f4a7c15ull;
    digest[0] ^= digest[0] >> 29u;
    digest[1] = (digest[1] ^ (value + 0x7f4a7c159e3779b9ull)) * 0xc2b2ae3d27d4eb4full;
    digest[1] ^= digest[1] >> 31u;
}

bool teachBounds() {
    static const bool bounded = std::getenv("APS5_NO_CORE_TEACH_BOUNDS") == nullptr;
    return bounded;
}

std::uint64_t pairOf(std::span<const DriverDetail::DrawProgram> programs, std::span<const Graphics::CompiledShader> stages) {
    if (programs.empty() || programs.size() != stages.size()) return 0;
    std::array<std::uint64_t, 2> digest{0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull};
    for (std::size_t index = 0; index < programs.size(); ++index) {
        if (stages[index].program == nullptr) return 0;
        mix(digest, programs[index].binary.codeAddress);
        mix(digest, reinterpret_cast<std::uintptr_t>(programs[index].snapshot.get()));
        mix(digest, programs[index].codeOffset);
        mix(digest, stages[index].program->variantId);
        mix(digest, static_cast<std::uint64_t>(stages[index].stage));
    }
    const auto hash = digest[0] ^ std::rotl(digest[1], 31);
    return hash == 0 ? 1 : hash;
}

struct PlanRefusals {
    std::mutex mutex;
    std::map<std::pair<std::uint64_t, std::uint32_t>, std::uint64_t> counts;
    std::uint64_t total = 0;
};

void notePlanRefusal(std::uint64_t program, std::uint32_t stage, std::uint32_t reason) {
    static const bool enabled = std::getenv("APS5_CORE_PLAN_DIAG") != nullptr;
    if (!enabled) return;
    static PlanRefusals held;
    const std::lock_guard lock(held.mutex);
    ++held.counts[{program, stage * 4u + reason}];
    ++held.total;
    if (held.total % 100000u != 0 && held.total != 1000u && held.total != 10000u) return;
    std::vector<std::pair<std::uint64_t, std::pair<std::uint64_t, std::uint32_t>>> top;
    for (const auto& [key, count] : held.counts) top.emplace_back(count, key);
    std::sort(top.begin(), top.end(), std::greater<>());
    static constexpr std::array<const char*, 4> names{"no variant", "variant without a plan", "incomplete plan", "uncacheable capture"};
    std::string text = "[core-plan] (cumulative) " + std::to_string(held.total) + " draws refused for plan, " + std::to_string(held.counts.size()) + " program and reason pairs;";
    for (std::size_t index = 0; index < top.size() && index < 10; ++index) {
        char item[128];
        std::snprintf(item, sizeof(item), " 0x%llx stage %u %s %llu;", static_cast<unsigned long long>(top[index].second.first), top[index].second.second / 4u, names[top[index].second.second % 4u], static_cast<unsigned long long>(top[index].first));
        text += item;
    }
    std::fprintf(stderr, "%s%c", text.c_str(), 10);
}

struct KeyDiagState {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>> last;
    std::map<std::uint32_t, std::uint64_t> differed;
    std::uint64_t compared = 0;
    std::uint64_t first = 0;
    std::uint64_t equal = 0;
    std::uint64_t registers = 0;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

std::atomic<std::uint64_t>& variantSerial() {
    static std::atomic<std::uint64_t> held{0};
    return held;
}

void mixWalk(std::array<std::uint64_t, 2>& digest, const PlanStage& stage, const StageWalk& walk) {
    mix(digest, walk.results.size());
    for (std::size_t index = 0; index < walk.results.size(); ++index) {
        if (index >= stage.keyResult.size() || stage.keyResult[index] == 0u) continue;
        const auto& value = walk.results[index];
        mix(digest, value.dwordCount | (static_cast<std::uint64_t>(value.availableDwordMask) << 32u));
        for (std::uint32_t dword = 0; dword < value.dwordCount && dword < value.dwords.size(); ++dword) mix(digest, value.dwords[dword]);
    }
    for (const auto active : walk.active) mix(digest, active);
}

bool sameKey(const PlanStage& stage, const StageWalk& a, const StageWalk& b) {
    if (a.active != b.active || a.results.size() != b.results.size()) return false;
    for (std::size_t index = 0; index < a.results.size(); ++index) {
        if (index < stage.keyResult.size() && stage.keyResult[index] != 0u && !(a.results[index] == b.results[index])) return false;
    }
    return true;
}

std::uint32_t userWordCount(const QueueState& queue, const DriverDetail::DrawProgram& program) {
    if (program.nullPixel) return 0;
    const auto found = queue.shader.find(program.resourceRegister);
    const auto resources = found == queue.shader.end() ? 0u : found->second;
    return ((resources >> 1u) & 0x1fu) | (((resources >> 27u) & 1u) << 5u);
}

bool decodeVertex(const DriverDetail::DrawProgram& program, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderVertexStageInfo& was, std::vector<Graphics::DecodeRead>& reads, ShaderRecompiler::ShaderVertexStageInfo& fresh) {
    reads.clear();
    try {
        fresh = Graphics::DecodeVertexStageInfo(program.binary.header, program.binary.headerAddress, userData, &reads, false, 8u - program.firstUserSgpr);
    } catch (const std::exception&) {
        return false;
    }
    fresh.paClVsOutCntl = was.paClVsOutCntl;
    fresh.depthRemapMode = was.depthRemapMode;
    fresh.pointList = was.pointList;
    fresh.pointSize = was.pointSize;
    fresh.pointSizeMin = was.pointSizeMin;
    fresh.pointSizeMax = was.pointSizeMax;
    return true;
}

std::uint32_t bindingOf(const ShaderRecompiler::RecompileResult& result, Role role) {
    for (std::uint32_t index = 0; index < result.bindings.size(); ++index) {
        if (result.bindings[index].role == role) return index;
    }
    return NoBinding;
}

bool sameAttribute(const ShaderRecompiler::VertexAttribute& a, const ShaderRecompiler::VertexAttribute& b) {
    return a.location == b.location && a.components == b.components && a.fetchIndex == b.fetchIndex && a.formatComponents == b.formatComponents && a.resource.fields == b.resource.fields;
}

bool sameBinding(const ShaderRecompiler::DescriptorBinding& a, const ShaderRecompiler::DescriptorBinding& b) {
    return a.kind == b.kind && a.role == b.role && a.binding == b.binding && a.count == b.count && a.guestDescriptor == b.guestDescriptor;
}

void bump(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

constexpr std::uint32_t OpWords = static_cast<std::uint32_t>(sizeof(Op) / sizeof(std::uint32_t));
constexpr std::uint32_t DrawIndex8 = 4u;
constexpr std::size_t MostRanges = 96;
constexpr std::size_t MostReads = 160;
constexpr std::size_t MostUserWords = 4u * 64u;

struct FlatReader {
    std::array<Range, MostRanges> ranges;
    std::size_t count = 0;
    std::array<Range, MostReads> reads;
    std::size_t readCount = 0;
    bool overflow = false;

    bool Note(std::uint64_t begin, std::uint64_t end) {
        if (count != 0 && ranges[count - 1].second == begin) {
            ranges[count - 1].second = end;
            return true;
        }
        if (count == ranges.size()) {
            overflow = true;
            return false;
        }
        ranges[count++] = {begin, end};
        return true;
    }

    bool Reads(std::uint64_t begin, std::uint64_t bytes) {
        if (begin == 0 || bytes == 0 || bytes > ~std::uint64_t{0} - begin) return true;
        if (readCount == reads.size()) {
            overflow = true;
            return false;
        }
        reads[readCount++] = {begin, begin + bytes};
        return true;
    }
};

bool flatReadWords(void* context, std::uint64_t address, std::uint32_t* out, std::uint32_t count) {
    auto& reader = *static_cast<FlatReader*>(context);
    const auto words = std::span<std::uint32_t>(out, count);
    if (!ReadGuestBytes(address, std::as_writable_bytes(words))) return false;
    return reader.Note(address, address + words.size_bytes());
}

void mixZone(std::array<std::uint64_t, 2>& digest, const PlanStage& stage, std::span<const std::uint32_t> zone) {
    const auto& zoneOf = stage.tape->Driver();
    mix(digest, zoneOf.sources.size());
    for (std::size_t index = 0; index < zoneOf.sources.size(); ++index) {
        if (index >= stage.keyResult.size() || stage.keyResult[index] == 0u) continue;
        const auto& source = zoneOf.sources[index];
        for (std::uint32_t dword = 0; dword < source.dwords; ++dword) mix(digest, zone[source.word + dword]);
    }
    const auto activeEnd = std::min<std::size_t>(zone.size(), std::size_t{zoneOf.activeWord} + (zoneOf.activeBits + 31u) / 32u);
    for (std::size_t word = zoneOf.activeWord; word < activeEnd; ++word) mix(digest, zone[word]);
}

bool sameZone(const PlanStage& stage, std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) {
    const auto& zoneOf = stage.tape->Driver();
    if (a.size() != b.size() || a.size() != zoneOf.words) return false;
    for (std::size_t index = 0; index < zoneOf.sources.size(); ++index) {
        if (index >= stage.keyResult.size() || stage.keyResult[index] == 0u) continue;
        const auto& source = zoneOf.sources[index];
        if (!std::equal(a.begin() + source.word, a.begin() + source.word + source.dwords, b.begin() + source.word)) return false;
    }
    const auto activeEnd = std::min<std::size_t>(a.size(), std::size_t{zoneOf.activeWord} + (zoneOf.activeBits + 31u) / 32u);
    for (std::size_t word = zoneOf.activeWord; word < activeEnd; ++word) {
        if (a[word] != b[word]) return false;
    }
    return true;
}

bool flatReadWord(void* context, std::uint64_t address, std::uint32_t* value) {
    auto& reader = *static_cast<FlatReader*>(context);
    if (!ReadGuestBytes(address, std::as_writable_bytes(std::span<std::uint32_t>(value, 1)))) return false;
    return reader.Note(address, address + sizeof(std::uint32_t));
}

RingChunk plainChunk(void* context, std::size_t bytes) {
    static_cast<void>(context);
    auto* memory = static_cast<std::byte*>(std::malloc(bytes));
    if (memory == nullptr) return {};
    RingChunk chunk;
    chunk.bytes = memory;
    chunk.deviceAddress = reinterpret_cast<std::uintptr_t>(memory);
    chunk.owner = std::shared_ptr<void>(memory, [](void* held) { std::free(held); });
    return chunk;
}

std::uint64_t steadyNanoseconds(void* context) {
    static_cast<void>(context);
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::uint32_t flatWords(const RunPlanData& data, const FrontVariant& variant) {
    std::uint32_t words = OpWords;
    for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
        words += data.stages[stage].flatWords + static_cast<std::uint32_t>(data.stages[stage].bufferResult.size()) * 4u + data.stages[stage].userWords;
        if (variant.stages[stage].vertexInfo) words += static_cast<std::uint32_t>(variant.stages[stage].base->vertexAttributes.size()) * 4u;
    }
    return words;
}

struct VertexMemo {
    std::uint64_t variant = 0;
    std::uint32_t stage = 0;
    std::vector<std::uint32_t> userData;
    std::vector<Graphics::DecodeRead> reads;
    ShaderRecompiler::ShaderVertexStageInfo fresh{};
};

struct FlatState {
    std::array<VertexMemo, 256> vertexMemo;
    BlockRing ring{RingSource{&plainChunk, nullptr, nullptr}};
    std::vector<std::shared_ptr<FrontVariant>> kept;
    ImageSetCache zoned{std::size_t{1} << 16u};
    std::shared_ptr<const RunPlanData> plan;
    StateKey key;
    std::shared_ptr<FrontVariant> variant;
    std::array<StageWalk, 4> walks;
    std::array<std::uint64_t, 2> digest{};
    FlatReader reader;
    std::vector<Graphics::DecodeRead> decodeReads;
    std::array<std::uint32_t, MostUserWords> userData{};
    std::array<ShaderRecompiler::ShaderVertexStageInfo, 4> vertex{};
    CoarseClock clock{&steadyNanoseconds, nullptr, Graphics::CoreClockPeriod()};
};

struct Pool {
    std::mutex mutex;
    std::vector<std::shared_ptr<CoreDraw>> spare;
};

Pool& pool() {
    static Pool held;
    return held;
}

}

const char* FrontLeftName(FrontLeft reason) {
    static constexpr std::array<const char*, static_cast<std::size_t>(FrontLeft::Count)> names{"entry", "no plan", "programs", "user words", "walk", "no variant", "variant key", "views", "vertex", "queued", "pending", "moved", "stale", "finisher", "nothing to draw", "precheck", "ring", "ranges"};
    return names[static_cast<std::size_t>(reason)];
}

const char* TeachRefusalName(TeachRefusal reason) {
    static constexpr std::array<const char*, static_cast<std::size_t>(TeachRefusal::Count)> names{"none", "path", "program", "stage", "plan", "tape", "walk", "words", "buffers", "forwarded", "vertex", "writes", "plan differs", "driver zone", "remembered"};
    return names[static_cast<std::size_t>(reason)];
}

struct Front::State {
    std::mutex mutex;
    RunPlans plans;
    ImageSetCache variants{std::size_t{1} << 16u};
    std::mutex inboxMutex;
    std::vector<std::pair<StateKey, std::shared_ptr<FrontVariant>>> inbox;
    std::atomic<bool> inboxed{false};
    std::vector<Range> ranges;
    std::vector<Graphics::DecodeRead> decodeReads;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> registers;
    std::vector<std::uint64_t> teaching;
    std::uint64_t walked = 0;
    FlatState flat;
    TeachFilter filter;
    RefusalMemory refusals;
    std::vector<std::shared_ptr<FrontVariant>> leaving;
    std::vector<std::shared_ptr<void>> trash;

    bool Wanted(std::uint64_t asked, std::uint64_t salt) {
        if (teachBounds()) {
            if (filter.Ask(asked ^ salt, walked)) return true;
            add(counters().filtered);
            return false;
        }
        if (std::find(teaching.begin(), teaching.end(), asked) != teaching.end()) return false;
        teaching.push_back(asked);
        return true;
    }

    void Forgotten() {
        if (!teachBounds()) return;
        filter.Clear();
        refusals.Clear();
    }

    void Sweep(std::vector<std::shared_ptr<void>>& gone) {
        std::vector<std::shared_ptr<FrontVariant>> staying;
        for (auto& variant : leaving) {
            if (variant->pending.load(std::memory_order_acquire) == 0) gone.push_back(std::move(variant));
            else staying.push_back(std::move(variant));
        }
        leaving.swap(staying);
        add(counters().released, gone.size());
        auto plansGone = plans.Collect();
        if (!plansGone.empty()) gone.push_back(std::make_shared<std::vector<std::unique_ptr<RunPlan>>>(std::move(plansGone)));
    }

    void Arrive() {
        auto& held = counters();
        const bool bounded = teachBounds();
        std::vector<std::shared_ptr<void>> gone;
        if (bounded) Sweep(gone);
        teaching.clear();
        std::vector<std::pair<StateKey, std::shared_ptr<FrontVariant>>> arrived;
        {
            const std::lock_guard inboxLock(inboxMutex);
            arrived.swap(inbox);
        }
        for (auto& [key, variant] : arrived) {
            const auto* known = plans.Find(key);
            if (known == nullptr) {
                RunPlan plan;
                plan.key = key;
                plan.stageCount = variant->plan->stageCount;
                plan.resources = std::const_pointer_cast<RunPlanData>(variant->plan);
                known = plans.Teach(std::move(plan));
                std::const_pointer_cast<RunPlanData>(variant->plan)->id = known->id;
            } else {
                const auto data = std::static_pointer_cast<const RunPlanData>(known->resources);
                bool same = data->stageCount == variant->plan->stageCount;
                for (std::uint32_t stage = 0; same && stage < data->stageCount; ++stage) same = data->stages[stage].plan == variant->plan->stages[stage].plan;
                if (!same) {
                    add(held.refused[static_cast<std::size_t>(TeachRefusal::PlanDiffers)]);
                    if (bounded) gone.push_back(std::move(variant));
                    continue;
                }
                variant->plan = data;
            }
            ImageSetValue value;
            value.owner = variant;
            const auto keep = [&](ImageSetCache& cache, const ImageSetKey& slot) {
                auto displaced = cache.Keep(slot, Proof{}, value);
                if (bounded && displaced.owner != nullptr && displaced.owner != value.owner) leaving.push_back(std::static_pointer_cast<FrontVariant>(std::move(displaced.owner)));
            };
            keep(variants, ImageSetKey{variant->plan->id, variant->digest});
            add(held.taught);
            if (!bounded) flat.kept.push_back(variant);
            const auto& data = *variant->plan;
            if (!variant->zoned) {
                data.zoned = 2;
            } else {
                keep(flat.zoned, ImageSetKey{data.id, variant->zoneDigest});
                if (data.zoned == 0) data.zoned = 1;
            }
            if (bounded) gone.push_back(std::move(variant));
        }
        flat.plan = nullptr;
        flat.variant = nullptr;
        if (gone.empty()) return;
        const std::lock_guard inboxLock(inboxMutex);
        for (auto& item : gone) trash.push_back(std::move(item));
    }
};

Front::Front() : state(std::make_unique<State>()) {}

Front& Front::Get() {
    static Front* front = new Front();
    return *front;
}

FrontCounts Front::Counted() {
    const auto& held = counters();
    const auto load = [](const std::atomic<std::uint64_t>& counter) { return counter.load(std::memory_order_relaxed); };
    FrontCounts counts;
    counts.draws = load(held.draws);
    counts.continued = load(held.continued);
    counts.plans = load(held.plans);
    counts.taken = load(held.taken);
    counts.takenContinued = load(held.takenContinued);
    counts.taught = load(held.taught);
    counts.verified = load(held.verified);
    counts.mismatches = load(held.mismatches);
    counts.digests = load(held.digests);
    counts.committed = load(held.committed);
    counts.restarted = load(held.restarted);
    counts.walkNs = load(held.walkNs);
    counts.frontNs = load(held.frontNs);
    counts.backNs = load(held.backNs);
    counts.retries = load(held.retries);
    counts.flatTaken = load(held.flatTaken);
    counts.flatSame = load(held.flatSame);
    counts.flatNs = load(held.flatNs);
    counts.flatBytes = load(held.flatBytes);
    counts.flatTimed = load(held.flatTimed);
    counts.flatVertexKnown = load(held.flatVertexKnown);
    counts.flatZoned = load(held.flatZoned);
    counts.filtered = load(held.filtered);
    counts.released = load(held.released);
    for (std::size_t index = 0; index < counts.left.size(); ++index) counts.left[index] = load(held.left[index]);
    for (std::size_t index = 0; index < counts.refused.size(); ++index) counts.refused[index] = load(held.refused[index]);
    return counts;
}

std::string Front::Report() {
    const auto& held = counters();
    const auto counts = Counted();
    const auto per = [](std::uint64_t nanoseconds, std::uint64_t count) { return count != 0 ? static_cast<double>(nanoseconds) / 1000.0 / static_cast<double>(count) : 0.0; };
    char head[768];
    std::snprintf(head, sizeof(head), "[core-run] (cumulative) %llu draws, %llu continued, %llu with a plan, %llu taken by the core (%llu continued) at %.2f us at the front (%.2f us of walks per draw looked at) and %.2f us per commit, %llu commits, %llu restarted by the full path; %llu variants taught, %llu walks made twice; verify: %llu digests, %llu candidates, %llu mismatches; left:", static_cast<unsigned long long>(counts.draws), static_cast<unsigned long long>(counts.continued), static_cast<unsigned long long>(counts.plans), static_cast<unsigned long long>(counts.taken), static_cast<unsigned long long>(counts.takenContinued), per(counts.frontNs, counts.taken), per(counts.walkNs, counts.plans), per(counts.backNs, counts.committed), static_cast<unsigned long long>(counts.committed), static_cast<unsigned long long>(counts.restarted), static_cast<unsigned long long>(counts.taught), static_cast<unsigned long long>(counts.retries), static_cast<unsigned long long>(counts.digests), static_cast<unsigned long long>(counts.verified), static_cast<unsigned long long>(counts.mismatches));
    std::string text = head;
    for (std::size_t index = 0; index < counts.left.size(); ++index) {
        if (counts.left[index] != 0) text += " " + std::string(FrontLeftName(static_cast<FrontLeft>(index))) + " " + std::to_string(counts.left[index]) + ";";
    }
    text += " front sections (us per taken draw): key";
    for (std::size_t index = 0; index < 8; ++index) {
        char item[48];
        static constexpr std::array<const char*, 8> names{"", "words", "walk", "variant", "views and vertex", "queue checks", "patch", "enqueue"};
        std::snprintf(item, sizeof(item), "%s %.2f", names[index], per(counters().sections[index].load(std::memory_order_relaxed), counts.taken));
        text += item;
        if (index + 1 < 8) text += ",";
    }
    text += "; not taught:";
    for (std::size_t index = 1; index < counts.refused.size(); ++index) {
        if (counts.refused[index] != 0) text += " " + std::string(TeachRefusalName(static_cast<TeachRefusal>(index))) + " " + std::to_string(counts.refused[index]) + ";";
    }
    if (teachBounds()) {
        char bounds[160];
        std::snprintf(bounds, sizeof(bounds), " teach bounds: %llu asks held back by the filter, %llu variants and plan sets released;", static_cast<unsigned long long>(counts.filtered), static_cast<unsigned long long>(counts.released));
        text += bounds;
    }
    if (counts.flatTaken != 0) {
        char flat[256];
        std::snprintf(flat, sizeof(flat), " flat draws: %llu taken (%llu on the variant of the draw before) at %.2f us at the front (%llu timed), %llu ring bytes, %llu vertex decodes answered by the memo, %llu walked into the driver zone", static_cast<unsigned long long>(counts.flatTaken), static_cast<unsigned long long>(counts.flatSame), per(counts.flatNs, counts.flatTimed), static_cast<unsigned long long>(counts.flatTimed), static_cast<unsigned long long>(counts.flatBytes), static_cast<unsigned long long>(counts.flatVertexKnown), static_cast<unsigned long long>(counts.flatZoned));
        text += flat;
        const auto postTimed = held.flatPostTimed.load(std::memory_order_relaxed);
        char post[192];
        std::snprintf(post, sizeof(post), ", posted to the committer at %.2f us (%llu timed, %llu of them while the finisher was busy)", per(held.flatPostNs.load(std::memory_order_relaxed), postTimed), static_cast<unsigned long long>(postTimed), static_cast<unsigned long long>(held.flatPostBusy.load(std::memory_order_relaxed)));
        text += post;
        if (const auto* const ops = DriverDetail::DrawPipeline::Ops()) {
            const auto counted = ops->Counted();
            char line[320];
            std::snprintf(line, sizeof(line), "; ops: %llu posted, %llu retired, %llu elements of the old path ordered against them (%llu passed), %llu waits for room, %llu wakes of the committer, %llu records wider than a flat one", static_cast<unsigned long long>(counted.posted), static_cast<unsigned long long>(counted.retired), static_cast<unsigned long long>(counted.gates), static_cast<unsigned long long>(counted.passed), static_cast<unsigned long long>(counted.roomWaits), static_cast<unsigned long long>(counted.wakes), static_cast<unsigned long long>(counted.wide));
            text += line;
        }
    }
    return text;
}

void Front::NoteFlatPosted(bool busy, std::uint64_t nanoseconds) {
    auto& held = counters();
    add(held.flatPostNs, nanoseconds);
    add(held.flatPostTimed);
    if (busy) add(held.flatPostBusy);
}

void Front::NoteEnqueued(std::uint64_t nanoseconds) {
    add(counters().sections[7], nanoseconds);
}

void Front::NoteCommitted(bool restarted, std::uint64_t nanoseconds) {
    add(counters().committed);
    add(counters().backNs, nanoseconds);
    if (restarted) add(counters().restarted);
}

StateKey Front::KeyOf(const QueueState& queue, const Pm4::DrawParameters& parameters, std::uint32_t opcode, std::uint64_t device) {
    StateKey key;
    key.shader = queue.shaderDigest;
    key.context = queue.contextDigest;
    key.userConfig = queue.userConfigDigest;
    key.programs[0] = EveryProgram;
    key.indexType = queue.indexType | (parameters.indexed ? 0x100u : 0u) | (opcode << 16u);
    key.topology = parameters.indexSize;
    key.device = device;
    return key;
}

void Front::VerifyDigests(const QueueState& queue) {
    const std::lock_guard lock(state->mutex);
    auto& registers = state->registers;
    const auto check = [&](const Registers& bank, Bank kind, const BankDigest& held, const char* name) {
        registers.clear();
        bank.ForEach([&](std::uint32_t offset, std::uint32_t value) { registers.emplace_back(offset, value); });
        const auto full = DigestOf(registers, kind);
        add(counters().digests);
        if (full == held) return;
        add(counters().mismatches);
        char message[256];
        std::snprintf(message, sizeof(message), "APS5_VERIFY_CORE_RUN: the incremental digest of the %s register bank (%016llx %016llx) differs from the digest of its %zu registers (%016llx %016llx)", name, static_cast<unsigned long long>(held.low), static_cast<unsigned long long>(held.high), registers.size(), static_cast<unsigned long long>(full.low), static_cast<unsigned long long>(full.high));
        throw std::runtime_error(message);
    };
    check(queue.shader, Bank::Shader, queue.shaderDigest, "shader");
    check(queue.context, Bank::Context, queue.contextDigest, "context");
    check(queue.userConfig, Bank::UserConfig, queue.userConfigDigest, "user configuration");
}

void Front::Patch(const RunPlanData& plan, std::size_t stage, const VariantStage& taught, const StageWalk& walk, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderVertexStageInfo* vertex, ShaderRecompiler::RecompileResult& out) {
    const auto& base = *taught.base;
    const auto& planStage = plan.stages[stage];
    if (taught.flatBinding != NoBinding) {
        auto& words = out.bindings[taught.flatBinding].guestDescriptor;
        const auto count = std::min<std::size_t>({words.size(), walk.flat.size(), base.blockLayout.words});
        std::copy_n(walk.flat.begin(), count, words.begin());
    }
    if (taught.buffersBinding != NoBinding) {
        auto& words = out.bindings[taught.buffersBinding].guestDescriptor;
        for (std::size_t element = 0; element < planStage.bufferResult.size() && (element + 1u) * 4u <= words.size(); ++element) {
            const auto result = planStage.bufferResult[element];
            if (result < 0) continue;
            const auto& raw = walk.results[static_cast<std::size_t>(result)].dwords;
            for (std::uint32_t dword = 0; dword < 4; ++dword) {
                if ((planStage.bufferMask[element] >> dword & 1u) != 0u) words[element * 4u + dword] = raw[dword];
            }
        }
    }
    const auto& forwarded = base.shaderDataUserWords;
    for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
        if (forwarded[slot] >= userData.size() || (base.blockModule && slot - base.blockAddressDword < 2u)) continue;
        const auto value = userData[forwarded[slot]];
        const auto at = static_cast<std::size_t>(slot) * sizeof(std::uint32_t);
        if (at + sizeof(std::uint32_t) <= out.pushConstants.size()) std::memcpy(out.pushConstants.data() + at, &value, sizeof(value));
        for (auto& binding : out.bindings) {
            if (binding.role == Role::ShaderData && slot < binding.guestDescriptor.size()) binding.guestDescriptor[slot] = value;
        }
    }
    if (vertex != nullptr) {
        for (auto& attribute : out.vertexAttributes) {
            if (attribute.location < vertex->resourcesNum) attribute.resource = vertex->resources[attribute.location];
        }
    }
}

FrontOutcome Front::Walk(const QueueState& queue, const DriverDetail::ShaderRegistry& shaders, const Pm4::DrawParameters& parameters, std::uint32_t opcode, std::uint64_t device, bool verifyOnly) {
    using DriverDetail::DrawFinisher;
    using DriverDetail::DrawPipeline;
    const std::lock_guard lock(state->mutex);
    auto& held = counters();
    FrontOutcome outcome;
    const auto started = std::chrono::steady_clock::now();
    auto lapAt = started;
    const auto lap = [&](std::size_t section) {
        const auto now = std::chrono::steady_clock::now();
        add(held.sections[section], static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - lapAt).count()));
        lapAt = now;
    };
    const bool continued = queue.runMask.Continues();
    add(held.draws);
    if (continued) add(held.continued);
    std::uint64_t leftSalt = 0;
    const auto left = [&](FrontLeft reason, bool teach) {
        add(held.left[static_cast<std::size_t>(reason)]);
        if (teach) teach = state->Wanted(outcome.key.Hash(), leftSalt);
        outcome.teach = teach;
        outcome.left = reason;
        outcome.candidate = nullptr;
        return outcome;
    };
    if ((++state->walked & 0xfffu) == 0) state->teaching.clear();
    if (state->inboxed.exchange(false, std::memory_order_acq_rel)) state->Arrive();
    outcome.key = KeyOf(queue, parameters, opcode, device);
    const auto* plan = state->plans.Find(outcome.key);
    if (plan == nullptr) return left(FrontLeft::NoPlan, true);
    const auto data = std::static_pointer_cast<const RunPlanData>(plan->resources);
    if (!DriverDetail::DecodeProgramsCurrent(*data->decode, shaders)) {
        state->plans.Forget(EveryProgram);
        state->Forgotten();
        return left(FrontLeft::Programs, true);
    }
    add(held.plans);
    lap(0);
    std::shared_ptr<CoreDraw> record;
    {
        auto& spare = pool();
        const std::lock_guard poolLock(spare.mutex);
        if (!spare.spare.empty()) {
            record = std::move(spare.spare.back());
            spare.spare.pop_back();
        }
    }
    if (record == nullptr) record = std::make_shared<CoreDraw>();
    const auto giveBack = [&](FrontLeft reason, bool teach) {
        Release(std::move(record));
        return left(reason, teach);
    };
    auto& ranges = state->ranges;
    auto& pipeline = DrawPipeline::Queue0();
    std::shared_ptr<FrontVariant> variant;
    FrontLeft leftReason = FrontLeft::Entry;
    bool leftTeach = false;
    const auto attempt = [&](bool watched) -> int {
    ranges.clear();
    DrawPipeline::Watch watch;
    if (watched) watch = pipeline.WatchWrites();
    const auto retired = pipeline.Retired();
    const auto publish = Graphics::Recorder::PublishGeneration();
    const auto pendingSerial = Graphics::StorageTexture::PendingSerial();
    const auto forget = GuestMemory::ForgetSerial();
    if ((forget & 1u) != 0) { leftReason = FrontLeft::Moved; leftTeach = false; return 1; }
    const auto& programs = data->decode->programs;
    for (std::uint32_t stage = 0; stage < data->stageCount; ++stage) {
        const auto& program = programs[stage];
        const auto count = userWordCount(queue, program);
        if (count != data->stages[stage].userWords) { leftReason = FrontLeft::UserWords; leftTeach = false; return 1; }
        auto& words = record->userData[stage];
        words.resize(count);
        for (std::uint32_t index = 0; index < count; ++index) words[index] = DriverDetail::readUserData(queue.shader, program.userDataBase + index);
    }
    lap(1);
    const auto walkStarted = std::chrono::steady_clock::now();
    std::array<std::uint64_t, 2> digest{0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull};
    for (std::uint32_t stage = 0; stage < data->stageCount; ++stage) {
        const GuestMemory::ReadSiteScope readSite(GuestMemory::ReadSite::DrawCache);
        if (!walkStage(data->stages[stage], record->userData[stage], programs[stage].binary.codeAddress, ranges, record->walks[stage])) {
            add(held.walkNs, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - walkStarted).count()));
            { leftReason = FrontLeft::Walk; leftTeach = false; return 1; }
        }
        mixWalk(digest, data->stages[stage], record->walks[stage]);
    }
    add(held.walkNs, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - walkStarted).count()));
    lap(2);
    leftSalt = digest[0] ^ std::rotl(digest[1], 23);
    ImageSetValue found;
    if (state->variants.Find(ImageSetKey{data->id, digest}, Proof{}, found) != Hit::Fresh) { leftReason = FrontLeft::NoVariant; leftTeach = true; return 1; }
    variant = std::static_pointer_cast<FrontVariant>(found.owner);
    if (variant->plan != data) { leftReason = FrontLeft::NoVariant; leftTeach = true; return 1; }
    for (std::uint32_t stage = 0; stage < data->stageCount; ++stage) {
        if (!sameKey(data->stages[stage], variant->stages[stage].taught, record->walks[stage])) { leftReason = FrontLeft::NoVariant; leftTeach = true; return 1; }
    }
    lap(3);
    if (variant->stale.load(std::memory_order_acquire)) { leftReason = FrontLeft::Stale; leftTeach = true; return 1; }
    for (std::uint32_t stage = 0; stage < data->stageCount; ++stage) {
        const auto& taught = variant->stages[stage];
        if (ShaderRecompiler::BlockVariantKey(taught.base->blockLayout, record->walks[stage].flat, record->userData[stage]) != taught.variantKey) { leftReason = FrontLeft::VariantKey; leftTeach = false; return 1; }
        if (taught.registeredViews != nullptr && !DriverDetail::CaptureRegisteredViewsCurrent(*taught.registeredViews)) { leftReason = FrontLeft::Views; leftTeach = true; return 1; }
        if (!taught.vertexInfo) continue;
        if (!decodeVertex(programs[stage], record->userData[stage], *taught.vertexInfo, state->decodeReads, record->vertex[stage])) { leftReason = FrontLeft::Vertex; leftTeach = false; return 1; }
        if (state->decodeReads.empty() || !DriverDetail::Driver::CoreVertexInputsMatch(*taught.vertexInfo, record->vertex[stage])) { leftReason = FrontLeft::Vertex; leftTeach = false; return 1; }
        for (const auto& attribute : taught.base->vertexAttributes) {
            if (attribute.location >= record->vertex[stage].resourcesNum) { leftReason = FrontLeft::Vertex; leftTeach = false; return 1; }
        }
        for (const auto& read : state->decodeReads) ranges.emplace_back(read.address, read.address + read.size);
    }
    lap(4);
    if (DrawFinisher::Busy() && DrawFinisher::Get().PendingOverlapsAny(ranges)) { leftReason = FrontLeft::Finisher; leftTeach = false; return 1; }
    if (!watched) {
        const auto written = pipeline.WrittenWhileQueued(retired, ranges);
        if (!written.has_value()) return 2;
        if (*written) { leftReason = FrontLeft::Queued; leftTeach = false; return 1; }
    } else if (!(watch.items == 0 && !pipeline.Busy() && pipeline.Retired() == watch.retired) && pipeline.WrittenSince(watch, ranges)) {
        leftReason = FrontLeft::Queued;
        leftTeach = false;
        return 1;
    }
    DriverDetail::PendingView pending;
    pending.Load();
    for (const auto& [begin, end] : ranges) {
        if (pending.Overlaps(begin, static_cast<std::size_t>(end - begin))) { leftReason = FrontLeft::Pending; leftTeach = false; return 1; }
    }
    if (Graphics::StorageTexture::AnyPendingOverlaps(ranges) || Graphics::AnyShadowedOverlaps(ranges)) { leftReason = FrontLeft::Pending; leftTeach = false; return 1; }
    if (Graphics::Recorder::PublishGeneration() != publish || Graphics::StorageTexture::PendingSerial() != pendingSerial || GuestMemory::ForgetSerial() != forget) { leftReason = FrontLeft::Moved; leftTeach = false; return 1; }
    lap(5);
    return 0;
    };
    auto verdict = attempt(false);
    if (verdict == 2) {
        add(held.retries);
        verdict = attempt(true);
    }
    if (verdict != 0) return giveBack(leftReason, leftTeach);
    for (std::uint32_t stage = 0; stage < data->stageCount; ++stage) {
        const auto& taught = variant->stages[stage];
        Patch(*data, stage, taught, record->walks[stage], record->userData[stage], taught.vertexInfo ? &record->vertex[stage] : nullptr, variant->frontResults[stage]);
    }
    lap(6);
    record->variant = std::move(variant);
    record->parameters = parameters;
    record->opcode = opcode;
    outcome.candidate = std::move(record);
    outcome.taken = !verifyOnly;
    if (outcome.taken) {
        add(held.taken);
        if (continued) add(held.takenContinued);
        add(held.frontNs, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
    }
    return outcome;
}

FrontOutcome Front::Flat(const FlatInputs& inputs) {
    using DriverDetail::DrawFinisher;
    using DriverDetail::DrawPipeline;
    auto& held = counters();
    auto& flat = state->flat;
    FrontOutcome outcome;
    outcome.looked = true;
    const auto& queue = *inputs.queue;
    const auto& parameters = *inputs.parameters;
    const auto opcode = (inputs.packet[0] >> 8u) & 0xffu;
    const bool timed = flat.clock.Tick();
    const auto started = flat.clock.Now();
    const bool continued = queue.runMask.Continues();
    bump(held.draws);
    if (continued) bump(held.continued);
    std::uint64_t leftSalt = 0;
    const auto left = [&](FrontLeft reason, bool teach) {
        bump(held.left[static_cast<std::size_t>(reason)]);
        if (teach) teach = state->Wanted(outcome.key.Hash(), leftSalt);
        outcome.teach = teach;
        outcome.left = reason;
        outcome.candidate = nullptr;
        return outcome;
    };
    if ((++state->walked & 0xfffu) == 0) state->teaching.clear();
    if (state->inboxed.load(std::memory_order_acquire) && state->inboxed.exchange(false, std::memory_order_acq_rel)) state->Arrive();
    outcome.key = KeyOf(queue, parameters, opcode, inputs.device);
    if (flat.plan == nullptr || !(outcome.key == flat.key)) {
        const auto* plan = state->plans.Find(outcome.key);
        if (plan == nullptr) {
            flat.plan = nullptr;
            flat.variant = nullptr;
            return left(FrontLeft::NoPlan, true);
        }
        auto found = std::static_pointer_cast<const RunPlanData>(plan->resources);
        if (found != flat.plan) {
            flat.plan = std::move(found);
            flat.variant = nullptr;
        }
        flat.key = outcome.key;
    }
    const auto& data = *flat.plan;
    if (!DriverDetail::DecodeProgramsCurrent(*data.decode, *inputs.shaders)) {
        state->plans.Forget(EveryProgram);
        state->Forgotten();
        flat.plan = nullptr;
        flat.variant = nullptr;
        return left(FrontLeft::Programs, true);
    }
    if (!parameters.indexed && (parameters.indexCount == 0 || parameters.instanceCount == 0)) return left(FrontLeft::Nothing, false);
    if (data.precheck == 0) {
        data.precheck = inputs.plain(inputs.context, queue, parameters) ? 1 : 2;
    } else if (inputs.verifyOnly && (inputs.plain(inputs.context, queue, parameters) ? 1 : 2) != data.precheck) {
        add(held.mismatches);
        throw std::runtime_error("APS5_VERIFY_CORE_RUN: the precheck of a draw differs from the precheck kept by the plan of its state key");
    }
    if (data.precheck == 2) return left(FrontLeft::Precheck, false);
    bump(held.plans);
    const auto& programs = data.decode->programs;
    std::array<std::span<const std::uint32_t>, 4> userData;
    std::uint32_t userWords = 0;
    for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
        const auto& program = programs[stage];
        const auto count = userWordCount(queue, program);
        if (count != data.stages[stage].userWords || userWords + count > flat.userData.size()) return left(FrontLeft::UserWords, false);
        auto* words = flat.userData.data() + userWords;
        for (std::uint32_t index = 0; index < count; ++index) words[index] = DriverDetail::readUserData(queue.shader, program.userDataBase + index);
        userData[stage] = std::span<const std::uint32_t>(words, count);
        userWords += count;
    }
    auto& pipeline = DrawPipeline::Queue0();
    auto& reader = flat.reader;
    static const bool zoneWalk = std::getenv("APS5_CORE_TAPE_ZONE") != nullptr;
    const bool zoned = zoneWalk && data.zoned == 1;
    FrontVariant* variant = nullptr;
    std::shared_ptr<FrontVariant> fetched;
    std::array<std::uint64_t, 2> digest{};
    bool same = false;
    for (int pass = 0;; ++pass) {
        const bool watched = pass != 0;
        reader.count = 0;
        reader.readCount = 0;
        reader.overflow = false;
        DrawPipeline::Watch watch;
        if (watched) watch = pipeline.WatchWrites();
        const auto retired = pipeline.Retired();
        const auto publish = Graphics::Recorder::PublishGeneration();
        const auto pendingSerial = Graphics::StorageTexture::PendingSerial();
        const auto forget = GuestMemory::ForgetSerial();
        if ((forget & 1u) != 0) return left(FrontLeft::Moved, false);
        digest = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull};
        {
            const GuestMemory::ReadSiteScope readSite(GuestMemory::ReadSite::DrawCache);
            for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
                if (zoned) {
                    const auto& planStage = data.stages[stage];
                    auto& walk = flat.walks[stage];
                    walk.flat.resize(planStage.plan->srtReads.size());
                    walk.zone.resize(planStage.tape->Driver().words);
                    if (planStage.tape->RunInto(*planStage.plan, userData[stage], programs[stage].binary.codeAddress, &flatReadWords, &reader, walk.flat, walk.zone) != ShaderRecompiler::BlockWalk::Done) return left(reader.overflow ? FrontLeft::Ranges : FrontLeft::Walk, false);
                    mixZone(digest, planStage, walk.zone);
                    continue;
                }
                ShaderRecompiler::SrtRuntime runtime;
                runtime.userData = userData[stage];
                runtime.shaderBase = programs[stage].binary.codeAddress;
                runtime.userContext = &reader;
                runtime.readMemory = &flatReadWord;
                runtime.readSpecializationMemory = &flatReadWord;
                auto& walk = flat.walks[stage];
                if (!data.stages[stage].tape->Run(*data.stages[stage].plan, runtime, walk.results, walk.flat, walk.active)) return left(reader.overflow ? FrontLeft::Ranges : FrontLeft::Walk, false);
                mixWalk(digest, data.stages[stage], walk);
            }
        }
        leftSalt = digest[0] ^ std::rotl(digest[1], 23);
        same = flat.variant != nullptr && digest == flat.digest;
        if (same) {
            variant = flat.variant.get();
        } else {
            ImageSetValue found;
            if ((zoned ? flat.zoned : state->variants).Find(ImageSetKey{data.id, digest}, Proof{}, found) != Hit::Fresh) return left(FrontLeft::NoVariant, true);
            fetched = std::static_pointer_cast<FrontVariant>(found.owner);
            variant = fetched.get();
            if (variant->plan.get() != &data || (zoned && !variant->zoned)) return left(FrontLeft::NoVariant, true);
        }
        for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
            if (zoned ? !sameZone(data.stages[stage], variant->stages[stage].taught.zone, flat.walks[stage].zone) : !sameKey(data.stages[stage], variant->stages[stage].taught, flat.walks[stage])) return left(FrontLeft::NoVariant, true);
        }
        if (variant->stale.load(std::memory_order_acquire)) return left(FrontLeft::Stale, true);
        if (parameters.indexed && parameters.indexSize != 0 && parameters.indexCount != 0 && !reader.Reads(parameters.indexAddress, static_cast<std::uint64_t>(parameters.indexCount) * parameters.indexSize)) return left(FrontLeft::Ranges, false);
        for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
            const auto& taught = variant->stages[stage];
            const auto& planStage = data.stages[stage];
            const auto& walk = flat.walks[stage];
            if (walk.flat.size() < planStage.flatWords) return left(FrontLeft::Walk, false);
            if (ShaderRecompiler::BlockVariantKey(taught.base->blockLayout, walk.flat, userData[stage]) != taught.variantKey) return left(FrontLeft::VariantKey, false);
            for (std::size_t element = 0; element < planStage.bufferResult.size(); ++element) {
                const auto slot = planStage.bufferResult[element];
                if (slot < 0) continue;
                const auto* raw = zoned ? walk.zone.data() + planStage.tape->Driver().sources[static_cast<std::size_t>(slot)].word : walk.results[static_cast<std::size_t>(slot)].dwords.data();
                const ShaderRecompiler::ShaderBufferResource descriptor{{raw[0], raw[1], (planStage.bufferMask[element] & 4u) != 0u ? raw[2] : 0u, 0u}};
                if (!reader.Reads(descriptor.Base48(), descriptor.GetSize())) return left(FrontLeft::Ranges, false);
            }
            if (taught.registeredViews != nullptr && !DriverDetail::CaptureRegisteredViewsCurrent(*taught.registeredViews)) return left(FrontLeft::Views, true);
            if (!taught.vertexInfo) continue;
            auto& fresh = flat.vertex[stage];
            auto& memo = flat.vertexMemo[((reinterpret_cast<std::uintptr_t>(variant) >> 6u) ^ stage) % flat.vertexMemo.size()];
            const std::vector<Graphics::DecodeRead>* vertexReads = &flat.decodeReads;
            bool known = memo.variant == variant->serial && memo.stage == stage && memo.userData.size() == userData[stage].size() && std::equal(memo.userData.begin(), memo.userData.end(), userData[stage].begin());
            for (std::size_t index = 0; known && index < memo.reads.size(); ++index) {
                const auto& read = memo.reads[index];
                std::array<std::byte, 16> now{};
                known = read.size <= now.size() && ReadGuestBytes(read.address, std::span<std::byte>(now.data(), read.size)) && std::memcmp(now.data(), read.data.data(), read.size) == 0;
            }
            if (known) {
                fresh = memo.fresh;
                vertexReads = &memo.reads;
                bump(held.flatVertexKnown);
            } else {
                memo.variant = 0;
                if (!decodeVertex(programs[stage], userData[stage], *taught.vertexInfo, flat.decodeReads, fresh)) return left(FrontLeft::Vertex, false);
                if (flat.decodeReads.empty() || !DriverDetail::Driver::CoreVertexInputsMatch(*taught.vertexInfo, fresh)) return left(FrontLeft::Vertex, false);
                memo.stage = stage;
                memo.userData.assign(userData[stage].begin(), userData[stage].end());
                memo.reads = flat.decodeReads;
                memo.fresh = fresh;
                memo.variant = variant->serial;
            }
            for (const auto& attribute : taught.base->vertexAttributes) {
                if (attribute.location >= fresh.resourcesNum) return left(FrontLeft::Vertex, false);
                auto bound = attribute;
                bound.resource = fresh.resources[attribute.location];
                if (Graphics::NullVertexDescriptor(bound)) continue;
                std::uint64_t bytes = 0;
                try {
                    if (Graphics::VertexFetchOutOfRange(bound)) continue;
                    bytes = Graphics::VertexBufferExtent(bound);
                } catch (const std::runtime_error&) {
                    continue;
                }
                if (!reader.Reads(bound.resource.fields[0] | (static_cast<std::uint64_t>(bound.resource.fields[1] & 0xffffu) << 32u), bytes)) return left(FrontLeft::Ranges, false);
            }
            for (const auto& read : *vertexReads) {
                if (!reader.Note(read.address, read.address + read.size)) return left(FrontLeft::Ranges, false);
            }
        }
        const std::span<const Range> ranges(reader.ranges.data(), reader.count);
        if (DrawFinisher::Busy() && !DrawFinisher::Overlap() && DrawFinisher::Get().PendingOverlapsAny(ranges)) return left(FrontLeft::Finisher, false);
        if (!watched) {
            const auto queued = pipeline.WrittenWhileQueued(retired, ranges);
            if (!queued.has_value()) {
                add(held.retries);
                continue;
            }
            if (*queued) return left(FrontLeft::Queued, false);
        } else if (!(watch.items == 0 && !pipeline.Busy() && pipeline.Retired() == watch.retired) && pipeline.WrittenSince(watch, ranges)) {
            return left(FrontLeft::Queued, false);
        }
        DriverDetail::PendingView pending;
        pending.Load();
        for (const auto& [begin, end] : ranges) {
            if (pending.Overlaps(begin, static_cast<std::size_t>(end - begin))) return left(FrontLeft::Pending, false);
        }
        if (Graphics::StorageTexture::AnyPendingOverlaps(ranges) || Graphics::AnyShadowedOverlaps(ranges)) return left(FrontLeft::Pending, false);
        if (Graphics::Recorder::PublishGeneration() != publish || Graphics::StorageTexture::PendingSerial() != pendingSerial || GuestMemory::ForgetSerial() != forget) return left(FrontLeft::Moved, false);
        break;
    }
    if (!same) {
        flat.variant = std::move(fetched);
        flat.digest = digest;
    }
    outcome.variant = variant;
    outcome.reads = std::span<const Range>(reader.reads.data(), reader.readCount);
    if (inputs.verifyOnly) {
        if (zoned) {
            std::array<std::uint64_t, 2> classic{0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull};
            for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
                ShaderRecompiler::SrtRuntime runtime;
                runtime.userData = userData[stage];
                runtime.shaderBase = programs[stage].binary.codeAddress;
                runtime.userContext = &reader;
                runtime.readMemory = &flatReadWord;
                runtime.readSpecializationMemory = &flatReadWord;
                auto& walk = flat.walks[stage];
                if (!data.stages[stage].tape->Run(*data.stages[stage].plan, runtime, walk.results, walk.flat, walk.active)) return left(FrontLeft::Walk, false);
                mixWalk(classic, data.stages[stage], walk);
            }
            ImageSetValue found;
            if (state->variants.Find(ImageSetKey{data.id, classic}, Proof{}, found) == Hit::Fresh && found.owner.get() != variant && !static_cast<FrontVariant*>(found.owner.get())->stale.load(std::memory_order_acquire)) {
                add(held.mismatches);
                throw std::runtime_error("APS5_VERIFY_CORE_RUN: the variant chosen from the driver zone of the tape differs from the variant chosen from the walk results");
            }
        }
        auto record = std::make_shared<CoreDraw>();
        record->variant = flat.variant;
        record->parameters = parameters;
        record->opcode = opcode;
        for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
            record->userData[stage].assign(userData[stage].begin(), userData[stage].end());
            record->walks[stage] = flat.walks[stage];
            if (variant->stages[stage].vertexInfo) record->vertex[stage] = flat.vertex[stage];
        }
        outcome.candidate = std::move(record);
        return outcome;
    }
    const auto words = flatWords(data, *variant);
    const auto slice = flat.ring.Take(static_cast<std::size_t>(words) * sizeof(std::uint32_t), 16);
    if (slice.bytes == nullptr) return left(FrontLeft::Ring, false);
    auto* const first = reinterpret_cast<std::uint32_t*>(slice.bytes);
    auto* out = first + OpWords;
    Op op = MakeOp(OpKind::Draw);
    op.stages = static_cast<std::uint8_t>(data.stageCount);
    op.epoch = static_cast<std::uint32_t>(inputs.epoch);
    op.order = inputs.epoch;
    for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
        const auto& planStage = data.stages[stage];
        const auto& walk = flat.walks[stage];
        op.draw.section[stage] = static_cast<std::uint32_t>((out - first) * sizeof(std::uint32_t));
        std::copy_n(walk.flat.begin(), planStage.flatWords, out);
        out += planStage.flatWords;
        for (std::size_t element = 0; element < planStage.bufferResult.size(); ++element, out += 4) {
            const auto slot = planStage.bufferResult[element];
            const auto* raw = slot < 0 ? nullptr : zoned ? walk.zone.data() + planStage.tape->Driver().sources[static_cast<std::size_t>(slot)].word : walk.results[static_cast<std::size_t>(slot)].dwords.data();
            for (std::uint32_t dword = 0; dword < 4; ++dword) out[dword] = raw == nullptr ? 0u : raw[dword];
        }
        std::copy(userData[stage].begin(), userData[stage].end(), out);
        out += userData[stage].size();
    }
    for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
        const auto& taught = variant->stages[stage];
        if (!taught.vertexInfo) continue;
        for (const auto& attribute : taught.base->vertexAttributes) {
            for (std::uint32_t field = 0; field < 4; ++field) out[field] = flat.vertex[stage].resources[attribute.location].fields[field];
            out += 4;
        }
    }
    op.draw.variant = variant;
    variant->pending.fetch_add(1, std::memory_order_relaxed);
    op.draw.slice = slice;
    op.draw.indexAddress = parameters.indexAddress;
    op.draw.indexBytes = parameters.indexSize;
    op.draw.indexCount = parameters.indexCount;
    op.draw.firstIndex = parameters.flags;
    op.draw.baseVertex = static_cast<std::int32_t>(parameters.firstVertex);
    op.draw.instances = parameters.instanceCount;
    op.draw.firstInstance = parameters.firstInstance;
    op.draw.extra = 0;
    op.draw.extraWords = static_cast<std::uint32_t>(out - first);
    op.draw.opcode = opcode;
    op.draw.drawFlags = (parameters.indexed ? DrawIndexed : 0u) | (parameters.indexSize == 4u ? DrawIndex32 : parameters.indexSize == 1u ? DrawIndex8 : 0u);
    std::memcpy(first, &op, sizeof(op));
    outcome.op = slice.bytes;
    outcome.taken = true;
    bump(held.taken);
    bump(held.flatTaken);
    if (continued) bump(held.takenContinued);
    if (same) bump(held.flatSame);
    if (zoned) bump(held.flatZoned);
    bump(held.flatBytes, static_cast<std::uint64_t>(words) * sizeof(std::uint32_t));
    if (timed) {
        bump(held.flatNs, steadyNanoseconds(nullptr) - started);
        bump(held.flatTimed);
    }
    return outcome;
}

void Front::FlatPosted(const std::byte* at) {
    Op op = MakeOp(OpKind::None);
    std::memcpy(&op, at, sizeof(op));
    state->flat.ring.Posted(op.draw.slice);
}

FlatDraw Front::FlatRead(const std::byte* at) {
    FlatDraw draw{MakeOp(OpKind::None)};
    std::memcpy(&draw.op, at, sizeof(Op));
    draw.variant = static_cast<FrontVariant*>(const_cast<void*>(draw.op.draw.variant));
    const auto& data = *draw.variant->plan;
    const auto* begin = reinterpret_cast<const std::uint32_t*>(at);
    const auto* words = begin + OpWords;
    for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
        const auto& planStage = data.stages[stage];
        draw.section[stage] = std::span<const std::uint32_t>(words, planStage.flatWords);
        words += planStage.flatWords;
        draw.buffers[stage] = std::span<const std::uint32_t>(words, planStage.bufferResult.size() * 4u);
        words += planStage.bufferResult.size() * 4u;
        draw.userData[stage] = std::span<const std::uint32_t>(words, planStage.userWords);
        words += planStage.userWords;
    }
    draw.vertex = std::span<const std::uint32_t>(words, begin + draw.op.draw.extraWords);
    return draw;
}

void Front::FlatEmitted(const FlatDraw& draw) {
    state->flat.ring.Emitted(draw.op.draw.slice, 0);
    draw.variant->pending.fetch_sub(1, std::memory_order_release);
}

void Front::FlatPatch(const FlatDraw& draw, std::size_t stage, ShaderRecompiler::RecompileResult& out) {
    const auto& variant = *draw.variant;
    const auto& data = *variant.plan;
    const auto& taught = variant.stages[stage];
    const auto& base = *taught.base;
    const auto& planStage = data.stages[stage];
    const auto section = draw.section[stage];
    const auto buffers = draw.buffers[stage];
    const auto userData = draw.userData[stage];
    if (taught.flatBinding != NoBinding) {
        auto& words = out.bindings[taught.flatBinding].guestDescriptor;
        const auto count = std::min<std::size_t>({words.size(), section.size(), base.blockLayout.words});
        std::copy_n(section.begin(), count, words.begin());
    }
    if (taught.buffersBinding != NoBinding) {
        auto& words = out.bindings[taught.buffersBinding].guestDescriptor;
        for (std::size_t element = 0; element < planStage.bufferResult.size() && (element + 1u) * 4u <= words.size(); ++element) {
            if (planStage.bufferResult[element] < 0) continue;
            for (std::uint32_t dword = 0; dword < 4; ++dword) {
                if ((planStage.bufferMask[element] >> dword & 1u) != 0u) words[element * 4u + dword] = buffers[element * 4u + dword];
            }
        }
    }
    const auto& forwarded = base.shaderDataUserWords;
    for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
        if (forwarded[slot] >= userData.size() || (base.blockModule && slot - base.blockAddressDword < 2u)) continue;
        const auto value = userData[forwarded[slot]];
        const auto at = static_cast<std::size_t>(slot) * sizeof(std::uint32_t);
        if (at + sizeof(std::uint32_t) <= out.pushConstants.size()) std::memcpy(out.pushConstants.data() + at, &value, sizeof(value));
        for (auto& binding : out.bindings) {
            if (binding.role == Role::ShaderData && slot < binding.guestDescriptor.size()) binding.guestDescriptor[slot] = value;
        }
    }
    if (!taught.vertexInfo) return;
    std::size_t at = 0;
    for (std::size_t before = 0; before < stage; ++before) {
        if (variant.stages[before].vertexInfo) at += variant.stages[before].base->vertexAttributes.size() * 4u;
    }
    for (auto& attribute : out.vertexAttributes) {
        if (at + 4u > draw.vertex.size()) break;
        for (std::uint32_t field = 0; field < 4; ++field) attribute.resource.fields[field] = draw.vertex[at + field];
        at += 4u;
    }
}

void Front::Release(std::shared_ptr<CoreDraw> record) {
    if (record == nullptr) return;
    record->variant = nullptr;
    auto& spare = pool();
    const std::lock_guard lock(spare.mutex);
    if (spare.spare.size() < 4096) spare.spare.push_back(std::move(record));
}

void Front::NoteVerified(bool mismatch) {
    add(counters().verified);
    if (mismatch) add(counters().mismatches);
}

bool Front::TeachBounds() {
    return teachBounds();
}

bool Front::KeyDiagEnabled() {
    static const bool enabled = std::getenv("APS5_CORE_KEY_DIAG") != nullptr;
    return enabled;
}

void Front::KeyDiag(const QueueState& queue, std::uint64_t shape, bool noPlan) {
    static KeyDiagState held;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> now;
    const auto take = [&](const Registers& bank, Bank kind) {
        bank.ForEach([&](std::uint32_t offset, std::uint32_t value) {
            if (KeyRegister(kind, offset)) now.emplace_back((static_cast<std::uint32_t>(kind) << 24u) | offset, value);
        });
    };
    take(queue.shader, Bank::Shader);
    take(queue.context, Bank::Context);
    take(queue.userConfig, Bank::UserConfig);
    const std::lock_guard lock(held.mutex);
    if (held.last.size() >= 65536 && !held.last.contains(shape)) held.last.clear();
    const auto [at, inserted] = held.last.try_emplace(shape);
    if (noPlan && inserted) ++held.first;
    if (noPlan && !inserted) {
        ++held.compared;
        const auto& was = at->second;
        std::size_t a = 0, b = 0;
        std::uint64_t differing = 0;
        while (a < was.size() || b < now.size()) {
            std::uint32_t which = 0;
            if (b == now.size() || (a < was.size() && was[a].first < now[b].first)) which = was[a++].first;
            else if (a == was.size() || now[b].first < was[a].first) which = now[b++].first;
            else {
                const bool equal = was[a].second == now[b].second;
                which = was[a].first;
                ++a;
                ++b;
                if (equal) continue;
            }
            ++held.differed[which];
            ++differing;
        }
        held.registers += differing;
        if (differing == 0) ++held.equal;
    }
    at->second = std::move(now);
    const auto current = std::chrono::steady_clock::now();
    if (current - held.reported < std::chrono::seconds(10)) return;
    held.reported = current;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> top;
    for (const auto& [which, count] : held.differed) top.emplace_back(count, which);
    std::sort(top.begin(), top.end(), std::greater<>());
    char head[320];
    std::snprintf(head, sizeof(head), "[core-key] (cumulative) %llu draws without a plan compared with the draw before of their shape (%llu more were the first of their shape, %llu had equal key registers, %.2f registers differing per draw, %zu shapes); registers that differed:", static_cast<unsigned long long>(held.compared), static_cast<unsigned long long>(held.first), static_cast<unsigned long long>(held.equal), held.compared != 0 ? static_cast<double>(held.registers) / static_cast<double>(held.compared) : 0.0, held.last.size());
    std::string text = head;
    static constexpr std::array<const char*, 3> names{"shader", "context", "user-config"};
    for (std::size_t index = 0; index < top.size() && index < 16; ++index) {
        char item[96];
        std::snprintf(item, sizeof(item), " %s 0x%x %llu (%.1f%%);", names[(top[index].second >> 24u) % names.size()], top[index].second & 0xffffffu, static_cast<unsigned long long>(top[index].first), held.compared != 0 ? 100.0 * static_cast<double>(top[index].first) / static_cast<double>(held.compared) : 0.0);
        text += item;
    }
    std::fprintf(stderr, "%s%c", text.c_str(), 10);
}

bool Front::Refused(std::span<const DriverDetail::DrawProgram> programs, std::span<const Graphics::CompiledShader> stages) const {
    const auto pair = pairOf(programs, stages);
    if (pair == 0 || !state->refusals.Holds(pair)) return false;
    add(counters().refused[static_cast<std::size_t>(TeachRefusal::Known)]);
    return true;
}

void Front::Teach(const TeachInputs& inputs) {
    auto& held = counters();
    const bool bounded = teachBounds();
    std::vector<std::shared_ptr<void>> gone;
    if (bounded) {
        const std::lock_guard lock(state->inboxMutex);
        gone.swap(state->trash);
    }
    std::uint64_t pair = 0;
    const auto refuse = [&](TeachRefusal reason, bool lasting = true) {
        add(held.refused[static_cast<std::size_t>(reason)]);
        if (bounded && lasting && pair != 0 && (reason == TeachRefusal::Program || reason == TeachRefusal::Stage || reason == TeachRefusal::Plan || reason == TeachRefusal::Tape || reason == TeachRefusal::Writes)) state->refusals.Keep(pair);
    };
    using Stage = ShaderRecompiler::ShaderStage;
    const auto count = inputs.stages.size();
    if (inputs.recipe == nullptr || inputs.decode == nullptr || count == 0 || count > 4 || count != inputs.programs.size() || count != inputs.served.size()) return refuse(TeachRefusal::Path);
    pair = pairOf(inputs.programs, inputs.stages);
    std::array<std::shared_ptr<const ShaderRecompiler::IrResourcePlan>, 4> plans;
    std::array<const ShaderRecompiler::Detail::SrtTape*, 4> tapes{};
    for (std::size_t index = 0; index < count; ++index) {
        const auto& program = inputs.programs[index];
        const auto& shader = inputs.stages[index];
        if (shader.program == nullptr || (shader.stage != Stage::Vertex && shader.stage != Stage::Fragment) || program.binary.stage != shader.stage) return refuse(TeachRefusal::Path);
        if (program.merged || program.hiddenUserWords != 0 || program.tailContinuation != nullptr || (program.snapshot != nullptr && program.snapshot->userDataTailRegister && program.snapshot->header.empty())) return refuse(TeachRefusal::Program);
        const auto& result = *shader.program;
        if (!result.blockModule || !result.blockLayout.Eligible()) return refuse(TeachRefusal::Stage);
        for (const auto& binding : result.bindings) {
            if (binding.role == Role::BdaPagetable) return refuse(TeachRefusal::Writes);
            if (binding.role == Role::GuestBuffers && std::any_of(binding.bufferWritten.begin(), binding.bufferWritten.end(), [](bool written) { return written; })) return refuse(TeachRefusal::Writes);
            if (binding.role == Role::GuestImages && binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) return refuse(TeachRefusal::Writes);
        }
        if (!result.imageTableRanges.empty()) return refuse(TeachRefusal::Stage);
        const auto& served = inputs.served[index];
        std::shared_ptr<const ShaderRecompiler::IrResourcePlan> plan;
        if (served != nullptr) plan = served->plan != nullptr ? served->plan : served->donor != nullptr ? served->donor->plan : nullptr;
        if (plan == nullptr && index < inputs.plans.size()) plan = inputs.plans[index];
        if (plan == nullptr || !plan->srtPlanComplete) {
            const bool uncacheable = ((inputs.uncacheable >> index) & 1u) != 0u;
            notePlanRefusal(program.binary.codeAddress, static_cast<std::uint32_t>(shader.stage), uncacheable ? 3u : served == nullptr ? 0u : plan == nullptr ? 1u : 2u);
            return refuse(TeachRefusal::Plan, served != nullptr || uncacheable);
        }
        tapes[index] = ShaderRecompiler::Detail::SrtTape::For(*plan);
        if (tapes[index] == nullptr) return refuse(TeachRefusal::Tape);
        plans[index] = std::move(plan);
    }
    auto data = std::make_shared<RunPlanData>();
    data->decode = inputs.decode;
    data->stageCount = static_cast<std::uint32_t>(count);
    auto variant = std::make_shared<FrontVariant>();
    variant->serial = variantSerial().fetch_add(1, std::memory_order_relaxed) + 1;
    variant->digest = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull};
    std::vector<Range> ranges;
    std::vector<Graphics::DecodeRead> decodeReads;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& program = inputs.programs[index];
        const auto& shader = inputs.stages[index];
        const auto& result = *shader.program;
        const auto& plan = plans[index];
        auto& planStage = data->stages[index];
        planStage.plan = plan;
        planStage.tape = tapes[index];
        planStage.userWords = static_cast<std::uint32_t>(program.userData.size());
        auto& taught = variant->stages[index];
        taught.pushOffset = shader.pushConstantOffset;
        taught.flatBinding = bindingOf(result, Role::FlattenedSrt);
        taught.buffersBinding = bindingOf(result, Role::GuestBuffers);
        if (!walkStage(planStage, program.userData, program.binary.codeAddress, ranges, taught.taught)) return refuse(TeachRefusal::Walk);
        const auto& walk = taught.taught;
        const auto& layout = result.blockLayout;
        if (layout.words != 0) {
            if (taught.flatBinding == NoBinding) return refuse(TeachRefusal::Words);
            const auto& words = result.bindings[taught.flatBinding].guestDescriptor;
            if (words.size() < layout.words || walk.flat.size() < layout.words || !std::equal(walk.flat.begin(), walk.flat.begin() + layout.words, words.begin())) return refuse(TeachRefusal::Words);
        }
        planStage.keyResult.assign(walk.results.size(), 1u);
        if (walk.results.size() != plan->materializationSources.size()) return refuse(TeachRefusal::Walk);
        if (taught.buffersBinding != NoBinding) {
            const auto& binding = result.bindings[taught.buffersBinding];
            if (binding.guestDescriptor.size() < static_cast<std::size_t>(binding.count) * 4u || binding.count > plan->info.buffers.size()) return refuse(TeachRefusal::Buffers);
            planStage.bufferResult.assign(binding.count, -1);
            planStage.bufferMask.assign(binding.count, 0u);
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto source = plan->info.buffers[element].source;
                const auto at = std::find(plan->materializationSources.begin(), plan->materializationSources.end(), source);
                if (at == plan->materializationSources.end()) return refuse(TeachRefusal::Buffers);
                const auto slot = static_cast<std::size_t>(at - plan->materializationSources.begin());
                const auto& value = walk.results[slot];
                std::uint8_t mask = 0;
                bool known = value.dwordCount >= 4;
                for (std::uint32_t dword = 0; known && dword < 4; ++dword) {
                    const auto bound = binding.guestDescriptor[static_cast<std::size_t>(element) * 4u + dword];
                    if (bound == value.dwords[dword]) mask |= static_cast<std::uint8_t>(1u << dword);
                    else if (bound != 0u || dword < 2u) known = false;
                }
                if (!known) return refuse(TeachRefusal::Buffers);
                planStage.bufferMask[element] = mask;
                planStage.bufferResult[element] = static_cast<std::int32_t>(slot);
                planStage.keyResult[slot] = 0u;
            }
        }
        planStage.flatWords = layout.words;
        const auto& forwarded = result.shaderDataUserWords;
        for (std::uint32_t slot = 0; slot < forwarded.size(); ++slot) {
            if (forwarded[slot] >= program.userData.size() || slot - result.blockAddressDword < 2u) continue;
            const auto at = static_cast<std::size_t>(slot) * sizeof(std::uint32_t);
            if (at + sizeof(std::uint32_t) > result.pushConstants.size()) continue;
            std::uint32_t pushed = 0;
            std::memcpy(&pushed, result.pushConstants.data() + at, sizeof(pushed));
            if (pushed != program.userData[forwarded[slot]]) return refuse(TeachRefusal::Forwarded);
        }
        if (!result.vertexAttributes.empty()) {
            const auto& served = inputs.served[index];
            const ShaderRecompiler::ShaderVertexStageInfo* was = served != nullptr && served->vertexInfo != nullptr ? served->vertexInfo.get() : index < inputs.vertexInfos.size() && inputs.vertexInfos[index] ? &*inputs.vertexInfos[index] : nullptr;
            if (was == nullptr) return refuse(TeachRefusal::Vertex);
            ShaderRecompiler::ShaderVertexStageInfo fresh{};
            if (!decodeVertex(program, program.userData, *was, decodeReads, fresh) || decodeReads.empty()) return refuse(TeachRefusal::Vertex);
            for (const auto& attribute : result.vertexAttributes) {
                if (attribute.location >= fresh.resourcesNum || attribute.resource.fields != fresh.resources[attribute.location].fields) return refuse(TeachRefusal::Vertex);
            }
            taught.vertexInfo = fresh;
            planStage.vertexInputs = true;
        }
        taught.variantKey = ShaderRecompiler::BlockVariantKey(layout, walk.flat, program.userData);
        mixWalk(variant->digest, planStage, walk);
    }
    variant->recipe = inputs.recipe;
    variant->writes = inputs.writes;
    variant->shape = inputs.shape;
    const bool fronts = !bounded || !Graphics::CoreFlatDraw();
    for (std::size_t index = 0; index < count; ++index) {
        const auto& shader = inputs.stages[index];
        const auto& result = *shader.program;
        auto& taught = variant->stages[index];
        if (bounded) {
            for (const auto& owner : inputs.owners) {
                if (owner.get() != shader.program) continue;
                taught.base = owner;
                break;
            }
        }
        if (taught.base == nullptr) taught.base = std::make_shared<ShaderRecompiler::RecompileResult>(result);
        if (shader.registeredImageViews != nullptr) taught.registeredViews = std::make_shared<const std::vector<ShaderRecompiler::RegisteredImageView>>(*shader.registeredImageViews);
        if (fronts) variant->frontResults[index] = result;
        variant->backResults[index] = result;
    }
    variant->plan = data;
    variant->zoned = true;
    variant->zoneDigest = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull};
    for (std::size_t index = 0; index < count && variant->zoned; ++index) {
        const auto& planStage = data->stages[index];
        const auto& zoneOf = planStage.tape->Driver();
        auto& taught = variant->stages[index].taught;
        variant->zoned = zoneOf.complete && zoneOf.sources.size() == taught.results.size();
        for (std::size_t element = 0; variant->zoned && element < planStage.bufferResult.size(); ++element) {
            const auto slot = planStage.bufferResult[element];
            variant->zoned = slot < 0 || zoneOf.sources[static_cast<std::size_t>(slot)].dwords >= 4u;
        }
        if (!variant->zoned) break;
        std::vector<std::uint32_t> again(planStage.plan->srtReads.size());
        taught.zone.assign(zoneOf.words, 0u);
        const auto reader = std::make_unique<FlatReader>();
        variant->zoned = planStage.tape->RunInto(*planStage.plan, inputs.programs[index].userData, inputs.programs[index].binary.codeAddress, &flatReadWords, reader.get(), again, taught.zone) == ShaderRecompiler::BlockWalk::Done && again.size() >= planStage.flatWords && taught.flat.size() >= planStage.flatWords && std::equal(again.begin(), again.begin() + planStage.flatWords, taught.flat.begin());
        for (std::size_t source = 0; variant->zoned && source < zoneOf.sources.size(); ++source) {
            const auto& value = taught.results[source];
            for (std::uint32_t dword = 0; dword < zoneOf.sources[source].dwords && dword < value.dwordCount && dword < value.dwords.size(); ++dword) variant->zoned = variant->zoned && taught.zone[zoneOf.sources[source].word + dword] == value.dwords[dword];
        }
        if (variant->zoned) mixZone(variant->zoneDigest, planStage, taught.zone);
    }
    if (!variant->zoned) refuse(TeachRefusal::Zone);
    variant->frontStages.reserve(count);
    variant->backStages.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto& shader = inputs.stages[index];
        const auto* views = variant->stages[index].registeredViews.get();
        variant->frontStages.push_back({shader.stage, &variant->frontResults[index], shader.pushConstantOffset, views});
        variant->backStages.push_back({shader.stage, &variant->backResults[index], shader.pushConstantOffset, views});
    }
    {
        const std::lock_guard lock(state->inboxMutex);
        state->inbox.emplace_back(inputs.key, std::move(variant));
    }
    state->inboxed.store(true, std::memory_order_release);
}

std::string Front::Compare(const CoreDraw& candidate, const DriverDetail::DrawDecode& decode, std::span<const Graphics::CompiledShader> stages, const DrawRecipe* recipe) const {
    const auto& variant = *candidate.variant;
    const auto& data = *variant.plan;
    if (stages.size() != data.stageCount) return "the count of stages";
    if (decode.programs.size() != data.decode->programs.size()) return "the count of programs";
    for (std::size_t index = 0; index < stages.size(); ++index) {
        const auto& a = decode.programs[index];
        const auto& b = data.decode->programs[index];
        if (a.binary.codeAddress != b.binary.codeAddress || a.snapshot != b.snapshot || a.codeOffset != b.codeOffset || a.userDataBase != b.userDataBase) return "the program of stage " + std::to_string(index);
        if (a.userData != candidate.userData[index]) return "the user words of stage " + std::to_string(index);
        if (stages[index].program == nullptr) return "a missing stage";
        const auto& old = *stages[index].program;
        ShaderRecompiler::RecompileResult patched = *variant.stages[index].base;
        Patch(data, index, variant.stages[index], candidate.walks[index], candidate.userData[index], variant.stages[index].vertexInfo ? &candidate.vertex[index] : nullptr, patched);
        const char* what = old.variantId != patched.variantId ? "variant" : old.specializationId != patched.specializationId ? "specialization" : old.pushConstants != patched.pushConstants ? "push constants" : old.shaderDataUserWords != patched.shaderDataUserWords ? "forwarded words" : stages[index].pushConstantOffset != variant.stages[index].pushOffset ? "push offset" : nullptr;
        if (what == nullptr && (old.bindings.size() != patched.bindings.size() || !std::equal(old.bindings.begin(), old.bindings.end(), patched.bindings.begin(), sameBinding))) {
            what = "bindings";
            for (std::size_t binding = 0; binding < old.bindings.size() && binding < patched.bindings.size(); ++binding) {
                if (sameBinding(old.bindings[binding], patched.bindings[binding])) continue;
                const auto role = old.bindings[binding].role;
                what = role == Role::FlattenedSrt ? "flattened words" : role == Role::GuestBuffers ? "buffer descriptors" : role == Role::GuestImages ? "image descriptors" : role == Role::GuestSamplers ? "sampler descriptors" : role == Role::ShaderData ? "shader data" : "bindings";
                break;
            }
        }
        if (what == nullptr && (old.vertexAttributes.size() != patched.vertexAttributes.size() || !std::equal(old.vertexAttributes.begin(), old.vertexAttributes.end(), patched.vertexAttributes.begin(), sameAttribute))) what = "vertex attributes";
        if (what != nullptr) return std::string(what) + " of stage " + std::to_string(index);
    }
    if (recipe != nullptr && variant.recipe.get() != recipe && (variant.recipe->pipelineKey != recipe->pipelineKey || variant.recipe->targetViews != recipe->targetViews || variant.recipe->passKey != recipe->passKey)) return "the recipe";
    if (recipe != nullptr && variant.recipe.get() != recipe) {
        const auto taught = variant.recipe->templateRef.lock();
        const auto classic = recipe->templateRef.lock();
        if (taught != nullptr && classic != nullptr && taught != classic) return "the template of the recipe";
    }
    return {};
}

}

namespace AgcDriver::DriverDetail {

bool Driver::tryFlatDraw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, const Pm4::DrawParameters& drawParameters, std::uint64_t deviceSerial, Core::FrontOutcome& outcome) {
    static const bool verify = Graphics::VerifyCoreRun();
    static const bool stats = Graphics::CoreRunStats();
    static const bool shaped = InstallCoreKeyRegisters();
    static_cast<void>(shaped);
    auto& front = Core::Front::Get();
    Pm4::RefreshDigests(queue);
    if (verify) front.VerifyDigests(queue);
    Core::FlatInputs inputs;
    inputs.queue = &queue;
    inputs.shaders = submission.shaders.get();
    inputs.packet = packet;
    inputs.parameters = &drawParameters;
    inputs.device = deviceSerial;
    inputs.epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
    inputs.verifyOnly = verify;
    inputs.plain = [](void* context, const QueueState& state, const Pm4::DrawParameters& parameters) { return static_cast<Driver*>(context)->precheckPlain(state, parameters); };
    inputs.context = this;
    outcome = front.Flat(inputs);
    if (stats) {
        static thread_local std::uint32_t since = 0;
        if ((++since & 0x3ffu) == 0) {
            static std::atomic<std::int64_t> lastReport{0};
            const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            auto last = lastReport.load(std::memory_order_relaxed);
            if (nowMs - last >= 5000 && lastReport.compare_exchange_strong(last, nowMs)) std::fprintf(stderr, "%s%c", Core::Front::Report().c_str(), 10);
        }
    }
    if (!outcome.taken) return false;
    auto* const variant = outcome.variant;
    const auto* const at = outcome.op;
    const auto epoch = inputs.epoch;
    static thread_local std::uint32_t posts = 0;
    const bool timedPost = stats && (++posts & 63u) == 0;
    const auto postStarted = timedPost ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const bool busy = DrawFinisher::Busy();
    static auto* const ops = DrawPipeline::Ops();
    if (ops != nullptr && !ops->Closed()) {
        static const bool runs = [] {
            DrawPipeline::RunOpsWith([](const Core::Op& op, bool skipped) {
                const auto* const record = reinterpret_cast<const std::byte*>(op.draw.slice.bytes);
                if (skipped) {
                    Core::Front::Get().FlatEmitted(Core::Front::FlatRead(record));
                    return;
                }
                Driver::Get().commitFlatDraw(record);
            });
            return true;
        }();
        static_cast<void>(runs);
        Core::Op op = Core::MakeOp(Core::OpKind::None);
        std::memcpy(&op, at, sizeof(op));
        front.FlatPosted(at);
        DrawPipeline::FollowEpoch(epoch);
        precollectCoreDraw(*variant->recipe, outcome.reads);
        if (!ops->Post(op, variant->writes, outcome.reads, GuestMemory::HeldCollectEpoch())) {
            DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Draw, [this, at] { commitFlatDraw(at); }, variant->writes, 0, {}, std::vector<std::pair<std::uint64_t, std::uint64_t>>(outcome.reads.begin(), outcome.reads.end()), true);
            return true;
        }
        DrawPipeline::Queue0().WakeForOps();
        if (timedPost) Core::Front::NoteFlatPosted(busy, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - postStarted).count()));
        return true;
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> reads(outcome.reads.begin(), outcome.reads.end());
    front.FlatPosted(at);
    if (busy) {
        DrawFinisher::Pending held;
        held.unknown = false;
        held.writes = variant->writes;
        DrawFinisher::Get().Enqueue([this, at, variant, epoch, reads = std::move(reads)]() mutable {
            DrawPipeline::FollowEpoch(epoch);
            precollectCoreDraw(*variant->recipe, reads);
            DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Draw, [this, at] { commitFlatDraw(at); }, variant->writes, 0, {}, std::move(reads), true);
        }, std::move(held));
    } else {
        DrawPipeline::FollowEpoch(epoch);
        precollectCoreDraw(*variant->recipe, reads);
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Draw, [this, at] { commitFlatDraw(at); }, variant->writes, 0, {}, std::move(reads), true);
    }
    if (timedPost) Core::Front::NoteFlatPosted(busy, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - postStarted).count()));
    return true;
}

void Driver::commitFlatDraw(const std::byte* at) {
    const auto started = std::chrono::steady_clock::now();
    const auto draw = Core::Front::FlatRead(at);
    struct Emitted {
        const Core::FlatDraw& draw;
        ~Emitted() { Core::Front::Get().FlatEmitted(draw); }
    } emitted{draw};
    DrawPipeline::FollowEpoch(draw.op.order);
    GuestMemory::SetCurrentPacket(draw.op.draw.opcode, 0);
    auto& variant = *draw.variant;
    for (std::uint32_t stage = 0; stage < variant.plan->stageCount; ++stage) Core::Front::FlatPatch(draw, stage, variant.backResults[stage]);
    const std::span<const Graphics::CompiledShader> stages(variant.backStages);
    Pm4::DrawParameters parameters{draw.op.draw.indexAddress, draw.op.draw.indexCount, draw.op.draw.indexBytes, draw.op.draw.instances, draw.op.draw.firstIndex, (draw.op.draw.drawFlags & Core::DrawIndexed) != 0u, static_cast<std::uint32_t>(draw.op.draw.baseVertex), draw.op.draw.firstInstance};
    const auto& graphics = variant.plan->decode->state;
    bool restarted = false;
    {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Draw);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        const auto localDevice = device.Load();
        struct SubmitDue {
            Driver& driver;
            const std::shared_ptr<VulkanDevice>& device;
            ~SubmitDue() noexcept(false) {
                if (std::uncaught_exceptions() == 0) driver.submitDueAfterCommit(device.get());
            }
        } submitDue{*this, localDevice};
        recordLabelsForPacket(localDevice.get(), 0);
        noteDrawWriters(stages, 0);
        if (!Core::RecordBack(*localDevice, graphics, parameters, stages, variant.recipe, variant.serial)) {
            restarted = true;
            variant.stale.store(true, std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
            localDevice->Draw(graphics, parameters, stages, {}, nullptr, nullptr, variant.shape);
        }
        Graphics::NoteDrawSite(variant.shape);
    }
    Core::Front::NoteCommitted(restarted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
    Core::ReportBack();
}

bool Driver::tryCoreDraw(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, const Pm4::DrawParameters& drawParameters, std::uint64_t deviceSerial, Core::FrontOutcome& outcome) {
    static const bool verify = Graphics::VerifyCoreRun();
    static const bool stats = Graphics::CoreRunStats();
    static const bool shaped = InstallCoreKeyRegisters();
    static_cast<void>(shaped);
    auto& front = Core::Front::Get();
    Pm4::RefreshDigests(queue);
    if (verify) front.VerifyDigests(queue);
    const auto opcode = (packet[0] >> 8u) & 0xffu;
    outcome = front.Walk(queue, *submission.shaders, drawParameters, opcode, deviceSerial, verify);
    if (stats) {
        static std::atomic<std::int64_t> lastReport{0};
        const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        auto last = lastReport.load(std::memory_order_relaxed);
        if (nowMs - last >= 5000 && lastReport.compare_exchange_strong(last, nowMs)) std::fprintf(stderr, "%s%c", Core::Front::Report().c_str(), 10);
    }
    if (!outcome.taken) return false;
    const auto enqueueStarted = std::chrono::steady_clock::now();
    auto record = std::move(outcome.candidate);
    record->epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
    record->queue = submission.queue;
    auto reads = QueuedDrawReadRanges(drawParameters, record->variant->frontStages);
    auto push = [this, record, writes = record->variant->writes, reads = std::move(reads)]() mutable {
        DrawPipeline::FollowEpoch(record->epoch);
        precollectCoreDraw(*record->variant->recipe, reads);
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Draw, [this, record] { commitCoreDraw(record); }, std::move(writes), 0, {}, std::move(reads), true);
    };
    if (DrawFinisher::Busy()) {
        DrawFinisher::Pending held;
        held.unknown = false;
        held.writes = record->variant->writes;
        DrawFinisher::Get().Enqueue(std::move(push), std::move(held));
    } else {
        push();
    }
    Core::Front::NoteEnqueued(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - enqueueStarted).count()));
    return true;
}

}
