#include "prx/libSceAgcDriver/Core/include/Back.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/VertexInput.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <vector>

namespace AgcDriver::Core {

namespace {

std::atomic<int>& testedMode() {
    static std::atomic<int> mode{-1};
    return mode;
}

class GuestReplayProof final : public ReplayProof {
public:
    std::uint64_t Generation() override {
        return GuestMemory::SettledGeneration();
    }

    bool PagesUnchanged(std::span<const ReplayQuery> queries) override {
        for (const auto& query : queries) {
            if (!GuestMemory::UnchangedSince(query.address, static_cast<std::size_t>(query.bytes), query.since)) return false;
        }
        return true;
    }

    bool BlockWritten(const ReplayQuery& query) override {
        return GuestMemory::WrittenSince(query.address, static_cast<std::size_t>(query.bytes), query.since);
    }

    bool TokensLive() override {
        return GuestMemory::EpochSweeps();
    }

    ReplayToken TokenFor(const ReplayRange& range) override {
        const auto token = GuestMemory::TokenFor(range.address, static_cast<std::size_t>(range.bytes));
        return {token.content, token.context};
    }

    bool StillCurrent(const ReplayRange& range, const ReplayToken& token) override {
        return GuestMemory::StillCurrent(range.address, static_cast<std::size_t>(range.bytes), GuestMemory::ContentToken{token.content, token.context});
    }
};

GuestReplayProof& replayProof() {
    static GuestReplayProof proof;
    return proof;
}

ReplayMeter& replayMeter() {
    static ReplayMeter meter(13, 13, 12, 0, 3);
    return meter;
}

std::uint64_t nanosecondsSince(std::chrono::steady_clock::time_point started) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
}

std::atomic<int>& testedReplay() {
    static std::atomic<int> mode{-1};
    return mode;
}

enum class ReplayCount : std::size_t { Draws, Replayed, Captured, Stored, Absent, Unmatched, Token, Pending, Unstorable, Failed, TimedReplayed, ReplayedNanoseconds, TimedOther, OtherNanoseconds, TimedLean, LeanNanoseconds, TimedRefreshed, RefreshedNanoseconds, Refreshed, Count };

std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ReplayCount::Count)>& replayCounters() {
    static std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ReplayCount::Count)> held{};
    return held;
}

void countReplay(ReplayCount kind, std::uint64_t amount = 1) {
    auto& counter = replayCounters()[static_cast<std::size_t>(kind)];
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

void releaseReplay(void*, const ReplayEntry& entry) noexcept {
    Graphics::DeleteCoreReplay(reinterpret_cast<Graphics::CoreReplay*>(static_cast<std::uintptr_t>(entry.resolved[0])));
}

DrawMemo& drawMemo() {
    static auto* memo = new DrawMemo(ReplayRelease{&releaseReplay, nullptr});
    return *memo;
}

bool replayStillCurrent(void*, std::uint64_t address, std::size_t bytes, const GuestMemory::ContentToken& token) {
    return GuestMemory::StillCurrent(address, bytes, token);
}

bool replayPendingGpuWrite(void* context, std::uint64_t address, std::size_t bytes) {
    const auto* recorder = static_cast<const Graphics::Recorder*>(context);
    return recorder->PendingWriteOverlaps(address, bytes) || recorder->PendingLabelIn(address, bytes) || recorder->QueuedStoreOverlaps(address, bytes) || recorder->QueuedKeyStoreOverlaps(address, bytes);
}

std::uint64_t replayInputsKey(const Pm4::DrawParameters& parameters, std::span<const Graphics::CompiledShader> stages) {
    auto hash = ReplayMix(0x452821e638d01377ull, parameters.indexAddress);
    hash = ReplayMix(hash, parameters.indexCount | (static_cast<std::uint64_t>(parameters.indexSize) << 32u));
    hash = ReplayMix(hash, parameters.instanceCount | (static_cast<std::uint64_t>(parameters.flags) << 32u));
    hash = ReplayMix(hash, parameters.firstVertex | (static_cast<std::uint64_t>(parameters.firstInstance) << 32u));
    hash = ReplayMix(hash, parameters.indexed ? 1u : 0u);
    if (!stages.empty() && stages.front().program != nullptr) {
        for (const auto& attribute : stages.front().program->vertexAttributes) {
            hash = ReplayMix(hash, attribute.resource.fields[0] | (static_cast<std::uint64_t>(attribute.resource.fields[1]) << 32u));
            hash = ReplayMix(hash, attribute.resource.fields[2] | (static_cast<std::uint64_t>(attribute.resource.fields[3]) << 32u));
        }
    }
    return hash;
}

bool replayBack(const Graphics::Context& context, const Graphics::State& graphics, const Pm4::DrawParameters& parameters, std::span<const Graphics::CompiledShader> stages, const std::shared_ptr<const DrawRecipe>& recipe, std::uint64_t identity, bool verifyBack, bool verifyReplay) {
    static Graphics::CoreReplay* spare = nullptr;
    thread_local std::uint32_t tick = 0;
    const bool timed = (++tick & 15u) == 0;
    const auto started = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    countReplay(ReplayCount::Draws);
    auto& memo = drawMemo();
    const ReplayKey key{identity, static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(recipe.get())), replayInputsKey(parameters, stages), 0};
    auto* entry = memo.Find(key);
    auto* held = entry != nullptr ? reinterpret_cast<Graphics::CoreReplay*>(static_cast<std::uintptr_t>(entry->resolved[0])) : nullptr;
    bool current = false;
    if (entry == nullptr) {
        countReplay(ReplayCount::Absent);
    } else if (!Graphics::CoreReplayMatches(*held, context.device, identity, recipe.get(), parameters, stages)) {
        countReplay(ReplayCount::Unmatched);
    } else {
        ReplayNow now;
        now.device = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(context.device));
        now.stillCurrent = &replayStillCurrent;
        now.pendingGpuWrite = &replayPendingGpuWrite;
        now.context = context.recorder;
        const auto miss = memo.Fresh(*entry, now);
        current = miss == ReplayMiss::None;
        if (miss == ReplayMiss::PendingGpuWrite) countReplay(ReplayCount::Pending);
        else if (!current) countReplay(ReplayCount::Token);
    }
    bool recorded = false;
    if (current) {
        recorded = Graphics::DrawCoreReplayed(context, graphics, parameters, stages, recipe, verifyBack || verifyReplay, *held, true, verifyReplay, identity).recorded;
        if (!recorded || !Graphics::CoreReplayRecaptured(*held)) {
            countReplay(recorded ? ReplayCount::Replayed : ReplayCount::Failed);
            if (timed) {
                const bool lean = recorded && Graphics::CoreReplayLean(*held);
                countReplay(lean ? ReplayCount::TimedLean : ReplayCount::TimedReplayed);
                countReplay(lean ? ReplayCount::LeanNanoseconds : ReplayCount::ReplayedNanoseconds, nanosecondsSince(started));
            }
            return recorded;
        }
    } else {
        if (held == nullptr) {
            if (spare == nullptr) spare = Graphics::NewCoreReplay();
            held = spare;
            Graphics::ClearCoreReplay(*held);
        }
        recorded = Graphics::DrawCoreReplayed(context, graphics, parameters, stages, recipe, verifyBack || verifyReplay, *held, false, verifyReplay, identity).recorded;
    }
    const bool refreshed = recorded && Graphics::CoreReplayLean(*held);
    countReplay(!recorded ? ReplayCount::Failed : refreshed ? ReplayCount::Refreshed : ReplayCount::Captured);
    const auto ranges = Graphics::CoreReplayRanges(*held);
    bool storable = recorded && Graphics::CoreReplayStorable(*held) && ranges <= DrawProof{}.ranges.size();
    DrawProof proof;
    proof.device = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(context.device));
    for (std::uint32_t index = 0; storable && index < ranges; ++index) {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        Graphics::CoreReplayRange(*held, index, address, bytes, proof.ranges[index]);
        if (bytes == 0 || bytes > 0xffffffffull) storable = false;
        proof.rangeAddress[index] = address;
        proof.rangeBytes[index] = static_cast<std::uint32_t>(bytes);
    }
    if (storable) {
        const std::uint64_t word = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(held));
        if (memo.Store(key, proof, std::span<const std::uint64_t>(&word, 1), 0) != nullptr) {
            countReplay(ReplayCount::Stored);
            if (held == spare) spare = nullptr;
        }
    } else if (recorded) {
        countReplay(ReplayCount::Unstorable);
    }
    if (timed) {
        countReplay(refreshed ? ReplayCount::TimedRefreshed : ReplayCount::TimedOther);
        countReplay(refreshed ? ReplayCount::RefreshedNanoseconds : ReplayCount::OtherNanoseconds, nanosecondsSince(started));
    }
    return recorded;
}

}

bool ReplayTimedNext() {
    thread_local std::uint32_t tick = 0;
    return (++tick & 15u) == 0;
}

bool ReplayStats() {
    static const bool enabled = std::getenv("APS5_CORE_REPLAY_STATS") != nullptr;
    return enabled;
}

bool ReplaySampled(const FrontVariant& variant) {
    return replayMeter().Sampled(variant.plan->id, variant.digest);
}

ReplayDraw ReplayDrawOf(const CoreDraw& record, std::span<const Graphics::CompiledShader> stages) {
    const auto& variant = *record.variant;
    ReplayDraw draw;
    draw.plan = variant.plan->id;
    draw.digest = variant.digest;
    draw.variant = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&variant));
    draw.recipe = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(variant.recipe.get()));
    const auto& parameters = record.parameters;
    const std::uint64_t indexBytes = parameters.indexed ? static_cast<std::uint64_t>(parameters.indexCount) * parameters.indexSize : 0u;
    auto inputs = ReplayMix(0x452821e638d01377ull, indexBytes != 0 ? parameters.indexAddress : 0u);
    inputs = ReplayMix(inputs, indexBytes);
    if (indexBytes != 0) AddReplayRange(draw, parameters.indexAddress, indexBytes);
    if (!stages.empty() && stages.front().program != nullptr) {
        for (const auto& attribute : stages.front().program->vertexAttributes) {
            for (const auto field : attribute.resource.fields) inputs = ReplayMix(inputs, field);
            inputs = ReplayMix(inputs, attribute.location | (static_cast<std::uint64_t>(attribute.fetchIndex) << 32u));
            if (Graphics::NullVertexDescriptor(attribute)) continue;
            const auto address = attribute.resource.fields[0] | (static_cast<std::uint64_t>(attribute.resource.fields[1] & 0xffffu) << 32u);
            try {
                if (Graphics::VertexFetchOutOfRange(attribute)) continue;
                AddReplayRange(draw, address, Graphics::VertexBufferExtent(attribute));
            } catch (const std::exception&) {
                draw.rangesComplete = false;
            }
        }
    }
    draw.inputs = inputs;
    auto drawn = ReplayMix(0x3f84d5b5b5470917ull, parameters.indexCount);
    drawn = ReplayMix(drawn, parameters.indexSize | (static_cast<std::uint64_t>(parameters.indexed ? 1u : 0u) << 32u));
    drawn = ReplayMix(drawn, parameters.instanceCount | (static_cast<std::uint64_t>(parameters.flags) << 32u));
    drawn = ReplayMix(drawn, parameters.firstVertex | (static_cast<std::uint64_t>(parameters.firstInstance) << 32u));
    draw.parameters = drawn;
    return draw;
}

const ReplayDraw& ReplayTargetsOf(const FrontVariant& variant, const Graphics::State& graphics) {
    static std::vector<ReplayDraw> cache(1024);
    const auto plan = variant.plan->id;
    auto& draw = cache[plan & 1023u];
    if (plan != 0 && draw.plan == plan) return draw;
    draw = ReplayDraw{};
    draw.plan = plan;
    auto targets = ReplayMix(0xbe5466cf34e90c6cull, graphics.colors.size() | (graphics.hasColorTarget ? 1ull << 32u : 0u));
    const auto color = [&](const Graphics::ColorTarget& target) {
        targets = ReplayMix(targets, target.address);
        targets = ReplayMix(targets, target.bytes);
        targets = ReplayMix(targets, target.dccAddress);
        targets = ReplayMix(targets, target.cmaskAddress);
        targets = ReplayMix(targets, target.cmaskBytes);
        targets = ReplayMix(targets, target.fmaskAddress);
        targets = ReplayMix(targets, target.arraySlice | (static_cast<std::uint64_t>(target.arrayLastSlice) << 32u));
        targets = ReplayMix(targets, static_cast<std::uint64_t>(target.format) | (static_cast<std::uint64_t>(target.extent.width) << 32u));
        targets = ReplayMix(targets, target.extent.height);
        if (target.dccAddress != 0) AddReplayMetadata(draw, target.dccAddress, Graphics::DccKeyBytes(target.bytes));
        if (target.cmaskAddress != 0) AddReplayMetadata(draw, target.cmaskAddress, target.cmaskBytes);
    };
    if (graphics.hasColorTarget) color(graphics.color);
    for (const auto& target : graphics.colors) color(target);
    if (graphics.depth) {
        const auto& depth = *graphics.depth;
        targets = ReplayMix(targets, depth.address);
        targets = ReplayMix(targets, depth.stencilAddress);
        targets = ReplayMix(targets, depth.htileAddress);
        targets = ReplayMix(targets, static_cast<std::uint64_t>(depth.extent.width) | (static_cast<std::uint64_t>(depth.extent.height) << 32u));
        if (depth.htileAddress != 0) AddReplayMetadata(draw, depth.htileAddress, Graphics::HtileMetadataBytes(depth.extent));
    }
    draw.targets = targets;
    return draw;
}

bool ReplayFilled(const ReplayDraw& targets) {
    return replayMeter().Filled(targets, replayProof());
}

void NoteReplay(ReplayDraw* draw, const ReplayDraw& targets, bool fallback, bool filled, bool timed, std::uint64_t spentNanoseconds) {
    auto& meter = replayMeter();
    auto& proof = replayProof();
    const auto started = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (draw != nullptr) {
        draw->targets = targets.targets;
        static_cast<void>(meter.Commit(*draw, fallback, filled, proof));
    } else {
        meter.NoteUnsampled();
    }
    meter.NoteTargets(targets, proof.Generation());
    if (timed) meter.NoteCost(spentNanoseconds + nanosecondsSince(started), draw != nullptr);
    if (((meter.Counted().draws + meter.Counted().unsampled) & 63u) != 0) return;
    static std::atomic<std::int64_t> lastReport{0};
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (last == 0) {
        lastReport.compare_exchange_strong(last, nowMs);
        return;
    }
    if (nowMs - last < 5000 || !lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto front = Front::Counted();
    std::fprintf(stderr, "%s%c", meter.TakeReport(front.draws, front.committed, proof.TokensLive(), static_cast<double>(nowMs - last) / 1000.0).c_str(), 10);
}

BackMode BackModeNow() {
    const auto forced = testedMode().load(std::memory_order_relaxed);
    if (forced >= 0) return static_cast<BackMode>(forced);
    static const BackMode configured = std::getenv("APS5_VERIFY_CORE_BACK") != nullptr ? BackMode::Verify : std::getenv("APS5_CORE_BACK") != nullptr ? BackMode::Light : BackMode::Recipe;
    return configured;
}

void SetBackModeForTest(int mode) {
    testedMode().store(mode, std::memory_order_relaxed);
}

BackReplay BackReplayNow() {
    const auto forced = testedReplay().load(std::memory_order_relaxed);
    if (forced >= 0) return static_cast<BackReplay>(forced);
    static const BackReplay configured = !GuestMemory::EpochSweeps() ? BackReplay::Off : std::getenv("APS5_VERIFY_CORE_REPLAY") != nullptr ? BackReplay::Verify : std::getenv("APS5_CORE_REPLAY") != nullptr ? BackReplay::On : BackReplay::Off;
    return configured;
}

void SetBackReplayForTest(int mode) {
    testedReplay().store(mode, std::memory_order_relaxed);
}

std::string BackReplayReport() {
    const auto at = [](ReplayCount kind) { return static_cast<unsigned long long>(replayCounters()[static_cast<std::size_t>(kind)].load(std::memory_order_relaxed)); };
    const auto back = Graphics::CoreReplayCounted();
    const auto memo = drawMemo().Counted();
    const auto share = [&](ReplayCount kind) { return at(ReplayCount::Draws) == 0 ? 0.0 : 100.0 * static_cast<double>(at(kind)) / static_cast<double>(at(ReplayCount::Draws)); };
    const auto each = [&](ReplayCount total, ReplayCount samples) { return at(samples) == 0 ? 0.0 : static_cast<double>(at(total)) / static_cast<double>(at(samples)); };
    const auto lean = Graphics::CoreReplayLeanRefusals();
    char text[1792];
    const auto wrote = std::snprintf(text, sizeof(text), "[core-replay-back] (cumulative) %llu draws offered, %llu replayed (%.1f %%) of which %llu emitted directly, %llu emitted directly with refreshed inputs (%.1f %%, %.0f ns each, %llu timed), %llu taken by the full path (%.1f %%): absent %llu, unmatched %llu, token %llu, pending GPU write %llu, in-place input gone %llu; %llu stored, %llu not storable, %llu left to the driver; kept once by the batch %llu; verified %llu, mismatches %llu; memo evicted %llu, replaced %llu; %.0f ns per draw emitted directly (%llu timed), %.0f ns per other replayed draw (%llu timed), %.0f ns per other draw (%llu timed)",
        at(ReplayCount::Draws), at(ReplayCount::Replayed), share(ReplayCount::Replayed), static_cast<unsigned long long>(back[6] - back[8]), at(ReplayCount::Refreshed), share(ReplayCount::Refreshed), each(ReplayCount::RefreshedNanoseconds, ReplayCount::TimedRefreshed), at(ReplayCount::TimedRefreshed), at(ReplayCount::Captured), share(ReplayCount::Captured), at(ReplayCount::Absent), at(ReplayCount::Unmatched), at(ReplayCount::Token), at(ReplayCount::Pending), static_cast<unsigned long long>(back[7]), at(ReplayCount::Stored), at(ReplayCount::Unstorable), at(ReplayCount::Failed),
        static_cast<unsigned long long>(back[5]), static_cast<unsigned long long>(back[3]), static_cast<unsigned long long>(back[4]), static_cast<unsigned long long>(memo.evicted), static_cast<unsigned long long>(memo.replaced), each(ReplayCount::LeanNanoseconds, ReplayCount::TimedLean), at(ReplayCount::TimedLean), each(ReplayCount::ReplayedNanoseconds, ReplayCount::TimedReplayed), at(ReplayCount::TimedReplayed), each(ReplayCount::OtherNanoseconds, ReplayCount::TimedOther), at(ReplayCount::TimedOther));
    if (wrote > 0 && static_cast<std::size_t>(wrote) < sizeof(text)) {
        std::snprintf(text + wrote, sizeof(text) - static_cast<std::size_t>(wrote), "; not emitted directly: shape %llu, run %llu, targets %llu, depth %llu, template %llu, proof %llu, queued stores %llu, summary %llu, pass %llu, objects %llu, moved buffers %llu, inputs %llu",
            static_cast<unsigned long long>(lean[0]), static_cast<unsigned long long>(lean[1]), static_cast<unsigned long long>(lean[2]), static_cast<unsigned long long>(lean[3]), static_cast<unsigned long long>(lean[4]), static_cast<unsigned long long>(lean[5]), static_cast<unsigned long long>(lean[6]), static_cast<unsigned long long>(lean[7]), static_cast<unsigned long long>(lean[8]), static_cast<unsigned long long>(lean[9]), static_cast<unsigned long long>(lean[10]), static_cast<unsigned long long>(lean[11]));
    }
    return text;
}

std::string BackReport() {
    const auto counts = Graphics::CoreBackCounted();
    char text[768];
    std::snprintf(text, sizeof(text), "[core-back] (cumulative) %llu draws committed by the light back end, %llu continued the run of the draw before, %llu template proofs inherited from the run, %llu answers verified, %llu mismatches, %llu left to the full path; inputs: %llu ranges, %llu held by the run, %llu bound in place; in the run: %llu templates without bindings, %llu templates already noted, %llu target marks already made", static_cast<unsigned long long>(counts[0]), static_cast<unsigned long long>(counts[1]), static_cast<unsigned long long>(counts[2]), static_cast<unsigned long long>(counts[3]), static_cast<unsigned long long>(counts[4]), static_cast<unsigned long long>(counts[5]), static_cast<unsigned long long>(counts[6]), static_cast<unsigned long long>(counts[7]), static_cast<unsigned long long>(counts[8]), static_cast<unsigned long long>(counts[9]), static_cast<unsigned long long>(counts[10]), static_cast<unsigned long long>(counts[11]));
    return text;
}

bool RecordBack(VulkanDevice& device, const Graphics::State& graphics, const Pm4::DrawParameters& parameters, std::span<const Graphics::CompiledShader> stages, const std::shared_ptr<const DrawRecipe>& recipe, std::uint64_t identity) {
    const auto mode = BackModeNow();
    if (mode == BackMode::Recipe) return device.DrawFromRecipe(graphics, parameters, stages, {}, recipe) == RecipeOutcome::Recorded;
    const auto context = device.graphicsContext();
    if (recipe->device != context.device) return false;
    const auto replay = BackReplayNow();
    if (replay == BackReplay::Off || identity == 0 || context.recorder == nullptr || parameters.indirect.has_value()) return Graphics::DrawCoreWithRecipe(context, graphics, parameters, stages, *recipe, mode == BackMode::Verify).recorded;
    return replayBack(context, graphics, parameters, stages, recipe, identity, mode == BackMode::Verify, replay == BackReplay::Verify);
}

void ReportBack() {
    if (BackModeNow() == BackMode::Recipe) return;
    const bool replay = BackReplayNow() != BackReplay::Off;
    if (!replay && !Graphics::CoreRunStats()) return;
    static std::atomic<std::int64_t> lastReport{0};
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (nowMs - last < 5000 || !lastReport.compare_exchange_strong(last, nowMs)) return;
    if (Graphics::CoreRunStats()) std::fprintf(stderr, "%s%c", BackReport().c_str(), 10);
    if (replay) std::fprintf(stderr, "%s%c", BackReplayReport().c_str(), 10);
}

std::span<const Graphics::CompiledShader> PrepareBack(CoreDraw& record) {
    auto& variant = *record.variant;
    const auto& data = *variant.plan;
    for (std::uint32_t stage = 0; stage < data.stageCount; ++stage) {
        const auto& taught = variant.stages[stage];
        Front::Patch(data, stage, taught, record.walks[stage], record.userData[stage], taught.vertexInfo ? &record.vertex[stage] : nullptr, variant.backResults[stage]);
    }
    return variant.backStages;
}

}

namespace AgcDriver::DriverDetail {

void Driver::commitCoreDraw(const std::shared_ptr<Core::CoreDraw>& record) {
    const auto started = std::chrono::steady_clock::now();
    DrawPipeline::FollowEpoch(record->epoch);
    GuestMemory::SetCurrentPacket(record->opcode, record->queue);
    auto& variant = *record->variant;
    const auto stages = Core::PrepareBack(*record);
    const auto& graphics = variant.plan->decode->state;
    const bool replayOn = Core::ReplayStats();
    std::optional<Core::ReplayDraw> replay;
    bool replayTimed = false;
    std::uint64_t replaySpent = 0;
    if (replayOn) {
        replayTimed = Core::ReplayTimedNext();
        const auto replayStarted = replayTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (Core::ReplaySampled(variant)) replay.emplace(Core::ReplayDrawOf(*record, stages));
        if (replayTimed) replaySpent = Core::nanosecondsSince(replayStarted);
    }
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
        recordLabelsForPacket(localDevice.get(), record->queue);
        noteDrawWriters(stages, record->queue);
        const Core::ReplayDraw* replayTargets = nullptr;
        bool filled = false;
        if (replayOn) {
            const auto fillStarted = replayTimed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
            replayTargets = &Core::ReplayTargetsOf(variant, graphics);
            filled = replay.has_value() && Core::ReplayFilled(*replayTargets);
            if (replayTimed) replaySpent += Core::nanosecondsSince(fillStarted);
        }
        if (!Core::RecordBack(*localDevice, graphics, record->parameters, stages, variant.recipe, variant.serial)) {
            restarted = true;
            variant.stale.store(true, std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
            localDevice->Draw(graphics, record->parameters, stages, {}, nullptr, nullptr, variant.shape);
        }
        if (replayTargets != nullptr) Core::NoteReplay(replay.has_value() ? &*replay : nullptr, *replayTargets, restarted, filled, replayTimed, replaySpent);
        Graphics::NoteDrawSite(variant.shape);
    }
    Core::Front::NoteCommitted(restarted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
    Core::ReportBack();
    Core::Front::Get().Release(record);
}

}
