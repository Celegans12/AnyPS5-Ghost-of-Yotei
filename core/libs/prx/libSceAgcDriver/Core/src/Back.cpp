#include "prx/libSceAgcDriver/Core/include/Back.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
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

std::string BackReport() {
    const auto counts = Graphics::CoreBackCounted();
    char text[768];
    std::snprintf(text, sizeof(text), "[core-back] (cumulative) %llu draws committed by the light back end, %llu continued the run of the draw before, %llu template proofs inherited from the run, %llu answers verified, %llu mismatches, %llu left to the full path; inputs: %llu ranges, %llu held by the run, %llu bound in place; in the run: %llu templates without bindings, %llu templates already noted, %llu target marks already made", static_cast<unsigned long long>(counts[0]), static_cast<unsigned long long>(counts[1]), static_cast<unsigned long long>(counts[2]), static_cast<unsigned long long>(counts[3]), static_cast<unsigned long long>(counts[4]), static_cast<unsigned long long>(counts[5]), static_cast<unsigned long long>(counts[6]), static_cast<unsigned long long>(counts[7]), static_cast<unsigned long long>(counts[8]), static_cast<unsigned long long>(counts[9]), static_cast<unsigned long long>(counts[10]), static_cast<unsigned long long>(counts[11]));
    return text;
}

bool RecordBack(VulkanDevice& device, const Graphics::State& graphics, const Pm4::DrawParameters& parameters, std::span<const Graphics::CompiledShader> stages, const std::shared_ptr<const DrawRecipe>& recipe) {
    const auto mode = BackModeNow();
    if (mode == BackMode::Recipe) return device.DrawFromRecipe(graphics, parameters, stages, {}, recipe) == RecipeOutcome::Recorded;
    const auto context = device.graphicsContext();
    return recipe->device == context.device && Graphics::DrawCoreWithRecipe(context, graphics, parameters, stages, *recipe, mode == BackMode::Verify).recorded;
}

void ReportBack() {
    if (BackModeNow() == BackMode::Recipe || !Graphics::CoreRunStats()) return;
    static std::atomic<std::int64_t> lastReport{0};
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (nowMs - last >= 5000 && lastReport.compare_exchange_strong(last, nowMs)) std::fprintf(stderr, "%s%c", BackReport().c_str(), 10);
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
        if (!Core::RecordBack(*localDevice, graphics, record->parameters, stages, variant.recipe)) {
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
