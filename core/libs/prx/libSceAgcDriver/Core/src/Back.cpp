#include "prx/libSceAgcDriver/Core/include/Back.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>

namespace AgcDriver::Core {

namespace {

std::atomic<int>& testedMode() {
    static std::atomic<int> mode{-1};
    return mode;
}

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
        if (!Core::RecordBack(*localDevice, graphics, record->parameters, stages, variant.recipe)) {
            restarted = true;
            variant.stale.store(true, std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
            localDevice->Draw(graphics, record->parameters, stages, {}, nullptr, nullptr, variant.shape);
        }
        Graphics::NoteDrawSite(variant.shape);
    }
    Core::Front::NoteCommitted(restarted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
    Core::ReportBack();
    Core::Front::Get().Release(record);
}

}
