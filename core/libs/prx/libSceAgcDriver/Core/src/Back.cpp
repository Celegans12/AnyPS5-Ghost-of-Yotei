#include "prx/libSceAgcDriver/Core/include/Back.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include <chrono>
#include <exception>
#include <mutex>

namespace AgcDriver::Core {

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
        if (localDevice->DrawFromRecipe(graphics, record->parameters, stages, {}, variant.recipe) != RecipeOutcome::Recorded) {
            restarted = true;
            variant.stale.store(true, std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, VulkanDevice::RecipeKind::Draw);
            localDevice->Draw(graphics, record->parameters, stages, {}, nullptr, nullptr, variant.shape);
        }
        Graphics::NoteDrawSite(variant.shape);
    }
    Core::Front::NoteCommitted(restarted, static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count()));
    Core::Front::Get().Release(record);
}

}
