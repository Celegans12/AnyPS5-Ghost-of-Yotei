#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BACK_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BACK_HPP

#include "prx/libSceAgcDriver/Core/include/Front.hpp"
#include "prx/libSceAgcDriver/Core/include/ReplayMeter.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include <array>
#include <cstdint>
#include <span>
#include <string>

namespace AgcDriver::Graphics {

DrawRecipeOutcome DrawCoreWithRecipe(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const DrawRecipe& recipe, bool verify);
std::array<std::uint64_t, 12> CoreBackCounted();

}

namespace AgcDriver::Core {

enum class BackMode : std::uint8_t { Recipe, Light, Verify };

[[nodiscard]] BackMode BackModeNow();
void SetBackModeForTest(int mode);
[[nodiscard]] std::string BackReport();
[[nodiscard]] std::span<const Graphics::CompiledShader> PrepareBack(CoreDraw& record);
[[nodiscard]] bool RecordBack(VulkanDevice& device, const Graphics::State& graphics, const Pm4::DrawParameters& parameters, std::span<const Graphics::CompiledShader> stages, const std::shared_ptr<const DrawRecipe>& recipe);
void ReportBack();
[[nodiscard]] bool ReplayStats();
[[nodiscard]] ReplayDraw ReplayDrawOf(const CoreDraw& record, const Graphics::State& graphics, std::span<const Graphics::CompiledShader> stages);
[[nodiscard]] bool ReplayFilled(const ReplayDraw& draw);
[[nodiscard]] bool ReplayTimedNext();
void NoteReplay(const ReplayDraw& draw, bool fallback, bool filled, bool timed, std::uint64_t spentNanoseconds);

}

#endif
