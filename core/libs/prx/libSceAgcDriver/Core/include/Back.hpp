#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BACK_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BACK_HPP

#include "prx/libSceAgcDriver/Core/include/DrawMemo.hpp"
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
struct CoreReplay;
DrawRecipeOutcome DrawCoreReplayed(const Context& context, const State& state, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders, const std::shared_ptr<const DrawRecipe>& recipe, bool verify, CoreReplay& replay, bool inputsCurrent, bool verifyReplay, std::uint64_t identity);
CoreReplay* NewCoreReplay();
void DeleteCoreReplay(CoreReplay* replay) noexcept;
bool CoreReplayStorable(const CoreReplay& replay);
std::uint32_t CoreReplayRanges(const CoreReplay& replay);
void CoreReplayRange(const CoreReplay& replay, std::uint32_t index, std::uint64_t& address, std::uint64_t& bytes, GuestMemory::ContentToken& token);
bool CoreReplayMatches(const CoreReplay& replay, VkDevice device, std::uint64_t identity, const DrawRecipe* recipe, const Pm4::DrawParameters& draw, std::span<const CompiledShader> shaders);
bool CoreReplayRecaptured(const CoreReplay& replay);
bool CoreReplayLean(const CoreReplay& replay);
std::array<std::uint64_t, 8> CoreReplayCounted();
std::array<std::uint64_t, 12> CoreReplayLeanRefusals();

}

namespace AgcDriver::Core {

enum class BackMode : std::uint8_t { Recipe, Light, Verify };

[[nodiscard]] BackMode BackModeNow();
void SetBackModeForTest(int mode);
[[nodiscard]] std::string BackReport();
[[nodiscard]] std::span<const Graphics::CompiledShader> PrepareBack(CoreDraw& record);
[[nodiscard]] bool RecordBack(VulkanDevice& device, const Graphics::State& graphics, const Pm4::DrawParameters& parameters, std::span<const Graphics::CompiledShader> stages, const std::shared_ptr<const DrawRecipe>& recipe, std::uint64_t identity = 0);
enum class BackReplay : std::uint8_t { Off, On, Verify };
[[nodiscard]] BackReplay BackReplayNow();
void SetBackReplayForTest(int mode);
[[nodiscard]] std::string BackReplayReport();
void ReportBack();
[[nodiscard]] bool ReplayStats();
[[nodiscard]] bool ReplaySampled(const FrontVariant& variant);
[[nodiscard]] ReplayDraw ReplayDrawOf(const CoreDraw& record, std::span<const Graphics::CompiledShader> stages);
[[nodiscard]] const ReplayDraw& ReplayTargetsOf(const FrontVariant& variant, const Graphics::State& graphics);
[[nodiscard]] bool ReplayFilled(const ReplayDraw& targets);
[[nodiscard]] bool ReplayTimedNext();
void NoteReplay(ReplayDraw* draw, const ReplayDraw& targets, bool fallback, bool filled, bool timed, std::uint64_t spentNanoseconds);

}

#endif
