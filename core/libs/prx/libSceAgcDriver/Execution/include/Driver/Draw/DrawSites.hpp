#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWSITES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWSITES_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

enum class DrawSiteRefusal : std::uint8_t { NoSite, Signature, UserWord, Memory, VertexDecode, VertexLayout, Views, Queued, Pending, Moved, NoRecipe, Count };

enum class DrawSiteMiss : std::uint8_t { FrontWords, FrontTables, PixelWords, PixelTables, Count };

enum class DrawSiteExclusion : std::uint8_t { Path, Program, Stage, Patched, Writes, Tables, Callee, Holder, Walk, Count };

struct DrawSiteBase {
    std::uint64_t delta = 0;
    std::uint32_t words = 0;
    std::int8_t base = -1;
    bool operator==(const DrawSiteBase&) const = default;
};

struct DrawSiteRun {
    std::uint64_t address = 0;
    std::uint32_t words = 0;
    std::uint32_t offset = 0;
};

struct DrawSiteStagePlan {
    std::uint32_t userDataBase = 0;
    std::uint32_t resourceRegister = 0;
    std::uint32_t words = 0;
    std::uint64_t frozen = 0;
    std::uint64_t aligned = 0;
    bool nullPixel = false;
    std::vector<DrawSiteBase> based;
    bool operator==(const DrawSiteStagePlan&) const = default;
};

struct DrawSiteStage {
    std::shared_ptr<const DispatchVariant> variant;
    std::vector<std::uint32_t> words;
    std::vector<std::uint32_t> basedOffsets;
    std::vector<DrawSiteRun> fixed;
    std::vector<DrawSiteRun> decoded;
    std::uint64_t wordsSignature = 0;
    std::uint64_t signature = 0;
};

struct RetainedDraw {
    std::vector<DrawSiteStage> stages;
    std::vector<std::shared_ptr<DispatchVariant>> roots;
    std::shared_ptr<DrawEntry> holder;
    std::uint64_t recipeKey = 0;
    std::uint64_t signature = 0;
    mutable std::atomic<std::uint64_t> used{0};
};

struct DrawSitePlan {
    std::vector<DrawSiteStagePlan> stages;
    std::vector<std::shared_ptr<const RetainedDraw>> retained;
};

struct DrawSite {
    std::shared_ptr<const DrawDecode> decode;
    std::vector<DrawSitePlan> plans;
};

struct DrawSiteStagePatch {
    static constexpr std::size_t MaxWords = 40;
    std::array<std::uint32_t, MaxWords> now{};
    std::uint32_t words = 0;
    std::array<std::pair<std::uint32_t, std::uint32_t>, MaxWords> forwarded{};
    std::uint32_t forwardedCount = 0;
    std::array<std::array<std::uint32_t, 3>, 2 * MaxWords> descriptors{};
    std::uint32_t descriptorCount = 0;
    std::array<std::pair<std::uint32_t, ShaderRecompiler::ShaderVertexBufferResource>, ShaderRecompiler::ShaderVertexStageInfo::MaxResources> vertex{};
    std::uint32_t vertexCount = 0;
    ShaderRecompiler::ShaderVertexStageInfo vertexInfo{};
    bool redecoded = false;
};

struct DrawSiteHit {
    static constexpr std::size_t MaxStages = 2;
    std::shared_ptr<const RetainedDraw> retained;
    std::shared_ptr<const DrawRecipe> recipe;
    std::array<DrawSiteStagePatch, MaxStages> stages;
    DrawSiteRefusal refusal = DrawSiteRefusal::NoSite;
    bool served = false;
    bool matched = false;
    std::uint32_t wordsRead = 0;
};

struct DrawSiteOutcome {
    std::shared_ptr<const RetainedDraw> retained;
    DrawSiteRefusal refusal = DrawSiteRefusal::NoSite;
    bool eligible = false;
    bool walked = false;
    bool matched = false;
    bool served = false;
};

struct DrawSiteCounters {
    std::uint64_t draws = 0, ineligible = 0, walks = 0, served = 0, wordsRead = 0, retries = 0;
    std::uint64_t redecoded = 0, rebased = 0, patched = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawSiteRefusal::Count)> refused{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrawSiteMiss::Count)> missed{};
    std::uint64_t combinations = 0, bufferWords = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawSiteExclusion::Count)> excluded{};
    std::uint64_t retained = 0, replaced = 0, kept = 0, dropped = 0, plansDropped = 0, sitesCreated = 0, sitesEvicted = 0;
    std::uint64_t sites = 0, held = 0;
};

struct DrawSiteAccess {
    static bool Enabled();
    static void Walk(Driver& driver, const QueueState& queue, std::uint64_t shape, DrawSiteHit& hit);
    static void NoteIneligible();
    static void Retain(Driver& driver, std::uint64_t shape, const DrawSiteOutcome& walked, const std::shared_ptr<const DrawDecode>& decode, const std::vector<DrawProgram>& programs, const std::vector<std::shared_ptr<DispatchVariant>>& served, const std::vector<std::shared_ptr<DispatchVariant>>& roots, std::uint64_t recipeKey, const std::shared_ptr<DrawEntry>& recipeDonor, bool patchedStages);
    static DrawSiteCounters Counters();
    static std::string Report();
};

}

#endif
