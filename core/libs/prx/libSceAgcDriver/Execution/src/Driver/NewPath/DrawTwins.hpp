#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DRAWTWINS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_SRC_DRIVER_NEWPATH_DRAWTWINS_HPP

#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/PushBlock.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/ResolvedTwins.hpp"
#include "prx/libSceAgcDriver/Execution/src/Driver/NewPath/Router.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ShaderRecompiler {
struct RecompileResult;
enum class ShaderStage;
}

// The draw twins of the new path (lane D-draws, APS5_NEW_PATH_EMIT_DRAWS=1 with APS5_NEW_PATH=1): an
// address-based draw (class A) whose every stage has a resolved twin (newPathResolved, T-res's graphics
// stages) draws with the twins' pipeline over today's set layouts and today's resource build, and a
// 256-byte push block of the 24-byte driver prefix (table root, command record) followed by today's
// assembled block shifted by 24 bytes. These are the pure pieces: shape refusals, the push end and its
// composition, the stage twin comparison.
namespace AgcDriver::NewPath {

struct DrawStage {
    ShaderRecompiler::ShaderStage stage;
    const ShaderRecompiler::RecompileResult* program = nullptr;
    std::uint32_t pushOffset = 0;
};

// What of the draw's pipeline and push block lies outside the resolved prefix's layout.
struct DrawShape {
    bool tessellation = false;
    bool mesh = false;
    bool rectList = false;
    bool depthRemap = false;
    bool stencilReplay = false;
    bool blendedTargets = false;
    bool splitPipeline = false;
};

enum class ShapeRefusal : std::uint8_t { None, Stages, FixedPushWords, StencilReplay, BlendedTargets, PixelLds, BarycentricEmulation, SplitPipeline };
inline constexpr std::size_t ShapeRefusalCount = 8;
[[nodiscard]] const char* ShapeRefusalName(ShapeRefusal refusal);
[[nodiscard]] ShapeRefusal DrawShapeRefusal(std::span<const DrawStage> stages, const DrawShape& shape);

inline constexpr std::uint32_t DrawPrefixBytes = 24;
// The end of today's assembled push block (the last stage's push words), and whether the 24-byte
// prefix fits in front of it (F5 for draws).
[[nodiscard]] std::uint32_t DrawPushEnd(std::span<const DrawStage> stages);
[[nodiscard]] bool DrawPushFits(std::span<const DrawStage> stages);
[[nodiscard]] std::optional<PushBytes> ComposeDrawPush(const std::array<std::byte, Abi::PushBlockBytes>& assembled, std::span<const DrawStage> stages, std::uint64_t tableRoot, std::uint64_t commandRecord);

// TwinDifference plus the fields a graphics stage's module hands the pipeline, the vertex input and
// the push patching (empty: the twin draws as today's stage does).
[[nodiscard]] std::string StageTwinDifference(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);
[[nodiscard]] bool StageTwinCompatible(const ShaderRecompiler::RecompileResult& compiled, const ShaderRecompiler::RecompileResult& twin);

// The twins of the draw's stages in order, or empty when a stage has none.
[[nodiscard]] std::vector<std::shared_ptr<const ShaderRecompiler::RecompileResult>> DrawTwinsOf(std::span<const DrawStage> stages, const ResolvedTwins& twins);

// The graphics stages' twins (apart from the compute twins' Twins()).
ResolvedTwins& DrawStageTwins();

// Draw stage twins ahead of the route (D-combo's TwinRoutes for draws, lane NP-overhead). The front built a stage twin
// for every fresh stage materialization, before the committer's router saw the draw (ab-combo: 3.12 M stage twins in A
// phases for 152,626 emitted draws), most of them for draws that are not class A. DrawRoutes keeps, per draw program key
// (the stages and their code addresses), the routes of its last twinned draws; one decision per draw packet covers all
// its fresh stages (DrawTwinDecision), so a probe builds every stage's twin together. Skipping only sends the draw to
// today's path (reason twin-skipped). APS5_NEW_PATH_TWIN_ALWAYS=1 builds every stage twin as before.
struct DrawProgramStage {
    std::uint32_t stage = 0;
    std::uint64_t codeAddress = 0;
};
[[nodiscard]] std::uint64_t DrawProgramKey(std::span<const DrawProgramStage> stages);
TwinRoutes& DrawRoutes();
// Whether the fresh stages of the draw packet (submission serial, packet offset) build their twins: asked once per packet.
[[nodiscard]] bool DrawTwinDecision(std::uint64_t submission, std::uint64_t packetOffset, std::uint64_t key);
// The draw program keys of the stages that have a twin (at most EmitCandidate::DrawProgramSlots, distinct) and whether a
// stage's twin was skipped; noted on the candidate, whose route is noted in DrawRoutes when it ends.
struct EmitCandidate;
void NoteDrawPrograms(EmitCandidate& candidate, std::span<const ShaderRecompiler::RecompileResult* const> programs);

struct DrawEmitCounts {
    std::array<std::atomic<std::uint64_t>, ShapeRefusalCount> shapes{};
    std::atomic<std::uint64_t> pipelineRequests{0}, pipelineNotReady{0}, pipelineFailed{0}, pipelineRebuilt{0}, uncachedPipelines{0};
    std::array<std::atomic<std::uint64_t>, 4> covers{};
    std::atomic<std::uint64_t> records{0}, tableUploads{0};
    // Class A draws by fact, every fact evaluated (F4 also without a twin when APS5_PROFILE_NEW_PATH is set):
    // the share that would route new once every stage has its twin.
    std::atomic<std::uint64_t> classA{0}, lean{0}, shapeOk{0}, twinned{0}, pushFits{0}, noCloth{0}, covered{0}, eligible{0};
    // D-draws2: deferred-table templates (A -> T): hits, and hits that fell back and took a build of their own.
    std::atomic<std::uint64_t> templateHits{0}, templateFallbacks{0};
};
DrawEmitCounts& DrawCounts();

// D-draws2 (retire step a for draws): the facts of a lean class A draw known before its build (shape from the
// state and the stages, twins, push fit), with the build's and the pipeline's facts (F4, cloth, pipeline)
// taken as passing; empty when a stage has no twin or a twin still binds the BDA table or fault buffer. When the
// router would route these new, the build defers its table and fault buffer.
[[nodiscard]] std::optional<CommandFacts> DrawFactsWithoutTable(const CommandFacts& decoded, std::span<const DrawStage> stages, const DrawShape& shape, const ResolvedTwins& twins);

// The committer's draw (Graphics::Draw) by route, APS5_PROFILE_NEW_PATH with the draw switch on.
enum class DrawCostBucket : std::uint8_t { NewDeferred, NewBuilt, NewTemplate, FallbackClassA, Other, Count };
inline constexpr std::size_t DrawCostBuckets = static_cast<std::size_t>(DrawCostBucket::Count);
[[nodiscard]] bool DrawCostProfiled();
void NoteDrawCost(DrawCostBucket bucket, std::uint64_t drawNs, std::uint64_t buildNs);
[[nodiscard]] std::string FormatDrawCost(const char* window);
[[nodiscard]] std::string FormatDrawEmit(const char* window, bool emitting);

}

#endif
