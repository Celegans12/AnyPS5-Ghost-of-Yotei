#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_FRONT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_FRONT_HPP

#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include "prx/libSceAgcDriver/Core/include/Op.hpp"
#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Recipe.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "Optimization/BlockLayout.hpp"
#include "Recompiler.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ShaderRecompiler::Detail {

class SrtTape;

}

namespace AgcDriver::Core {

enum class FrontLeft : std::uint32_t { Entry, NoPlan, Programs, UserWords, Walk, NoVariant, VariantKey, Views, Vertex, Queued, Pending, Moved, Stale, Finisher, Nothing, Precheck, Ring, Ranges, Count };
enum class TeachRefusal : std::uint32_t { None, Path, Program, Stage, Plan, Tape, Walk, Words, Buffers, Forwarded, Vertex, Writes, PlanDiffers, Count };

const char* FrontLeftName(FrontLeft reason);
const char* TeachRefusalName(TeachRefusal reason);

struct StageWalk {
    std::vector<ShaderRecompiler::DescriptorValue> results;
    std::vector<std::uint32_t> flat;
    std::vector<std::uint8_t> active;
};

struct PlanStage {
    std::shared_ptr<const ShaderRecompiler::IrResourcePlan> plan;
    const ShaderRecompiler::Detail::SrtTape* tape = nullptr;
    std::vector<std::int32_t> bufferResult;
    std::vector<std::uint8_t> bufferMask;
    std::vector<std::uint8_t> keyResult;
    std::uint32_t userWords = 0;
    bool vertexInputs = false;
    std::uint32_t flatWords = 0;
};

struct RunPlanData {
    std::shared_ptr<const DriverDetail::DrawDecode> decode;
    std::array<PlanStage, 4> stages;
    std::uint32_t stageCount = 0;
    std::uint64_t id = 0;
    mutable std::uint8_t precheck = 0;
};

struct VariantStage {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> base;
    std::shared_ptr<const std::vector<ShaderRecompiler::RegisteredImageView>> registeredViews;
    StageWalk taught;
    std::uint32_t flatBinding = 0xffffffffu;
    std::uint32_t buffersBinding = 0xffffffffu;
    std::uint64_t variantKey = 0;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertexInfo;
    std::uint32_t pushOffset = 0;
};

struct FrontVariant {
    std::shared_ptr<const RunPlanData> plan;
    std::array<VariantStage, 4> stages;
    std::shared_ptr<const DrawRecipe> recipe;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    std::uint64_t shape = 0;
    std::array<std::uint64_t, 2> digest{};
    std::atomic<bool> stale{false};
    std::array<ShaderRecompiler::RecompileResult, 4> frontResults;
    std::array<ShaderRecompiler::RecompileResult, 4> backResults;
    std::vector<Graphics::CompiledShader> frontStages;
    std::vector<Graphics::CompiledShader> backStages;
};

struct CoreDraw {
    std::shared_ptr<FrontVariant> variant;
    Pm4::DrawParameters parameters{};
    std::array<StageWalk, 4> walks;
    std::array<std::vector<std::uint32_t>, 4> userData;
    std::array<ShaderRecompiler::ShaderVertexStageInfo, 4> vertex{};
    std::uint64_t epoch = 0;
    std::uint32_t queue = 0;
    std::uint32_t opcode = 0;
};

struct FrontOutcome {
    bool taken = false;
    bool teach = false;
    StateKey key;
    std::shared_ptr<CoreDraw> candidate;
    bool looked = false;
    const std::byte* op = nullptr;
    FrontVariant* variant = nullptr;
    std::span<const std::pair<std::uint64_t, std::uint64_t>> reads;
};

struct FlatInputs {
    const QueueState* queue = nullptr;
    const DriverDetail::ShaderRegistry* shaders = nullptr;
    std::span<const std::uint32_t> packet;
    const Pm4::DrawParameters* parameters = nullptr;
    std::uint64_t device = 0;
    std::uint64_t epoch = 0;
    bool verifyOnly = false;
    bool (*plain)(void* context, const QueueState& queue, const Pm4::DrawParameters& parameters) = nullptr;
    void* context = nullptr;
};

struct FlatDraw {
    Op op;
    FrontVariant* variant = nullptr;
    std::array<std::span<const std::uint32_t>, 4> section;
    std::array<std::span<const std::uint32_t>, 4> buffers;
    std::array<std::span<const std::uint32_t>, 4> userData;
    std::span<const std::uint32_t> vertex;
};

struct TeachInputs {
    StateKey key;
    std::shared_ptr<const DriverDetail::DrawDecode> decode;
    std::span<const DriverDetail::DrawProgram> programs;
    std::span<const Graphics::CompiledShader> stages;
    std::span<const std::shared_ptr<DriverDetail::DispatchVariant>> served;
    std::shared_ptr<const DrawRecipe> recipe;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    std::uint64_t shape = 0;
};

struct FrontCounts {
    std::uint64_t draws = 0;
    std::uint64_t continued = 0;
    std::uint64_t plans = 0;
    std::uint64_t taken = 0;
    std::uint64_t takenContinued = 0;
    std::uint64_t taught = 0;
    std::uint64_t verified = 0;
    std::uint64_t mismatches = 0;
    std::uint64_t digests = 0;
    std::uint64_t committed = 0;
    std::uint64_t restarted = 0;
    std::uint64_t walkNs = 0;
    std::uint64_t frontNs = 0;
    std::uint64_t backNs = 0;
    std::uint64_t retries = 0;
    std::uint64_t flatTaken = 0;
    std::uint64_t flatSame = 0;
    std::uint64_t flatNs = 0;
    std::uint64_t flatBytes = 0;
    std::uint64_t flatLost = 0;
    std::uint64_t flatTimed = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(FrontLeft::Count)> left{};
    std::array<std::uint64_t, static_cast<std::size_t>(TeachRefusal::Count)> refused{};
};

class Front {
public:
    static Front& Get();
    static FrontCounts Counted();
    static std::string Report();
    static void NoteCommitted(bool restarted, std::uint64_t nanoseconds);
    static void NoteEnqueued(std::uint64_t nanoseconds);

    [[nodiscard]] StateKey KeyOf(const QueueState& queue, const Pm4::DrawParameters& parameters, std::uint32_t opcode, std::uint64_t device);
    void VerifyDigests(const QueueState& queue);
    [[nodiscard]] FrontOutcome Walk(const QueueState& queue, const DriverDetail::ShaderRegistry& shaders, const Pm4::DrawParameters& parameters, std::uint32_t opcode, std::uint64_t device, bool verifyOnly);
    void Teach(const TeachInputs& inputs);
    [[nodiscard]] std::string Compare(const CoreDraw& candidate, const DriverDetail::DrawDecode& decode, std::span<const Graphics::CompiledShader> stages, const DrawRecipe* recipe) const;
    void NoteVerified(bool mismatch);
    void Release(std::shared_ptr<CoreDraw> record);
    [[nodiscard]] FrontOutcome Flat(const FlatInputs& inputs);
    void FlatPosted(const std::byte* op);
    [[nodiscard]] static FlatDraw FlatRead(const std::byte* op);
    void FlatEmitted(const FlatDraw& draw);
    static void FlatPatch(const FlatDraw& draw, std::size_t stage, ShaderRecompiler::RecompileResult& out);

    static void Patch(const RunPlanData& plan, std::size_t stage, const VariantStage& taught, const StageWalk& walk, std::span<const std::uint32_t> userData, const ShaderRecompiler::ShaderVertexStageInfo* vertex, ShaderRecompiler::RecompileResult& out);

private:
    struct State;
    Front();
    std::unique_ptr<State> state;
};

}

#endif
