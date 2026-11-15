#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_RUNPLAN_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_RUNPLAN_HPP

#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "Optimization/BlockLayout.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ShaderRecompiler::Detail {

class SrtTape;

}

namespace AgcDriver::Core {

struct StagePlan {
    std::shared_ptr<const ShaderRecompiler::IrResourcePlan> plan;
    ShaderRecompiler::BlockLayout layout;
    const ShaderRecompiler::Detail::SrtTape* tape = nullptr;
    std::uint64_t variantKey = 0;
    std::uint32_t userDataBase = 0;
    std::uint32_t pushOffsetBytes = 0;
};

struct RunPlan {
    StateKey key;
    std::uint64_t id = 0;
    std::array<StagePlan, 4> stages;
    std::uint32_t stageCount = 0;
    std::shared_ptr<void> pipeline;
    std::shared_ptr<void> resources;
    std::uint64_t accessSummary = 0;
    std::int32_t vertexOffsetRegister = -1;
    std::int32_t instanceOffsetRegister = -1;
};

class RunPlans {
public:
    explicit RunPlans(std::size_t slots = 1u << 14u);
    ~RunPlans();
    RunPlans(const RunPlans&) = delete;
    RunPlans& operator=(const RunPlans&) = delete;
    [[nodiscard]] const RunPlan* Find(const StateKey& key) const;
    const RunPlan* Teach(RunPlan plan);
    void Forget(std::uint64_t program);
    struct Counts { std::uint64_t finds = 0, hits = 0, taught = 0, evicted = 0, forgotten = 0; };
    [[nodiscard]] Counts Counted() const;

private:
    struct State;
    std::unique_ptr<State> state;
};

}

#endif
