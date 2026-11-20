#include "graphics/host_gpu/renderer/pipeline/pipelineChainThreads.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <thread>
#include <vector>

namespace {

namespace Gfx = Libs::Graphics;
namespace IR  = Libs::Graphics::ShaderRecompiler::IR;

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_plan_scratch_slots: %s\n", what);
    std::abort();
}

struct Seen {
    const void* srt_scratch   = nullptr;
    const void* material_keys = nullptr;
    const void* selector_keys = nullptr;
    size_t      srt_words     = 0;
};

Seen Look(const IR::ResourcePlan& plan) {
    const auto scratch = IR::PlanScratch(plan);
    return {&scratch.srt_scratch, &scratch.material_keys, &scratch.selector_keys, scratch.srt_scratch.size()};
}

}

bool Is(const Gfx::PipelineChainThreadSetting& setting, uint32_t threads, bool capped) {
    return setting.threads == threads && setting.capped == capped;
}

int main() {
    Require(Is(Gfx::PipelineChainThreadCount(nullptr), 4, false), "4 pipeline chain threads by default");
    Require(Is(Gfx::PipelineChainThreadCount("16"), 16, false), "16 pipeline chain threads when asked");
    Require(Is(Gfx::PipelineChainThreadCount("32"), Gfx::MaxPipelineChainThreads, true),
            "32 pipeline chain threads are capped to the most there are scratch slots for, and the cap is reported");
    Require(Gfx::PipelineChainThreadCount("32").requested == 32, "the capped setting keeps the value asked for, for the log line");
    Require(Is(Gfx::PipelineChainThreadCount("100"), Gfx::MaxPipelineChainThreads, true), "100 is capped as well");
    Require(Is(Gfx::PipelineChainThreadCount("0"), 1, false), "0 runs one pipeline chain thread");
    Require(Is(Gfx::PipelineChainThreadCount("many"), 1, false), "text reads as 0 and runs one pipeline chain thread");

    IR::ResourcePlan plan;
    plan.srt_scratch.assign(24, 0u);
    plan.selector_keys.resize(3);
    const auto shared = Look(plan);
    Require(shared.srt_scratch == &plan.srt_scratch && shared.material_keys == &plan.material_keys &&
                shared.selector_keys == &plan.selector_keys,
            "a thread without a slot uses the plan's own scratch");
    Require(IR::ResourcePlan::ThreadScratchSlots == IR::ResourcePlan::FirstChainScratchSlot + IR::ResourcePlan::ChainScratchSlots,
            "every pipeline chain thread has a scratch slot after the ahead and execute slots");
    Require(IR::ResourcePlan::ChainScratchSlots >= Gfx::MaxPipelineChainThreads,
            "a scratch slot for each pipeline chain thread the setting allows");

    std::vector<Seen>        seen(Gfx::MaxPipelineChainThreads);
    std::vector<std::thread> chains;
    for (uint32_t index = 0; index < Gfx::MaxPipelineChainThreads; ++index) {
        chains.emplace_back([&, index] {
            IR::UsePlanScratchSlot(IR::ResourcePlan::FirstChainScratchSlot + index);
            seen[index] = Look(plan);
        });
    }
    for (auto& chain: chains) chain.join();

    std::set<const void*> distinct {shared.srt_scratch, shared.material_keys, shared.selector_keys};
    for (const auto& one: seen) {
        Require(one.srt_words == plan.srt_scratch.size(), "a chain thread's scratch is sized like the plan's");
        distinct.insert(one.srt_scratch);
        distinct.insert(one.material_keys);
        distinct.insert(one.selector_keys);
    }
    Require(distinct.size() == 3u * (seen.size() + 1u),
            "no two pipeline chain threads, and no chain thread and the plan, share scratch");
    std::puts("kyty_shim_plan_scratch_slots: ok");
    return 0;
}
