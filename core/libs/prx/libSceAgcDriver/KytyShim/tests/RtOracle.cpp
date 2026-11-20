#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryOracle.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <spirv-tools/libspirv.hpp>
#include <vector>

namespace {

namespace IR = Libs::Graphics::ShaderRecompiler::IR;
namespace Spirv = Libs::Graphics::ShaderRecompiler::Spirv;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

enum class Node { Children, Uniform, Memory, Constant };
enum class Child { InLane, Ballot, ReadLane, Store };

struct Walk {
    IR::Block block;
    std::vector<IR::Inst*> sites;
};

IR::Inst& Intersection(IR::Block& block, IR::Value node, uint32_t node_words) {
    const IR::Value zero(uint32_t {0});
    const IR::Value one(uint32_t {0x3f800000u});
    auto& descriptor = block.AppendNewInst(IR::ValueOpcode::CompositeConstructU32x4, {zero, zero, zero, zero});
    IR::Value high = node_words == 2 ? IR::Value(uint32_t {0}) : one;
    auto& address = block.AppendNewInst(IR::ValueOpcode::MakeImageAddress,
                                        {node, high, one, zero, zero, zero, one, one, one, one, one, one, one});
    return block.AppendNewInst(IR::ValueOpcode::BvhIntersect, {IR::Value(&descriptor), IR::Value(&address), IR::Value(true)},
                               node_words);
}

void Build(Walk& walk, Node node, Child child, uint32_t node_words, uint32_t sites) {
    auto& block = walk.block;
    auto& phi = block.AppendNewInst(IR::ValueOpcode::Phi);
    phi.AddPhiOperand(&block, IR::Value(uint32_t {5}));
    IR::Value current(&phi);
    if (node == Node::Uniform) current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::ReadFirstLane, {current, IR::Value(true)}));
    if (node == Node::Memory) current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::LoadSharedU32, {current, IR::Value(true)}));
    if (node == Node::Constant) current = IR::Value(uint32_t {5});
    IR::Value last;
    for (uint32_t site = 0; site < sites; ++site) {
        auto& bvh = Intersection(block, current, node_words);
        walk.sites.push_back(&bvh);
        auto& first = block.AppendNewInst(IR::ValueOpcode::CompositeExtractU32x4, {IR::Value(&bvh), IR::Value(uint32_t {0})});
        auto& masked = block.AppendNewInst(IR::ValueOpcode::BitwiseAnd32, {IR::Value(&first), IR::Value(uint32_t {0xfffffff8u})});
        current = IR::Value(&block.AppendNewInst(IR::ValueOpcode::BitwiseOr32, {IR::Value(&masked), IR::Value(uint32_t {5})}));
        last = IR::Value(&first);
    }
    auto& valid = block.AppendNewInst(IR::ValueOpcode::INotEqual32, {last, IR::Value(uint32_t {0xffffffffu})});
    switch (child) {
        case Child::InLane: break;
        case Child::Ballot: block.AppendNewInst(IR::ValueOpcode::Ballot, {IR::Value(&valid)}); break;
        case Child::ReadLane: block.AppendNewInst(IR::ValueOpcode::ReadLane, {last, IR::Value(uint32_t {0})}); break;
        case Child::Store: block.AppendNewInst(IR::ValueOpcode::WriteSharedU32, {IR::Value(uint32_t {0}), last, IR::Value(true)}); break;
    }
    if (node != Node::Constant) phi.AddPhiOperand(&block, current);
}

uint32_t Mark(Node node, Child child, uint32_t node_words = 1, uint32_t sites = 1) {
    Walk walk;
    Build(walk, node, child, node_words, sites);
    const auto flags = IR::MarkOracleWalk(walk.sites);
    for (const auto* bvh: walk.sites) {
        const auto set = bvh->Flags<uint32_t>();
        Expect((set & 0xffu) == node_words, "marking keeps the node word count");
        Expect((set & flags) == flags, "every intersection of the walk carries the walk flags");
    }
    return flags;
}

void EmptyWalks() {
    Walk served, left;
    Build(served, Node::Children, Child::InLane, 1, 1);
    Build(left, Node::Children, Child::InLane, 2, 2);
    IR::MarkOracleWalk(served.sites);
    std::vector<IR::Inst*> all(served.sites);
    all.insert(all.end(), left.sites.begin(), left.sites.end());
    IR::Program program;
    Expect(IR::BvhIntersectEmpty == 1u << 11, "the empty bit is bit 11");
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = false}) == 0, "the emulated variant runs no intersection empty");
    for (const auto* bvh: left.sites) {
        Expect((bvh->Flags<uint32_t>() & IR::BvhIntersectEmpty) == 0, "an emulated intersection keeps its box test");
    }
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = true}) == 2, "the native variant runs every unserved intersection empty");
    for (const auto* bvh: left.sites) {
        const auto set = bvh->Flags<uint32_t>();
        Expect((set & IR::BvhIntersectEmpty) != 0 && (set & IR::BvhIntersectNative) == 0, "an unserved intersection is empty, not native");
        Expect((set & 0xffu) == 2, "an empty intersection keeps its node word count");
    }
    for (const auto* bvh: served.sites) {
        Expect((bvh->Flags<uint32_t>() & IR::BvhIntersectEmpty) == 0, "an intersection the oracle serves is not empty");
    }
    Expect(IR::MarkEmptyWalk(program, all, {.rewrite = true}) == 0, "marking again adds nothing");
}

bool Lane(uint32_t flags) {
    return (flags & IR::BvhIntersectOracleLane) != 0;
}

void Marking() {
    const auto flags = Mark(Node::Children, Child::InLane);
    Expect((flags & IR::BvhIntersectNative) != 0 && (flags & IR::BvhIntersectOracle) != 0, "a marked walk is native and served by the oracle");
    Expect(IR::MarkOracleWalk({}) == 0u, "an empty walk marks nothing");
    Expect(IR::OracleRemaining == 1u && IR::OracleForms == 2u && IR::OracleOnly == 4u && IR::OraclePixel == 8u, "the oracle bits match KYTY_RT_ORACLE");
}

void Modes() {
    Expect(Lane(Mark(Node::Children, Child::InLane)), "a node fed only by the walk's own children walks per lane");
    Expect(Lane(Mark(Node::Children, Child::InLane, 2)), "a 64-bit node pointer fed by its children walks per lane");
    Expect(Lane(Mark(Node::Children, Child::InLane, 1, 2)), "two sites that feed each other walk per lane");
    Expect(!Lane(Mark(Node::Uniform, Child::InLane)), "a node read from the first lane walks the whole wave");
    Expect(!Lane(Mark(Node::Uniform, Child::InLane, 2)), "a 64-bit node read from the first lane walks the whole wave");
    Expect(!Lane(Mark(Node::Memory, Child::InLane)), "a node loaded from memory keeps the wave mode");
    Expect(!Lane(Mark(Node::Constant, Child::InLane)), "a constant node keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::Ballot)), "a child tested across lanes keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::ReadLane)), "a child read from another lane keeps the wave mode");
    Expect(!Lane(Mark(Node::Children, Child::Store)), "a child stored to memory keeps the wave mode");
}

void NodeCountFacts() {
    constexpr uint32_t Tlas = 1u << 29u;
    constexpr uint32_t Blas = 1u << 31u;
    IR::RayTracingLayout layout;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == 0, "both node counts at 88 by default");
    layout.node_count_80 = true;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == (Tlas | Blas), "an unset BLAS fact follows the TLAS one");
    layout.blas_node_count_80 = false;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == Tlas, "TLAS count at 80, BLAS count at 88");
    layout.node_count_80      = false;
    layout.blas_node_count_80 = true;
    Expect((layout.Pack()[3] & (Tlas | Blas)) == Blas, "TLAS count at 88, BLAS count at 80");
    auto shaped = layout;
    shaped.blas_node_count_80.reset();
    Expect(shaped.Pack()[0] == layout.Pack()[0] && shaped.Pack()[1] == layout.Pack()[1] &&
               shaped.Pack()[2] == layout.Pack()[2] && ((shaped.Pack()[3] ^ layout.Pack()[3]) & ~Blas) == 0,
           "the BLAS fact changes bit 31 only");
}

void TriangleIndexSpirv() {
    namespace Emit = Spirv::Emitter;
    const IR::Value unused(0u);
    IR::Block alpha;
    alpha.AppendNewInst(IR::ValueOpcode::RayQueryAlpha,
                        {unused, unused, unused, unused, unused, unused, unused,
                         unused, unused, unused, unused, unused, unused, unused});
    IR::Program program;
    program.blocks.push_back(&alpha);
    program.stage = Libs::Graphics::ShaderType::Compute;
    program.lane_local = true;
    program.info.uses_dma = true;
    program.info.ray_query = true;
    const Libs::Graphics::ShaderComputeInputInfo compute {};
    const Spirv::Profile profile {.ray_query = true};
    Emit::EmitterState state(program, {.compute = &compute}, profile);
    Emit::ValueEmitContext ctx(state);
    Emit::DefineModule(state);
    Emit::DefineGetBdaPointer(state);
    Emit::DefineIndirectBufferLoads(ctx);
    auto& builder = state.builder;
    builder.AddFunction(spv::OpFunction, Emit::TypeVoid(state), state.main_func,
                        spv::FunctionControlMaskNone, Emit::TypeFunction(state));
    Emit::EmitLabel(state, state.entry_label);
    const auto zero = Emit::ConstantU32(state, 0);
    const auto indices = Emit::EmitTriangleIndices(ctx, {zero, zero, zero, zero}, zero);
    Expect(indices[0] != 0 && indices[1] != 0 && indices[2] != 0, "the triangle load emits three indices");
    builder.AddFunction(spv::OpReturn);
    builder.AddFunction(spv::OpFunctionEnd);
    Emit::DefineBdaLoads(state);
    builder.AddEntryPoint(spv::ExecutionModelGLCompute, state.main_func, "main", state.interface_variables);

    spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_3);
    validator.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        std::fprintf(stderr, "%s\n", message);
    });
    Expect(validator.Validate(builder.Build()), "the ray-query triangle index load emits valid SPIR-V");
}

}

int main() {
    Marking();
    EmptyWalks();
    Modes();
    NodeCountFacts();
    TriangleIndexSpirv();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim ray tracing oracle: all checks passed\n");
    return 0;
}
