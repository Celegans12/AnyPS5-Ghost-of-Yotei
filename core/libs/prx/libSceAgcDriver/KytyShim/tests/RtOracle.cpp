#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryOracle.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

namespace IR = Libs::Graphics::ShaderRecompiler::IR;

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

bool Lane(uint32_t flags) {
    return (flags & IR::BvhIntersectOracleLane) != 0;
}

void Marking() {
    const auto flags = Mark(Node::Children, Child::InLane);
    Expect((flags & IR::BvhIntersectNative) != 0 && (flags & IR::BvhIntersectOracle) != 0, "a marked walk is native and served by the oracle");
    Expect(IR::MarkOracleWalk({}) == 0u, "an empty walk marks nothing");
    Expect(IR::OracleRemaining == 1u && IR::OracleForms == 2u && IR::OracleOnly == 4u, "the oracle bits match KYTY_RT_ORACLE");
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

}

int main() {
    Marking();
    Modes();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim ray tracing oracle: all checks passed\n");
    return 0;
}
