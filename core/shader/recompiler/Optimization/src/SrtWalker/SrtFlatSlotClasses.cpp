#include "Optimization/SrtWalker/SrtFlatSlotClasses.hpp"
#include "Optimization/SrtWalker/SrtInstructionPredicates.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace ShaderRecompiler::Detail {

std::vector<std::uint8_t> ComputePureFlatSlots(const IrResourcePlan& plan) {
    std::vector<std::uint8_t> pure(plan.srtReads.size(), 1u);
    if (pure.empty()) return pure;
    const bool disqualified = !plan.srtPlanComplete || plan.requiresSpecializationMemory
        || std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value(); })
        || (plan.uniformFill.fill.kind != UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size());
    if (disqualified) {
        std::fill(pure.begin(), pure.end(), 0u);
        return pure;
    }
    // The raw read node of each slot: a descriptor source dword may point at it directly (the
    // tracker resolved the handle argument before the plan builder patched the operands).
    std::unordered_map<const IrValue*, std::uint32_t> readSlots;
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value != nullptr) readSlots.emplace(plan.srtReads[slot].value->Resolve(), slot);
    }
    std::unordered_set<const IrValue*> visited;
    std::vector<IrValue*> pending;
    const auto mark = [&](IrValue* root) {
        if (root == nullptr) return;
        pending.push_back(root);
        while (!pending.empty()) {
            IrValue* value = pending.back()->Resolve();
            pending.pop_back();
            if (!visited.insert(value).second) continue;
            if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
                const IrValue* slot = value->Argument(1)->Resolve();
                if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < pure.size()) {
                    const auto index = slot->ImmediateU32();
                    pure[index] = 0u;
                    pending.push_back(plan.srtReads[index].value);
                }
            }
            if (const auto found = readSlots.find(value); found != readSlots.end()) pure[found->second] = 0u;
            for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
                if (value->Argument(i) != nullptr) pending.push_back(value->Argument(i));
            }
        }
    };
    for (const auto& source : plan.descriptorSources) {
        for (std::uint32_t i = 0; i < source.dwordCount && i < source.dwords.size(); ++i) mark(source.dwords[i]);
    }
    for (const auto& block : plan.controlFlow) mark(block.condition);
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        for (std::uint32_t i = 0; i < plan.uniformFill.fill.words; ++i) mark(plan.uniformFill.values[i]);
    }
    // The address cones: the arguments of every slot's raw read, never the read itself.
    for (const auto& read : plan.srtReads) {
        if (read.value == nullptr) continue;
        IrValue* raw = read.value->Resolve();
        for (std::size_t i = 0; i < raw->ArgumentCount(); ++i) mark(raw->Argument(i));
    }
    return pure;
}

std::uint32_t DescriptorDwordSlot(const IrResourcePlan& plan, std::size_t source, std::uint32_t dword) {
    if (source >= plan.descriptorSources.size()) return NoDescriptorDwordSlot;
    const auto& dwords = plan.descriptorSources[source];
    if (dword >= dwords.dwordCount || dword >= dwords.dwords.size() || dwords.dwords[dword] == nullptr) return NoDescriptorDwordSlot;
    const IrValue* value = dwords.dwords[dword]->Resolve();
    if (value->Opcode() != IrOpcode::ReadConst || value->ArgumentCount() < 2 || value->Argument(1) == nullptr) return NoDescriptorDwordSlot;
    const IrValue* slot = value->Argument(1)->Resolve();
    if (!slot->HasImmediate() || slot->Type() != IrType::U32 || slot->ImmediateU32() >= plan.srtReads.size() || plan.srtReads[slot->ImmediateU32()].value == nullptr) return NoDescriptorDwordSlot;
    return slot->ImmediateU32();
}

std::vector<std::uint8_t> ComputeDescriptorLeafDwords(const IrResourcePlan& plan) {
    std::vector<std::uint8_t> leaves(plan.descriptorSources.size(), 0u);
    if (leaves.empty() || plan.info.buffers.empty()) return leaves;
    const bool disqualified = !plan.srtPlanComplete || plan.requiresSpecializationMemory
        || std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value(); })
        || (plan.uniformFill.fill.kind != UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size());
    if (disqualified) return leaves;
    std::vector<std::uint8_t> buffer(leaves.size(), 0u);
    for (const auto& resource : plan.info.buffers) {
        if (resource.source < buffer.size()) buffer[resource.source] = 1u;
    }
    for (const auto& resource : plan.info.buffers) {
        if (resource.source < buffer.size() && resource.imageAlias != BufferResource::NoImageAlias) buffer[resource.source] = 0u;
    }
    for (const auto& image : plan.info.images) {
        if (image.source < buffer.size()) buffer[image.source] = 0u;
    }
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.source < buffer.size()) buffer[sampler.source] = 0u;
    }
    std::unordered_map<const IrValue*, std::uint32_t> roots;
    std::vector<std::uint32_t> slotRoots(plan.srtReads.size(), 0u);
    for (std::size_t source = 0; source < plan.descriptorSources.size(); ++source) {
        const auto& dwords = plan.descriptorSources[source];
        for (std::uint32_t i = 0; i < dwords.dwordCount && i < dwords.dwords.size(); ++i) {
            if (dwords.dwords[i] == nullptr) continue;
            ++roots[dwords.dwords[i]->Resolve()];
            if (const auto slot = DescriptorDwordSlot(plan, source, i); slot != NoDescriptorDwordSlot) ++slotRoots[slot];
        }
    }
    const auto clean = [&](std::uint32_t slot) {
        const auto flat = plan.srtReads[slot].flatOffset;
        return (slot < plan.cleanFlatSlots.size() && plan.cleanFlatSlots[slot] != 0u) || (flat < plan.cleanFlatSlots.size() && plan.cleanFlatSlots[flat] != 0u);
    };
    const auto candidate = [&](std::size_t source, std::uint32_t dword) {
        if (buffer[source] == 0u || dword >= DescriptorLeafDwordLimit) return false;
        const auto slot = DescriptorDwordSlot(plan, source, dword);
        if (slot == NoDescriptorDwordSlot || slotRoots[slot] != 1u || clean(slot) || roots[plan.descriptorSources[source].dwords[dword]->Resolve()] != 1u) return false;
        const IrValue* read = plan.srtReads[slot].value->Resolve();
        return read->Opcode() == IrOpcode::LoadAddressU32 || read->Opcode() == IrOpcode::ReadConstBuffer;
    };
    std::unordered_set<const IrValue*> consumed;
    std::vector<IrValue*> pending;
    const auto drain = [&] {
        while (!pending.empty()) {
            IrValue* value = pending.back()->Resolve();
            pending.pop_back();
            if (!consumed.insert(value).second) continue;
            if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
                const IrValue* slot = value->Argument(1)->Resolve();
                if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < plan.srtReads.size() && plan.srtReads[slot->ImmediateU32()].value != nullptr) pending.push_back(plan.srtReads[slot->ImmediateU32()].value);
            }
            for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
                if (value->Argument(i) != nullptr) pending.push_back(value->Argument(i));
            }
        }
    };
    const auto consume = [&](IrValue* root) {
        if (root == nullptr) return;
        pending.push_back(root);
        drain();
    };
    const auto consumeBelow = [&](IrValue* root) {
        if (root == nullptr) return;
        IrValue* value = root->Resolve();
        for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
            if (value->Argument(i) != nullptr) pending.push_back(value->Argument(i));
        }
        drain();
    };
    for (std::size_t source = 0; source < plan.descriptorSources.size(); ++source) {
        const auto& dwords = plan.descriptorSources[source];
        for (std::uint32_t i = 0; i < dwords.dwordCount && i < dwords.dwords.size(); ++i) {
            if (!candidate(source, i)) consume(dwords.dwords[i]);
        }
    }
    for (const auto& block : plan.controlFlow) consume(block.condition);
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        for (std::uint32_t i = 0; i < plan.uniformFill.fill.words; ++i) consume(plan.uniformFill.values[i]);
    }
    for (const auto& read : plan.srtReads) consumeBelow(read.value);
    for (std::size_t source = 0; source < plan.descriptorSources.size(); ++source) {
        const auto& dwords = plan.descriptorSources[source];
        for (std::uint32_t i = 0; i < dwords.dwordCount && i < DescriptorLeafDwordLimit; ++i) {
            if (!candidate(source, i)) continue;
            const auto slot = DescriptorDwordSlot(plan, source, i);
            if (!consumed.contains(dwords.dwords[i]->Resolve()) && !consumed.contains(plan.srtReads[slot].value->Resolve())) leaves[source] |= static_cast<std::uint8_t>(1u << i);
        }
    }
    return leaves;
}

std::vector<std::uint32_t> ComputeDescriptorUserWords(const IrResourcePlan& plan) {
    std::vector<std::uint32_t> words;
    if (plan.descriptorSources.empty() || plan.info.buffers.empty()) return words;
    const bool disqualified = !plan.srtPlanComplete || plan.requiresSpecializationMemory
        || std::any_of(plan.descriptorSources.begin(), plan.descriptorSources.end(), [](const DescriptorSource& source) { return source.tableColumn.has_value(); })
        || (plan.uniformFill.fill.kind != UniformFillKind::None && plan.uniformFill.fill.words > plan.uniformFill.values.size());
    if (disqualified) return words;
    std::vector<std::uint8_t> buffer(plan.descriptorSources.size(), 0u);
    for (const auto& resource : plan.info.buffers) {
        if (resource.source < buffer.size()) buffer[resource.source] = 1u;
    }
    for (const auto& resource : plan.info.buffers) {
        if (resource.source < buffer.size() && resource.imageAlias != BufferResource::NoImageAlias) buffer[resource.source] = 0u;
    }
    for (const auto& image : plan.info.images) {
        if (image.source < buffer.size()) buffer[image.source] = 0u;
    }
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.source < buffer.size()) buffer[sampler.source] = 0u;
    }
    constexpr std::uint32_t NoWord = ~std::uint32_t{0};
    const auto userWord = [&](const IrValue* value) {
        if (value->Opcode() != IrOpcode::GetUserData || value->ArgumentCount() < 1 || value->Argument(0) == nullptr) return NoWord;
        const auto reg = RegIndex(static_cast<ScalarReg>(value->Argument(0)->Register().index));
        return reg >= plan.userDataBase && reg - plan.userDataBase < plan.userDataCount ? reg - plan.userDataBase : NoWord;
    };
    std::unordered_map<const IrValue*, std::uint32_t> roots;
    for (const auto& source : plan.descriptorSources) {
        for (std::uint32_t i = 0; i < source.dwordCount && i < source.dwords.size(); ++i) {
            if (source.dwords[i] != nullptr) ++roots[source.dwords[i]->Resolve()];
        }
    }
    const auto candidate = [&](std::size_t source, std::uint32_t dword) {
        if (buffer[source] == 0u || dword >= DescriptorLeafDwordLimit || plan.descriptorSources[source].dwords[dword] == nullptr) return false;
        const IrValue* value = plan.descriptorSources[source].dwords[dword]->Resolve();
        return userWord(value) != NoWord && roots[value] == 1u;
    };
    std::unordered_set<const IrValue*> consumed;
    std::unordered_map<std::uint32_t, std::unordered_set<const IrValue*>> readers;
    std::vector<IrValue*> pending;
    const auto drain = [&] {
        while (!pending.empty()) {
            IrValue* value = pending.back()->Resolve();
            pending.pop_back();
            if (!consumed.insert(value).second) continue;
            if (const auto word = userWord(value); word != NoWord) readers[word].insert(value);
            if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
                const IrValue* slot = value->Argument(1)->Resolve();
                if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < plan.srtReads.size() && plan.srtReads[slot->ImmediateU32()].value != nullptr) pending.push_back(plan.srtReads[slot->ImmediateU32()].value);
            }
            for (std::size_t i = 0; i < value->ArgumentCount(); ++i) {
                if (value->Argument(i) != nullptr) pending.push_back(value->Argument(i));
            }
        }
    };
    const auto consume = [&](IrValue* root) {
        if (root == nullptr) return;
        pending.push_back(root);
        drain();
    };
    for (std::size_t source = 0; source < plan.descriptorSources.size(); ++source) {
        const auto& dwords = plan.descriptorSources[source];
        for (std::uint32_t i = 0; i < dwords.dwordCount && i < dwords.dwords.size(); ++i) {
            if (candidate(source, i)) readers[userWord(dwords.dwords[i]->Resolve())].insert(dwords.dwords[i]->Resolve());
            else consume(dwords.dwords[i]);
        }
    }
    for (const auto& block : plan.controlFlow) consume(block.condition);
    if (plan.uniformFill.fill.kind != UniformFillKind::None) {
        for (std::uint32_t i = 0; i < plan.uniformFill.fill.words; ++i) consume(plan.uniformFill.values[i]);
    }
    for (const auto& read : plan.srtReads) consume(read.value);
    for (std::size_t source = 0; source < plan.descriptorSources.size(); ++source) {
        const auto& dwords = plan.descriptorSources[source];
        for (std::uint32_t i = 0; i < dwords.dwordCount && i < DescriptorLeafDwordLimit; ++i) {
            if (!candidate(source, i)) continue;
            const IrValue* value = dwords.dwords[i]->Resolve();
            const auto word = userWord(value);
            if (consumed.contains(value) || readers[word].size() != 1u) continue;
            words.push_back(static_cast<std::uint32_t>(source) * 8u + i);
            words.push_back(word);
        }
    }
    return words;
}

std::vector<std::uint32_t> ComputeGuardedFlatSlots(const IrResourcePlan& plan) {
    std::unordered_map<const IrValue*, std::vector<const IrValue*>> users;
    std::unordered_set<const IrValue*> seen;
    std::vector<const IrValue*> pending;
    std::vector<const IrValue*> poisonable;
    for (const auto& read : plan.srtReads) {
        if (read.value != nullptr) pending.push_back(read.value->Resolve());
    }
    while (!pending.empty()) {
        const IrValue* value = pending.back();
        pending.pop_back();
        if (!seen.insert(value).second) continue;
        const IrValue* handle = value->ArgumentCount() != 0u && value->Argument(0) != nullptr ? value->Argument(0)->Resolve() : nullptr;
        if (IsRawRead(plan, *value) && handle != nullptr && LoadedFromMemory(*handle)) poisonable.push_back(value);
        const auto follow = [&](const IrValue* argument) {
            if (argument == nullptr) return;
            argument = argument->Resolve();
            users[argument].push_back(value);
            pending.push_back(argument);
        };
        for (std::size_t i = 0; i < value->ArgumentCount(); ++i) follow(value->Argument(i));
        if (value->Opcode() == IrOpcode::ReadConst && value->ArgumentCount() >= 2 && value->Argument(1) != nullptr) {
            const IrValue* slot = value->Argument(1)->Resolve();
            if (slot->HasImmediate() && slot->Type() == IrType::U32 && slot->ImmediateU32() < plan.srtReads.size()) follow(plan.srtReads[slot->ImmediateU32()].value);
        }
    }
    std::unordered_set<const IrValue*> reaching(poisonable.begin(), poisonable.end());
    while (!poisonable.empty()) {
        const IrValue* value = poisonable.back();
        poisonable.pop_back();
        const auto found = users.find(value);
        if (found == users.end()) continue;
        for (const IrValue* user : found->second) {
            if (reaching.insert(user).second) poisonable.push_back(user);
        }
    }
    std::vector<std::uint32_t> guarded;
    for (std::uint32_t slot = 0; slot < plan.srtReads.size(); ++slot) {
        if (plan.srtReads[slot].value != nullptr && reaching.contains(plan.srtReads[slot].value->Resolve())) guarded.push_back(slot);
    }
    return guarded;
}

}
