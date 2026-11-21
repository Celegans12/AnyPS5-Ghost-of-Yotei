#include "graphics/shader/recompiler/ir/passes/RelativeVgprLowering.h"

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

using Slots = std::unordered_map<uint32_t, Value>;

struct Match {
	Value    index;
	uint32_t offset = 0;
};

uint32_t FlagBase(uint64_t flags) {
	return static_cast<uint32_t>((flags >> 24u) & 0xffu);
}

std::optional<Match> IndexMatch(Value condition) {
	const auto* inst = condition.Resolve().TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::IEqual32) return std::nullopt;
	const auto offset = inst->Arg(1).Resolve();
	if (!offset.IsImmediate()) return std::nullopt;
	return Match {inst->Arg(0).Resolve(), offset.U32()};
}

bool IsMarked(const Inst* inst, bool write) {
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::SelectU32) return false;
	const auto flags = inst->Flags<uint64_t>();
	return (flags & RelativeVgprMark) != 0 && ((flags & RelativeVgprWrite) != 0) == write;
}

std::optional<uint32_t> PhiRegister(const Inst& inst) {
	if (inst.GetOpcode() != ValueOpcode::Phi) return std::nullopt;
	const auto flags = inst.Flags<uint64_t>();
	if ((flags & RelativeVgprMark) == 0) return std::nullopt;
	return static_cast<uint32_t>((flags >> 40u) & 0xffu);
}

struct ReadSite {
	uint32_t           base = 0;
	Value              index;
	std::vector<Value> registers;
};

std::optional<ReadSite> MatchRead(Inst& head) {
	if (std::ranges::any_of(head.Uses(), [&](const Use& use) { return use.operand == 2 && IsMarked(use.user, false); })) {
		return std::nullopt;
	}
	const auto                        flags = head.Flags<uint64_t>();
	const auto                        steps = static_cast<uint32_t>((flags >> 32u) & 0xffffu);
	ReadSite                          site {.base = FlagBase(flags)};
	std::vector<std::optional<Value>> present(steps + 1u);
	Inst*                             select = &head;
	uint32_t                          below  = steps + 1u;
	Value                             bottom;
	while (true) {
		const auto match = IndexMatch(select->Arg(0));
		if (!match || match->offset == 0 || match->offset >= below ||
		    (below != steps + 1u && !(match->index == site.index))) {
			return std::nullopt;
		}
		site.index             = match->index;
		below                  = match->offset;
		present[match->offset] = select->Arg(1).Resolve();
		const auto next        = select->Arg(2).Resolve();
		auto*      next_select = next.TryInstruction();
		if (!IsMarked(next_select, false) || next_select->Uses().size() != 1u ||
		    FlagBase(next_select->Flags<uint64_t>()) != site.base) {
			bottom = next;
			break;
		}
		select = next_select;
	}
	site.registers.resize(steps + 1u, bottom);
	for (uint32_t offset = 1; offset <= steps; offset++) {
		if (present[offset]) site.registers[offset] = *present[offset];
	}
	return site;
}

struct WriteSite {
	uint32_t           base = 0;
	std::vector<Inst*> selects;
	Value              index;
	Value              exec;
	Value              value;
};

std::optional<WriteSite> MatchWrite(Block::iterator first, Block::iterator end) {
	const auto flags = first->Flags<uint64_t>();
	const auto site  = static_cast<uint32_t>(flags & 0xffffffu);
	const auto count = static_cast<uint32_t>((flags >> 48u) & 0x3fffu);
	WriteSite  write {.base = FlagBase(flags), .selects = std::vector<Inst*>(count, nullptr)};
	bool       seen = false;
	for (auto it = first; it != end; ++it) {
		if (!IsMarked(&*it, true) || static_cast<uint32_t>(it->Flags<uint64_t>() & 0xffffffu) != site ||
		    FlagBase(it->Flags<uint64_t>()) != write.base) {
			continue;
		}
		const auto offset = static_cast<uint32_t>((it->Flags<uint64_t>() >> 32u) & 0xffffu);
		if (offset >= count || write.selects[offset] != nullptr) return std::nullopt;
		auto        condition = it->Arg(0).Resolve();
		Value       exec      = Value(true);
		const auto* both      = condition.TryInstruction();
		if (both != nullptr && both->GetOpcode() == ValueOpcode::LogicalAnd) {
			exec      = both->Arg(0).Resolve();
			condition = both->Arg(1).Resolve();
		}
		const auto match = IndexMatch(condition);
		const auto value = it->Arg(1).Resolve();
		if (!match || match->offset != offset ||
		    (seen && (!(match->index == write.index) || !(exec == write.exec) || !(value == write.value)))) {
			return std::nullopt;
		}
		write.index           = match->index;
		write.exec            = exec;
		write.value           = value;
		write.selects[offset] = &*it;
		seen                  = true;
	}
	return seen ? std::optional<WriteSite>(write) : std::nullopt;
}

struct Window {
	uint32_t                     end = 0;
	std::unordered_set<uint32_t> registers;
};

Window FindWindow(const Program& program) {
	Window window;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::SelectU32) continue;
			const auto flags = inst.Flags<uint64_t>();
			if ((flags & RelativeVgprMark) == 0) continue;
			const auto base  = FlagBase(flags);
			const auto count = (flags & RelativeVgprWrite) != 0 ? static_cast<uint32_t>((flags >> 48u) & 0x3fffu)
			                                                    : static_cast<uint32_t>((flags >> 32u) & 0xffffu) + 1u;
			for (uint32_t reg = base; reg < base + count; reg++) window.registers.insert(reg);
			window.end = std::max(window.end, base + count);
		}
	}
	return window;
}

}

uint32_t LowerRelativeVgprs(Program& program) {
	if (program.stage == ShaderType::Pixel) return 0;
	const auto window = FindWindow(program);
	if (window.end == 0) return 0;
	const auto spare = window.end;

	std::vector<std::pair<Block*, Block::iterator>>                         phis;
	std::unordered_map<const Block*, Block::iterator>                      bodies;
	std::unordered_map<const Block*, std::unordered_map<uint32_t, Value>> incoming_by_edge;
	std::unordered_set<uint32_t>                                           split_registers;
	for (auto* block: program.blocks) {
		auto& instructions = block->Instructions();
		auto  body         = std::ranges::find_if(instructions, [](const Inst& inst) {
			return inst.GetOpcode() != ValueOpcode::Phi && inst.GetOpcode() != ValueOpcode::Identity;
		});
		bodies.emplace(block, body);
		for (auto it = instructions.begin(); it != body; ++it) {
			const auto reg = PhiRegister(*it);
			if (!reg || !window.registers.contains(*reg)) continue;
			phis.emplace_back(block, it);
			for (size_t index = 0; index < it->NumArgs(); index++) {
				const auto incoming = it->Arg(index).Resolve();
				const auto [found, inserted] = incoming_by_edge[it->PhiBlock(index)].try_emplace(*reg, incoming);
				if (!inserted && !(found->second == incoming)) split_registers.insert(*reg);
			}
		}
	}

	uint32_t                                                                   lowered = 0;
	std::unordered_map<const Block*, Slots>                                    entry;
	std::unordered_map<const Block*, std::vector<std::pair<uint32_t, Value>>> outgoing;
	for (auto& [block, it]: phis) {
		const auto reg = *PhiRegister(*it);
		if (split_registers.contains(reg)) continue;
		for (size_t index = 0; index < it->NumArgs(); index++) {
			outgoing[it->PhiBlock(index)].emplace_back(reg, it->Arg(index));
		}
		const auto loaded =
		    Value(&*block->PrependNewInst(bodies.at(block), ValueOpcode::LoadRelativeVgpr, {Value(reg)}));
		entry[block][reg] = loaded;
		it->SetFlags(it->Flags<Type>());
		it->ReplaceUsesWith(loaded);
		lowered++;
	}

	std::unordered_map<const Block*, Slots> exit_state;
	for (auto* block: program.blocks) {
		Slots stored = entry[block];
		if (const auto predecessors = block->ImmPredecessors(); predecessors.size() == 1u) {
			if (const auto found = exit_state.find(predecessors.front()); found != exit_state.end()) {
				for (const auto& [slot, value]: found->second) stored.try_emplace(slot, value);
			}
		}
		const auto sync = [&](Block::iterator at, uint32_t slot, Value value) {
			const auto found = stored.find(slot);
			if (found != stored.end() && found->second == value) return;
			block->PrependNewInst(at, ValueOpcode::StoreRelativeVgpr, {Value(slot), value});
			stored[slot] = value;
		};
		const auto emit = [&](Block::iterator at, ValueOpcode opcode, std::initializer_list<Value> args) {
			return Value(&*block->PrependNewInst(at, opcode, args));
		};
		auto& instructions = block->Instructions();
		for (auto it = instructions.begin(); it != instructions.end(); ++it) {
			if (it->GetOpcode() != ValueOpcode::SelectU32) continue;
			const auto flags = it->Flags<uint64_t>();
			if ((flags & RelativeVgprMark) == 0) continue;
			if ((flags & RelativeVgprWrite) == 0) {
				const auto read = MatchRead(*it);
				if (!read) continue;
				const auto count = static_cast<uint32_t>(read->registers.size());
				for (uint32_t offset = 0; offset < count; offset++) {
					sync(it, read->base + offset, read->registers[offset]);
				}
				const auto in_range = emit(it, ValueOpcode::ULessThan32, {read->index, Value(count)});
				const auto shifted  = emit(it, ValueOpcode::IAdd32, {read->index, Value(read->base)});
				const auto slot     = emit(it, ValueOpcode::SelectU32, {in_range, shifted, Value(read->base)});
				const auto loaded   = emit(it, ValueOpcode::LoadRelativeVgpr, {slot});
				it->SetFlags(uint64_t {0});
				it->ReplaceUsesWith(loaded);
				lowered++;
				continue;
			}
			const auto write = MatchWrite(it, instructions.end());
			if (!write) continue;
			const auto count = static_cast<uint32_t>(write->selects.size());
			for (uint32_t offset = 0; offset < count; offset++) {
				if (write->selects[offset] != nullptr) {
					sync(it, write->base + offset, write->selects[offset]->Arg(2).Resolve());
				}
			}
			const auto in_range = emit(it, ValueOpcode::ULessThan32, {write->index, Value(count)});
			const auto active   = emit(it, ValueOpcode::LogicalAnd, {write->exec, in_range});
			const auto shifted  = emit(it, ValueOpcode::IAdd32, {write->index, Value(write->base)});
			const auto slot     = emit(it, ValueOpcode::SelectU32, {active, shifted, Value(spare)});
			block->PrependNewInst(it, ValueOpcode::StoreRelativeVgpr, {slot, write->value});
			for (uint32_t offset = 0; offset < count; offset++) {
				const auto reg = write->base + offset;
				if (write->selects[offset] == nullptr) {
					stored.erase(reg);
					continue;
				}
				const auto loaded = emit(it, ValueOpcode::LoadRelativeVgpr, {Value(reg)});
				stored[reg]       = loaded;
				write->selects[offset]->SetFlags(uint64_t {0});
				write->selects[offset]->ReplaceUsesWith(loaded);
			}
			lowered++;
		}
		if (const auto found = outgoing.find(block); found != outgoing.end()) {
			for (const auto& [reg, incoming]: found->second) {
				sync(block->end(), reg, incoming.Resolve());
			}
		}
		exit_state[block] = std::move(stored);
	}
	program.relative_vgpr_dwords = spare + 1u;
	return lowered;
}

}
