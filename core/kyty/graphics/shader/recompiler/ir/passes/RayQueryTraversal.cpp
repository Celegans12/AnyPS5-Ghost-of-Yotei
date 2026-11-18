#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

std::array<uint32_t, 4> RayTracingLayout::Pack() const {
	std::array<uint32_t, 4> words {};
	const uint32_t          base = TransformBase();
	for (uint32_t i = 0; i < 12; i++) {
		words[i / 8] |= ((transform[i] - base) & 15u) << ((i % 8) * 4);
	}
	words[1] |= ((instance_offset / 8u) & 15u) << 16u | (blas_dword & 15u) << 20u | (blas_packed ? 1u : 0u) << 24u |
	            (root_dword & 15u) << 25u | (flip_facing ? 1u : 0u) << 29u | ((record_stride - 128u) / 32u & 3u) << 30u;
	const bool flags = flags_dword != NoField;
	const bool mask  = mask_dword != NoField;
	words[2] = (mask ? mask_dword & 15u : 0u) | (mask_shift & 31u) << 4u | (flags ? flags_dword & 15u : 0u) << 9u |
	           (flags ? 1u : 0u) << 13u | (flags_shift & 31u) << 14u | (mask ? 1u : 0u) << 19u | (base & 31u) << 20u |
	           (mask ? mask_width & 31u : 0u) << 25u;
	if (inline_type != NoField) {
		words[3] = ((inline_offset / 8u) & 0xffu) | (inline_root & 0xffu) << 8u | (inline_type & 7u) << 16u | 1u << 19u |
		           ((inline_offset >> 2u) & 1u) << 28u;
	}
	words[3] |= (blas_dword >> 4u & 3u) << 20u | (root_dword >> 4u & 3u) << 22u | (mask ? mask_dword >> 4u & 3u : 0u) << 24u |
	            (flags ? flags_dword >> 4u & 3u : 0u) << 26u | (node_count_80 ? 1u : 0u) << 29u;
	return words;
}

uint32_t RayTracingLayout::TransformBase() const {
	const auto [low, high] = std::ranges::minmax(transform);
	return high < 16 ? 0u : low;
}

uint32_t RayTracingAlphaFields::Pack(const RayTracingLayout& layout) const {
	return (material_dword & 63u) | (table_dword & 63u) << 6u | (first_dword & 63u) << 12u |
	       (vertex_align ? 1u : 0u) << 18u | (vertex_mask & 0xffu) << 19u | (layout.blas_dword & 15u) << 27u |
	       (layout.blas_packed ? 1u : 0u) << 31u;
}

namespace {

constexpr std::array<uint32_t, 4> TraversalDescriptor {0u, 0x80000000u, 0xffffffffu, 0x810003ffu};
constexpr uint32_t                One = 0x3f800000u;
constexpr uint32_t OddRemaps = 0x2942u;
constexpr uint32_t Barycentrics = 1u << 24;

constexpr uint32_t InstanceBlas = 0, InstanceRoot = 56, InstanceFlags = 60;

Inst* Producer(Value value, ValueOpcode opcode) {
	auto* inst = value.Resolve().TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
}

std::optional<uint32_t> Immediate(Value value) {
	value = value.Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::U32) return std::nullopt;
	return value.U32();
}

std::optional<uint32_t> Slot(Value value) {
	const auto* read = Producer(value, ValueOpcode::ReadConst);
	return read != nullptr ? Immediate(read->Arg(1)) : std::nullopt;
}

size_t Incoming(const Inst& phi, const Block* from) {
	for (size_t index = 0; index < phi.NumArgs(); ++index) {
		if (phi.PhiBlock(index) == from) return index;
	}
	return phi.NumArgs();
}

Value IncomingValue(const Inst& phi, const Block* from) {
	const auto index = Incoming(phi, from);
	return index < phi.NumArgs() ? phi.Arg(index) : Value {};
}

void Collect(Value value, const Block* header, std::unordered_set<const Inst*>& reached) {
	const auto* inst = value.Resolve().TryInstruction();
	if (inst == nullptr || !reached.insert(inst).second) return;
	if (inst->GetOpcode() == ValueOpcode::SelectU32) {
		Collect(inst->Arg(1), header, reached);
		Collect(inst->Arg(2), header, reached);
	} else if (inst->GetOpcode() == ValueOpcode::Phi && inst->Parent() != header) {
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			Collect(inst->Arg(index), header, reached);
		}
	}
}

bool DirectlyUsed(const Inst& inst) {
	return std::ranges::any_of(inst.Uses(), [](const Use& use) {
		const auto op = use.user->GetOpcode();
		return op != ValueOpcode::Phi && (op != ValueOpcode::SelectU32 || use.operand != 2);
	});
}

bool GlobalLoad(const Program& program, const Inst& inst, uint32_t offset) {
	if (inst.GetOpcode() != ValueOpcode::LoadAddressU32) return false;
	const auto& memory = program.memory_info[inst.Flags<MemoryFlags>().index];
	return memory.kind == ResourceKind::Global && memory.address_is_full && memory.offset == offset;
}

void Redirect(Inst& inst, Value replacement, const Inst& keep) {
	const auto uses = inst.Uses();
	for (const auto& use: uses) {
		if (use.user != &keep) use.user->SetArg(use.operand, replacement);
	}
}

struct Inserter {
	Block*          block;
	Block::iterator at;

	Value operator()(ValueOpcode opcode, std::initializer_list<Value> args,
	                 uint64_t flags = 0) const {
		auto& inst = *block->PrependNewInst(at, opcode, args, flags);
		return TypeOf(opcode) == Type::Void ? Value {} : Value(&inst);
	}

	Value operator()(const Inst& inst, std::span<const Value> args) const {
		auto& copy = *block->Instructions().emplace(at, inst.GetOpcode(), inst.Flags<uint64_t>());
		copy.SetParent(block);
		for (size_t arg = 0; arg < args.size(); ++arg) {
			copy.SetArg(arg, args[arg]);
		}
		return Value(&copy);
	}
};

Block::iterator Position(Inst& inst) {
	return std::ranges::find_if(*inst.Parent(), [&](const Inst& other) { return &other == &inst; });
}

size_t BlockIndex(const Program& program, const Block* block) {
	return static_cast<size_t>(std::ranges::find(program.blocks, block) - program.blocks.begin());
}

Block* EnclosingLoop(const Program& program, const Block* block) {
	const auto at = BlockIndex(program, block);
	for (auto header = at; header-- > 0;) {
		const auto& term = program.blocks[header]->terminator;
		if (term.loop_header && term.continue_block != nullptr &&
		    BlockIndex(program, term.continue_block) >= at) {
			return program.blocks[header];
		}
	}
	return nullptr;
}

struct Loop {
	Block*                  preheader = nullptr;
	Block*                  header    = nullptr;
	Block*                  latch     = nullptr;
	std::span<Block* const> loop;
};

struct Traversal: Loop {
	Inst*                bvh    = nullptr;
	Inst*                exec   = nullptr;
	Inst*                extent = nullptr;
	Inst*                vote   = nullptr;
	std::vector<Inst*>   lanes;
	std::array<Value, 7> ray {};
	std::array<Inst*, 5> state {};
	std::array<const Inst*, 3> loads {};
	std::array<Inst*, 4>    instance {};
	Inst*                   record = nullptr;
	std::vector<Inst*>      scalar;
	std::array<uint32_t, 2> tlas {};
	uint32_t                cull_mask = 0;
	const Inst*          walk     = nullptr;
	const Inst*          resource = nullptr;
	std::array<Value, 2> words {};
	const Inst* root       = nullptr;
	const Inst* blas       = nullptr;
	bool        packed     = false;
	uint32_t    mask_shift = 8;
};

Value Entry(const Loop& t, const Inst& phi) {
	return IncomingValue(phi, t.preheader);
}

Value Back(const Loop& t, const Inst& phi) {
	return IncomingValue(phi, t.latch);
}

Inst* HeaderPhi(const Loop& t, Value value) {
	auto* inst = Producer(value, ValueOpcode::Phi);
	return inst != nullptr && inst->Parent() == t.header ? inst : nullptr;
}

bool Inside(const Loop& t, const Block* block) {
	return std::ranges::find(t.loop, block) != t.loop.end();
}

bool Enter(const Program& program, Block* header, Loop& l) {
	if (header == nullptr || header->ImmPredecessors().size() != 2) return false;
	const auto entries = header->ImmPredecessors();
	l.header           = header;
	l.latch            = header->terminator.continue_block;
	l.preheader        = entries[0] == l.latch ? entries[1] : entries[0];
	l.loop             = std::span(program.blocks.begin() + BlockIndex(program, header),
	                               program.blocks.begin() + BlockIndex(program, l.latch) + 1);
	return std::ranges::find(entries, l.latch) != entries.end();
}

bool MatchLoop(const Program& program, Inst& bvh, Value extent, Traversal& t) {
	if (!std::ranges::all_of(bvh.Uses(), [](const Use& use) {
		    return use.user->GetOpcode() == ValueOpcode::CompositeExtractU32x4 &&
			       Immediate(use.user->Arg(1)) < 4u;
	    })) {
		return false;
	}
	if (!Enter(program, EnclosingLoop(program, bvh.Parent()), t) ||
	    t.header->terminator.kind != CFG::TerminatorKind::Branch ||
	    t.header->terminator.true_block == nullptr) {
		return false;
	}
	t.bvh    = &bvh;
	t.exec   = HeaderPhi(t, bvh.Arg(2));
	t.extent = HeaderPhi(t, extent);
	if (t.exec == nullptr) return false;

	const auto lanes = Entry(t, *t.exec);
	const auto alive = Back(t, *t.exec);
	for (auto& inst: *t.header) {
		if (inst.GetOpcode() != ValueOpcode::Phi || inst.GetType() != Type::U1 ||
		    Back(t, inst) != alive) {
			continue;
		}
		if (Entry(t, inst) != lanes) return false;
		t.lanes.push_back(&inst);
	}
	const auto& term = t.latch->terminator;
	t.vote           = Producer(t.latch->condition, ValueOpcode::ConditionRef);
	if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.true_block != t.header ||
	    t.vote == nullptr || t.vote->Arg(0) != alive ||
	    t.vote->Flags<CFG::BranchCondition>() != CFG::BranchCondition::ExecNonZero) {
		return false;
	}
	auto* voting = t.vote->Parent();
	if (voting != t.latch &&
	    (t.latch->ImmPredecessors().size() != 1 || t.latch->ImmPredecessors()[0] != voting)) {
		return false;
	}
	const auto* moved = t.extent != nullptr ? Back(t, *t.extent).Resolve().TryInstruction() : nullptr;
	for (auto it = Position(*t.vote); moved != nullptr && it != voting->end(); ++it) {
		if (&*it == moved) return false;
	}
	if (voting != t.latch && moved != nullptr && moved->Parent() == t.latch) return false;

	const auto exec   = Value(t.exec);
	const auto inside = [&](const Block* block) { return Inside(t, block); };
	for (const auto* block: t.loop) {
		if (block->terminator.kind == CFG::TerminatorKind::Return ||
		    (block != t.latch && !std::ranges::all_of(block->ImmSuccessors(), inside))) {
			return false;
		}
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == ValueOpcode::BvhIntersect && &inst != &bvh) return false;
			if (!HasSideEffects(op) || op == ValueOpcode::Reference) continue;
			uint32_t data = 3;
			if (op == ValueOpcode::WriteSharedU32) {
				data = 1;
			} else if (op != ValueOpcode::StoreAddressU32) {
				return false;
			}
			const auto* pushed = Producer(ResolveActiveU32(inst.Arg(data), exec),
			                              ValueOpcode::CompositeExtractU32x4);
			if (pushed == nullptr || pushed->Arg(0) != Value(&bvh)) return false;
		}
	}
	return true;
}

bool Match(const Program& program, Inst& bvh, Traversal& t, bool tracked = true) {
	const auto* descriptor = Producer(bvh.Arg(0), ValueOpcode::CompositeConstructU32x4);
	const auto* address    = Producer(bvh.Arg(1), ValueOpcode::MakeImageAddress);
	if (descriptor == nullptr || address == nullptr || bvh.Flags<uint32_t>() != 2u ||
	    !MatchLoop(program, bvh, address->Arg(2), t)) {
		return false;
	}
	for (uint32_t word = 0; word < 4; ++word) {
		if (Immediate(descriptor->Arg(word)) != TraversalDescriptor[word]) return false;
	}

	const auto before = [&](Value value) {
		const auto* inst = value.Resolve().TryInstruction();
		return inst == nullptr || BlockIndex(program, inst->Parent()) < BlockIndex(program, t.header);
	};
	for (uint32_t arg = 2; arg < 9; ++arg) {
		const auto* ray = HeaderPhi(t, address->Arg(arg));
		if (ray == nullptr && !tracked && arg == 2 && t.extent == nullptr && before(address->Arg(arg))) {
			t.ray[6] = address->Arg(arg);
			continue;
		}
		if (ray == nullptr) return false;
		t.ray[(arg + 4) % 7] = Entry(t, *ray);
	}

	const auto  exec  = Value(t.exec);
	const auto  lanes = Entry(t, *t.exec);
	const auto* sum   = Producer(ResolveActiveU32(address->Arg(0), exec), ValueOpcode::IAdd32);
	if (sum == nullptr) return false;
	auto* node      = HeaderPhi(t, sum->Arg(0));
	auto  base_part = sum->Arg(1);
	if (node == nullptr) {
		node      = HeaderPhi(t, sum->Arg(1));
		base_part = sum->Arg(0);
	}
	const auto* low = Producer(ResolveActiveU32(base_part, exec), ValueOpcode::CompositeExtractU64);
	const auto* shift =
	    low != nullptr ? Producer(low->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
	const auto* base =
	    shift != nullptr ? Producer(shift->Arg(0), ValueOpcode::CompositeConstructU64) : nullptr;
	std::array<Inst*, 2> shifted {};
	if (base == nullptr && !tracked && node != nullptr) {
		const auto* high = Producer(ResolveActiveU32(address->Arg(1), exec), ValueOpcode::IAdd32);
		shifted[0]       = HeaderPhi(t, ResolveActiveU32(base_part, exec));
		for (uint32_t arg = 0; high != nullptr && arg < 2 && shifted[1] == nullptr; ++arg) {
			shifted[1] = HeaderPhi(t, high->Arg(arg));
		}
		shift = nullptr;
		for (uint32_t word = 0; word < 2 && shifted[0] != nullptr && shifted[1] != nullptr; ++word) {
			const auto* part =
			    Producer(ResolveActiveU32(Entry(t, *shifted[word]), lanes), ValueOpcode::CompositeExtractU64);
			const auto* by = part != nullptr ? Producer(part->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
			if (by == nullptr || Immediate(part->Arg(1)) != word || (word == 1 && by != shift)) {
				shift = nullptr;
				break;
			}
			shift = by;
		}
		base = shift != nullptr ? Producer(shift->Arg(0), ValueOpcode::CompositeConstructU64) : nullptr;
	}
	const bool in_phis = shifted[0] != nullptr && base != nullptr;
	if (node == nullptr || base == nullptr || (!in_phis && Immediate(low->Arg(1)) != 0u) ||
	    Immediate(shift->Arg(1)) != 3u) {
		return false;
	}
	for (uint32_t word = 0; word < 2; ++word) {
		if (in_phis) {
			t.words[word] = base->Arg(word);
			if (!before(t.words[word])) return false;
			continue;
		}
		const auto* header = HeaderPhi(t, base->Arg(word));
		if (!tracked) {
			if (header == nullptr) return false;
			t.words[word] = ResolveActiveU32(Entry(t, *header), lanes);
			if (!before(t.words[word])) return false;
			continue;
		}
		const auto  slot =
		    header != nullptr ? Slot(ResolveActiveU32(Entry(t, *header), lanes)) : std::nullopt;
		if (!slot) return false;
		t.tlas[word] = *slot;
	}

	std::unordered_set<const Inst*> nodes;
	Collect(Back(t, *node), t.header, nodes);
	const Inst* root  = nullptr;
	const Inst* word0 = nullptr;
	for (const auto* inst: nodes) {
		for (auto [load, offset]: {std::pair {&root, InstanceRoot}, std::pair {&word0, 0u}}) {
			if (!GlobalLoad(program, *inst, offset)) continue;
			if (*load != nullptr) return false;
			*load = inst;
		}
	}
	if (root == nullptr || word0 == nullptr) return false;

	uint32_t blas_offset = InstanceBlas;
	if (in_phis) {
		std::unordered_set<const Inst*> taken;
		Collect(Back(t, *shifted[0]), t.header, taken);
		blas_offset = UINT32_MAX;
		for (const auto* inst: taken) {
			if (inst->GetOpcode() != ValueOpcode::LoadAddressU32 || inst->Arg(1) != root->Arg(1) ||
			    inst->Arg(2) != root->Arg(2)) {
				continue;
			}
			if (blas_offset != UINT32_MAX) return false;
			blas_offset = program.memory_info[inst->Flags<MemoryFlags>().index].offset;
		}
		if (blas_offset != 48u) return false;
	}

	const std::array<std::pair<const Inst*, uint32_t>, 3> wanted {
	    {{word0, 4}, {root, InstanceFlags}, {root, blas_offset}}};
	std::array<const Inst*, 3> found {};
	for (const auto* block: t.loop) {
		for (const auto& inst: *block) {
			for (size_t load = 0; load < wanted.size(); ++load) {
				const auto& [like, offset] = wanted[load];
				if (!GlobalLoad(program, inst, offset) || inst.Arg(1) != like->Arg(1) ||
				    inst.Arg(2) != like->Arg(2)) {
					continue;
				}
				if (found[load] != nullptr) return false;
				found[load] = &inst;
			}
		}
	}
	const auto [word1, flags, blas] = found;
	if (word1 == nullptr || flags == nullptr || blas == nullptr) return false;
	t.loads = {word0, word1, flags};
	t.root  = root;
	t.blas  = blas;
	if (in_phis) {
		std::unordered_set<const Inst*> taken;
		Collect(Back(t, *shifted[1]), t.header, taken);
		t.packed = std::ranges::any_of(taken, [&](const Inst* inst) {
			if (inst->GetOpcode() != ValueOpcode::BitwiseAnd32) return false;
			for (uint32_t arg = 0; arg < 2; ++arg) {
				const auto* word = ResolveActiveU32(inst->Arg(arg), root->Arg(3)).TryInstruction();
				if (Immediate(inst->Arg(1 - arg)) == 0x3fffffu && word != nullptr &&
				    GlobalLoad(program, *word, blas_offset + 4u) && word->Arg(1) == root->Arg(1) &&
				    word->Arg(2) == root->Arg(2)) {
					return true;
				}
			}
			return false;
		});
		if (!t.packed) return false;
	}

	std::vector<std::pair<Inst*, std::unordered_set<const Inst*>>> carried;
	for (auto& inst: *t.header) {
		if (inst.GetOpcode() == ValueOpcode::Phi && inst.GetType() == Type::U32 &&
		    DirectlyUsed(inst)) {
			carried.push_back({&inst, {}});
			Collect(Back(t, inst), t.header, carried.back().second);
		}
	}
	const std::array<const Inst*, 5> targets {word0, word1, root->Arg(1).Resolve().TryInstruction(),
	                                          root->Arg(2).Resolve().TryInstruction(), flags};
	t.state[0] = node;
	for (uint32_t role = 1; !tracked && role < targets.size(); ++role) {
		Inst*    only  = nullptr;
		uint32_t count = 0;
		for (auto& [carrier, values]: carried) {
			if (values.contains(targets[role])) {
				only = carrier;
				++count;
			}
		}
		if (count == 1 && std::ranges::find(t.state.begin(), t.state.begin() + role, only) == t.state.begin() + role) {
			t.state[role] = only;
		} else if (role == 1) {
			return false;
		}
	}
	for (uint32_t role = 1; tracked && role < targets.size(); ++role) {
		for (auto& [carrier, values]: carried) {
			if (!values.contains(targets[role])) continue;
			if (t.state[role] != nullptr) return false;
			t.state[role] = carrier;
		}
		if (t.state[role] == nullptr ||
		    std::ranges::find(t.state.begin(), t.state.begin() + role, t.state[role]) !=
		        t.state.begin() + role) {
			return false;
		}
	}

	const auto leaf = root->Arg(3);
	const auto test = [&](Value condition) -> std::optional<uint32_t> {
		const auto* both = Producer(condition, ValueOpcode::LogicalAnd);
		if (both == nullptr || (both->Arg(0) != leaf && both->Arg(1) != leaf)) return std::nullopt;
		const auto* sign = Producer(both->Arg(0) == leaf ? both->Arg(1) : both->Arg(0),
		                            ValueOpcode::SGreaterThan32);
		const auto* mask = sign != nullptr ? Producer(ResolveActiveU32(sign->Arg(1), leaf),
		                                              ValueOpcode::ShiftLeftLogical32)
		                                   : nullptr;
		const auto  bit  = mask != nullptr ? Immediate(mask->Arg(1)) : std::nullopt;
		const bool  range = bit && (t.packed ? *bit <= 7u : *bit >= 16u && *bit <= 23u);
		const bool  kept  = range && Immediate(sign->Arg(0)) == 0u &&
		                   ResolveActiveU32(mask->Arg(0), leaf).TryInstruction() == flags;
		return kept ? bit : std::nullopt;
	};
	auto                            bit = test(blas->Arg(3));
	std::unordered_set<const Inst*> next;
	if (!bit) Collect(Back(t, in_phis ? *shifted[0] : *HeaderPhi(t, base->Arg(0))), t.header, next);
	for (std::vector<const Inst*> taking {blas}; !bit && !taking.empty();) {
		const auto* value = taking.back();
		taking.pop_back();
		for (const auto& use: value->Uses()) {
			if (!bit && use.user->GetOpcode() == ValueOpcode::SelectU32 && use.operand == 1 &&
			    next.contains(use.user)) {
				bit = test(use.user->Arg(0));
				taking.push_back(use.user);
			}
		}
	}
	if (!bit) return false;
	t.cull_mask = 1u << (23u - *bit);
	if (t.packed) {
		t.cull_mask  = 1u << (7u - *bit);
		t.mask_shift = 24;
	}
	return true;
}

Inst* DescriptorAddress(const Inst& descriptor) {
	const auto* low   = Producer(descriptor.Arg(0), ValueOpcode::CompositeExtractU64);
	const auto* field = low != nullptr ? Producer(low->Arg(0), ValueOpcode::BitwiseAnd64) : nullptr;
	const auto* shift =
	    field != nullptr ? Producer(field->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
	const auto* flags = Producer(descriptor.Arg(3), ValueOpcode::BitwiseOr32);
	const auto  bits  = flags != nullptr ? Immediate(flags->Arg(1)) : std::nullopt;
	if (shift == nullptr || Immediate(low->Arg(1)) != 0u || Immediate(shift->Arg(1)) != 8u ||
	    !bits || (*bits & Barycentrics) == 0u) {
		return nullptr;
	}
	return Producer(shift->Arg(0), ValueOpcode::CompositeConstructU64);
}

struct BlasWords {
	Inst*       word0    = nullptr;
	Inst*       word1    = nullptr;
	const Inst* instance = nullptr;
	const Inst* wide     = nullptr;

	bool Base(const Inst* inst, uint32_t k) const {
		if (wide == nullptr) return inst == (k == 0 ? word0 : word1);
		return inst != nullptr && inst->GetOpcode() == ValueOpcode::CompositeExtractU64 &&
		       inst->Arg(0).Resolve().TryInstruction() == wide && inst->Arg(1).Resolve() == Value(k);
	}
};

const Inst* AddressWord(const Program& program, const Inst* load, uint32_t offset) {
	if (load == nullptr || load->GetOpcode() != ValueOpcode::LoadAddressU32) return nullptr;
	const auto& memory = program.memory_info[load->Flags<MemoryFlags>().index];
	const bool  based  = memory.kind == ResourceKind::ScalarAddress ||
	                     (memory.kind == ResourceKind::Global && !memory.address_is_full);
	if (!based || memory.offset != offset || Immediate(load->Arg(1)) != 0u) return nullptr;
	return Producer(load->Arg(0), ValueOpcode::GetAddressResource);
}

std::optional<uint32_t> MaskTest(const Program& program, const Inst& branch, const Inst& instance) {
	const auto* test = Producer(branch.Arg(0), ValueOpcode::INotEqual32);
	const auto* bits = test != nullptr && Immediate(test->Arg(1)) == 0u
	                       ? Producer(test->Arg(0), ValueOpcode::BitwiseAnd32)
	                       : nullptr;
	const auto  mask = bits != nullptr ? Immediate(bits->Arg(1)) : std::nullopt;
	if (!mask || *mask == 0u || (*mask & ~0xff00u) != 0u ||
	    AddressWord(program, bits->Arg(0).Resolve().TryInstruction(), InstanceFlags) != &instance) {
		return std::nullopt;
	}
	return *mask >> 8;
}

bool TlasWord(Value value, Value word, uint32_t slot, bool tracked) {
	return tracked ? Slot(value) == slot : value.Resolve() == word.Resolve();
}

bool InstanceArray(const Inst& low, uint32_t slot, Value word = {}, bool tracked = true) {
	for (uint32_t arg = 0; arg < 2; ++arg) {
		const auto* part = Producer(low.Arg(arg), ValueOpcode::CompositeExtractU64);
		const auto* scale =
		    part != nullptr ? Producer(part->Arg(0), ValueOpcode::ShiftLeftLogical64) : nullptr;
		const auto* index = scale != nullptr
		                        ? Producer(scale->Arg(0), ValueOpcode::CompositeConstructU64)
		                        : nullptr;
		const auto* node =
		    index != nullptr ? Producer(index->Arg(0), ValueOpcode::ShiftRightLogical32) : nullptr;
		const auto* array = Producer(low.Arg(1 - arg), ValueOpcode::IAdd32);
		if (node != nullptr && array != nullptr && Immediate(part->Arg(1)) == 0u &&
		    Immediate(scale->Arg(1)) == 7u && Immediate(index->Arg(1)) == 0u &&
		    Immediate(node->Arg(1)) == 3u &&
		    (TlasWord(array->Arg(0), word, slot, tracked) ||
		     TlasWord(array->Arg(1), word, slot, tracked))) {
			return true;
		}
	}
	return false;
}

std::array<const Inst*, 2> RecordBase(Value resource) {
	const auto* buffer = Producer(resource, ValueOpcode::GetBufferResource);
	const auto* stride =
	    buffer != nullptr ? Producer(buffer->Arg(1), ValueOpcode::BitwiseOr32) : nullptr;
	const auto bits = stride != nullptr ? Immediate(stride->Arg(1)) : std::nullopt;
	if (!bits || ((*bits >> 16) & 0x3fffu) != 8u) return {};
	return {buffer->Arg(0).Resolve().TryInstruction(), stride->Arg(0).Resolve().TryInstruction()};
}

bool Kept(const Inst& phi, const Inst& exec) {
	return std::ranges::all_of(phi.Uses(), [&](const Use& use) {
		return use.user->GetOpcode() == ValueOpcode::SelectU32 && use.operand == 2 &&
		       use.user->Arg(0).Resolve().TryInstruction() == &exec;
	});
}

bool Replaced(const Traversal& t, const Inst& user) {
	const auto other = [&](Value value) { return value.Resolve().TryInstruction() != &user; };
	return user.Parent() != t.header && Inside(t, user.Parent()) && other(t.bvh->Arg(0)) &&
	       other(t.bvh->Arg(1)) && other(t.record->Arg(0));
}

BlasWords WideBlasWords(const Program& program, const Inst* wide) {
	if (wide == nullptr || wide->GetOpcode() != ValueOpcode::ShiftLeftLogical64 ||
	    Immediate(wide->Arg(1)) != 3u) {
		return {};
	}
	const auto* pair = Producer(wide->Arg(0), ValueOpcode::CompositeConstructU64);
	const auto* bits = pair != nullptr ? Producer(pair->Arg(1), ValueOpcode::BitwiseAnd32) : nullptr;
	BlasWords   words {.wide = wide};
	words.word0 = pair != nullptr ? Producer(pair->Arg(0), ValueOpcode::LoadAddressU32) : nullptr;
	for (uint32_t arg = 0; bits != nullptr && arg < 2; ++arg) {
		if (Immediate(bits->Arg(1 - arg)) == 0x3fffffu) {
			words.word1 = Producer(bits->Arg(arg), ValueOpcode::LoadAddressU32);
		}
	}
	words.instance = AddressWord(program, words.word0, 48);
	if (words.word1 == nullptr || words.instance == nullptr ||
	    AddressWord(program, words.word1, 52) != words.instance) {
		return {};
	}
	return words;
}

BlasWords InstanceBlasWords(const Program& program, const Inst& descriptor, bool tracked) {
	const auto* low   = Producer(descriptor.Arg(0), ValueOpcode::CompositeExtractU64);
	const auto* field = low != nullptr ? Producer(low->Arg(0), ValueOpcode::BitwiseAnd64) : nullptr;
	const auto* shift =
	    field != nullptr ? Producer(field->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
	if (!tracked && DescriptorAddress(descriptor) == nullptr && shift != nullptr) {
		return WideBlasWords(program, shift->Arg(0).Resolve().TryInstruction());
	}
	const auto* blas = DescriptorAddress(descriptor);
	BlasWords   words;
	words.word0    = blas != nullptr ? Producer(blas->Arg(0), ValueOpcode::LoadAddressU32) : nullptr;
	words.word1    = blas != nullptr ? Producer(blas->Arg(1), ValueOpcode::LoadAddressU32) : nullptr;
	words.instance = AddressWord(program, words.word0, InstanceBlas);
	if (words.instance == nullptr ||
	    AddressWord(program, words.word1, InstanceBlas + 4) != words.instance) {
		return {};
	}
	return words;
}

bool MatchNested(const Program& program, Inst& bvh, std::span<Inst* const> intersections,
                 Traversal& t, bool tracked = true) {
	const auto* descriptor = Producer(bvh.Arg(0), ValueOpcode::CompositeConstructU32x4);
	const auto* address    = Producer(bvh.Arg(1), ValueOpcode::MakeImageAddress);
	if (descriptor == nullptr || address == nullptr ||
	    !MatchLoop(program, bvh, address->Arg(1), t)) {
		return false;
	}
	auto*      node  = HeaderPhi(t, address->Arg(0));
	const auto words = InstanceBlasWords(program, *descriptor, tracked);
	auto* const word0    = words.word0;
	auto* const word1    = words.word1;
	const auto* instance = words.instance;
	if (node == nullptr || instance == nullptr || Inside(t, word0->Parent()) ||
	    Inside(t, word1->Parent())) {
		return false;
	}
	auto* low  = Producer(instance->Arg(0), ValueOpcode::IAdd32);
	auto* high = instance->Arg(1).Resolve().TryInstruction();
	if (low == nullptr || high == nullptr) return false;
	t.instance = {low, high, word0, word1};

	const Inst* walk = nullptr;
	for (const auto* other: intersections) {
		const auto* around = EnclosingLoop(program, other->Parent());
		if (other->Flags<uint32_t>() != 1u || around == nullptr ||
		    BlockIndex(program, around) >= BlockIndex(program, t.header) ||
		    BlockIndex(program, around->terminator.continue_block) < BlockIndex(program, t.latch)) {
			continue;
		}
		const auto* words     = Producer(other->Arg(0), ValueOpcode::CompositeConstructU32x4);
		const auto* tlas      = words != nullptr ? DescriptorAddress(*words) : nullptr;
		const auto  low_slot  = tlas != nullptr ? Slot(tlas->Arg(0)) : std::nullopt;
		const auto  high_slot = tlas != nullptr ? Slot(tlas->Arg(1)) : std::nullopt;
		if (tlas == nullptr || (tracked && (!low_slot || !high_slot))) continue;
		if (walk != nullptr) return false;
		walk    = other;
		t.tlas  = {low_slot.value_or(0u), high_slot.value_or(0u)};
		t.words = {tlas->Arg(0), tlas->Arg(1)};
	}
	t.walk = walk;
	if (walk == nullptr) return false;

	const auto* ray   = Producer(walk->Arg(1), ValueOpcode::MakeImageAddress);
	auto*       outer = EnclosingLoop(program, walk->Parent());
	while (auto* up = EnclosingLoop(program, outer)) {
		outer = up;
	}
	if (ray == nullptr) return false;
	const auto before = [&](Value word) {
		const auto* value = word.Resolve().TryInstruction();
		return value == nullptr || BlockIndex(program, value->Parent()) < BlockIndex(program, outer);
	};
	for (uint32_t arg = 2; arg < 8; ++arg) {
		t.ray[arg - 2] = ray->Arg(arg);
	}
	t.ray[6] = t.extent != nullptr ? Entry(t, *t.extent) : address->Arg(1);

	if ((!tracked && (!before(t.words[0]) || !before(t.words[1]))) ||
	    !InstanceArray(*low, t.tlas[0], t.words[0], tracked)) {
		return false;
	}

	std::unordered_set<const Inst*> next;
	Collect(Back(t, *node), t.header, next);
	Inst* record = nullptr;
	for (const auto* inst: next) {
		auto* load =
		    inst->GetOpcode() == ValueOpcode::CompositeExtractU32x2 && Immediate(inst->Arg(1)) == 0u
		        ? Producer(inst->Arg(0), ValueOpcode::LoadBufferU32x2)
		        : nullptr;
		if (load == nullptr) continue;
		if (record != nullptr && record != load) return false;
		record = load;
	}
	t.record        = record;
	const auto base = record != nullptr ? RecordBase(record->Arg(0)) : std::array<const Inst*, 2> {};
	if (record == nullptr || !words.Base(base[0], 0) || !words.Base(base[1], 1) ||
	    Immediate(record->Arg(2)) != 0u || Immediate(record->Arg(3)) != 0u ||
	    !Inside(t, record->Parent())) {
		return false;
	}
	const auto& memory = program.memory_info[record->Flags<MemoryFlags>().index];
	if (!memory.idxen || memory.offen || memory.offset != 0u) return false;
	const auto record_uses = record->Uses();
	const auto second      = std::ranges::find_if(record_uses, [](const Use& use) {
		return use.user->GetOpcode() == ValueOpcode::CompositeExtractU32x2 &&
		       Immediate(use.user->Arg(1)) == 1u;
	});
	if (second == record_uses.end()) return false;
	for (auto& inst: *t.header) {
		std::unordered_set<const Inst*> values;
		if (inst.GetOpcode() == ValueOpcode::Phi && inst.GetType() == Type::U32 && &inst != node &&
		    DirectlyUsed(inst)) {
			Collect(Back(t, inst), t.header, values);
		}
		if (!values.contains(second->user)) continue;
		if (t.state[1] != nullptr) return false;
		t.state[1] = &inst;
	}
	if (t.state[1] == nullptr) return false;
	t.state[0] = node;

	t.resource = instance;
	return true;
}

bool NestedReads(const Program& program, Traversal& t) {
	if (t.extent == nullptr) return false;
	auto* outer = EnclosingLoop(program, t.walk->Parent());
	while (auto* up = EnclosingLoop(program, outer)) {
		outer = up;
	}
	for (uint32_t k = 0; k < 6; ++k) {
		const auto* value = t.ray[k].Resolve().TryInstruction();
		if (value != nullptr && BlockIndex(program, value->Parent()) >= BlockIndex(program, outer)) {
			return false;
		}
	}
	const auto* node     = t.state[0];
	const auto* instance = t.resource;
	auto* const low      = t.instance[0];
	auto* const high     = t.instance[1];
	const auto                            first = BlockIndex(program, instance->Parent());
	const auto                            begin = BlockIndex(program, t.header);
	const auto                            last  = BlockIndex(program, t.latch);
	std::optional<uint32_t>               mask;
	std::unordered_set<const Inst*>       seen;
	std::vector<std::pair<Inst*, size_t>> reads;
	for (std::vector<const Inst*> work {low, high}; !work.empty();) {
		const auto* value = work.back();
		work.pop_back();
		if (!seen.insert(value).second) continue;
		const bool swapped = std::ranges::find(t.instance, value) != t.instance.end();
		for (const auto& use: value->Uses()) {
			auto*      user = use.user;
			const auto at   = BlockIndex(program, user->Parent());
			if (at < first || at > last) continue;
			if (at < begin && user->GetOpcode() != ValueOpcode::ConditionRef) {
				work.push_back(user);
			} else if (at < begin) {
				if (mask) return false;
				mask = MaskTest(program, *user, *instance);
				if (!mask) return false;
			} else if (user->Parent() == t.header) {
				if (user != node && !Kept(*user, *t.exec)) return false;
			} else if (Replaced(t, *user)) {
				if (!swapped) return false;
				reads.emplace_back(user, use.operand);
			}
		}
	}
	t.cull_mask = mask.value_or(0xffu);

	while (!reads.empty()) {
		const auto [inst, operand] = reads.back();
		reads.pop_back();
		const auto op = inst->GetOpcode();
		if (!Inside(t, inst->Parent())) continue;
		if (op == ValueOpcode::ReadConstBuffer || op == ValueOpcode::GetBufferResource ||
		    op == ValueOpcode::GetImageResource || op == ValueOpcode::GetSamplerResource ||
		    ((op == ValueOpcode::ReadLane || op == ValueOpcode::WriteLane) && operand != 0)) {
			return false;
		}
		if (op == ValueOpcode::Ballot || op == ValueOpcode::ReadFirstLane ||
		    op == ValueOpcode::ReadLane || op == ValueOpcode::WriteLane ||
		    op == ValueOpcode::SubgroupUMinU32 || op == ValueOpcode::SubgroupUMaxU32 ||
		    op == ValueOpcode::SubgroupOrU32 || !seen.insert(inst).second) {
			continue;
		}
		if (op == ValueOpcode::LoadAddressU32 &&
		    program.memory_info[inst->Flags<MemoryFlags>().index].kind ==
		        ResourceKind::ScalarAddress) {
			t.scalar.push_back(inst);
		}
		for (const auto& use: inst->Uses()) {
			reads.emplace_back(use.user, use.operand);
		}
	}
	return true;
}

struct Walk: Loop {
	Inst*                skip  = nullptr;
	Inst*                leave = nullptr;
	Value                lanes;
	std::array<Value, 7> ray {};
	Inst*                bvh   = nullptr;
	Inst*                flags = nullptr;
	Inst*                word1 = nullptr;
	std::array<Inst*, 2> instance {};
	std::array<Inst*, 3> lanes_in {};
	Inst*                accepted = nullptr;
	Inst*                commit   = nullptr;
	std::vector<std::pair<Inst*, Value>> commits;
	std::array<uint32_t, 2>              tlas {};
	const Inst*                          walk = nullptr;
	std::array<Value, 2>                 words {};
	Loop                                 blas;
};

enum Leaf { Lanes, Numerator, Denominator, Flags, Word1, InstanceLow, InstanceHigh, None };

Leaf LeafOf(const Walk& w, const Inst& inst) {
	if (std::ranges::find(w.lanes_in, &inst) != w.lanes_in.end()) return Lanes;
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU32x4 && inst.Arg(0) == Value(w.bvh)) {
		const auto index = Immediate(inst.Arg(1));
		if (index == 0u) return Numerator;
		return index == 1u ? Denominator : None;
	}
	if (&inst == w.flags) return Flags;
	if (&inst == w.word1) return Word1;
	if (&inst == w.instance[0]) return InstanceLow;
	return &inst == w.instance[1] ? InstanceHigh : None;
}

bool LaneSelect(const Walk& w, const Inst& inst) {
	const auto* lanes = inst.Arg(0).Resolve().TryInstruction();
	return (inst.GetOpcode() == ValueOpcode::SelectU32 ||
	        inst.GetOpcode() == ValueOpcode::SelectU1) &&
	       lanes != nullptr && LeafOf(w, *lanes) == Lanes;
}

bool PerLane(ValueOpcode op) {
	using Op = ValueOpcode;
	constexpr std::array excluded {Op::ConditionRef,  Op::ReadFirstLane,       Op::ReadLane,
	                               Op::WriteLane,     Op::SubgroupUMinU32,     Op::SubgroupUMaxU32,
	                               Op::SubgroupOrU32, Op::Permlane16U32,       Op::PermuteU32,
	                               Op::BpermuteU32,   Op::SwizzleU32,          Op::DppMoveU32,
	                               Op::DppUpdateU32,  Op::LoadDeviceAddressU32};
	const auto           plain = [](Type type) {
		return type == Type::U1 || type == Type::U32 || type == Type::F32;
	};
	if (!plain(TypeOf(op)) || NumArgsOf(op) == 0 || HasSideEffects(op) ||
	    SharedAccessOf(op) != SharedAccess::None ||
	    std::ranges::find(excluded, op) != excluded.end()) {
		return false;
	}
	for (size_t arg = 0; arg < NumArgsOf(op); ++arg) {
		if (!plain(ArgTypeOf(op, arg))) return false;
	}
	return true;
}

Value Copy(const Walk& w, Value value, std::unordered_map<const Inst*, Value>& copies,
           const Inserter* at = nullptr, std::span<const Value> natives = {}) {
	value            = value.Resolve();
	auto* const inst = value.TryInstruction();
	if (inst == nullptr || !Inside(w, inst->Parent())) return value;
	if (const auto leaf = LeafOf(w, *inst); leaf != None) {
		return at != nullptr ? natives[leaf] : value;
	}
	if (const auto found = copies.find(inst); found != copies.end()) return found->second;
	Value copy;
	if (LaneSelect(w, *inst)) {
		copy = Copy(w, inst->Arg(1), copies, at, natives);
	} else if (PerLane(inst->GetOpcode())) {
		std::vector<Value> args;
		for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
			args.push_back(Copy(w, inst->Arg(arg), copies, at, natives));
		}
		if (std::ranges::none_of(args, &Value::IsEmpty)) {
			copy = at != nullptr ? (*at)(*inst, args) : value;
		}
	}
	copies.emplace(inst, copy);
	return copy;
}

Value CullMask(const Walk& w, Value test) {
	const auto* inst = test.Resolve().TryInstruction();
	for (uint32_t arg = 0; inst != nullptr && arg < 2; ++arg) {
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			if (const auto mask = CullMask(w, inst->Arg(arg)); mask != Value(0xffu)) return mask;
			continue;
		}
		const auto* bits =
		    inst->GetOpcode() == ValueOpcode::INotEqual32 && Immediate(inst->Arg(1 - arg)) == 0u
		        ? Producer(inst->Arg(arg), ValueOpcode::BitwiseAnd32)
		        : nullptr;
		for (uint32_t side = 0; bits != nullptr && side < 2; ++side) {
			const auto* shift = Producer(bits->Arg(side), ValueOpcode::ShiftRightLogical32);
			const auto* mask  = bits->Arg(1 - side).Resolve().TryInstruction();
			if (shift != nullptr && shift->Arg(0) == Value(w.flags) &&
			    Immediate(shift->Arg(1)) == 8u && (mask == nullptr || !Inside(w, mask->Parent()))) {
				return bits->Arg(1 - side);
			}
		}
	}
	return Value(0xffu);
}

bool MatchWalk(const Program& program, Inst& bvh, std::span<Inst* const> intersections, Walk& w,
               bool tracked = true) {
	Loop records, blases;
	if (!Enter(program, EnclosingLoop(program, bvh.Parent()), records) ||
	    !Enter(program, EnclosingLoop(program, records.header), blases) ||
	    !Enter(program, EnclosingLoop(program, blases.header), w) ||
	    w.header->terminator.kind != CFG::TerminatorKind::Branch ||
	    w.header->terminator.true_block == nullptr || w.latch->ImmPredecessors().size() != 1) {
		return false;
	}
	auto* enter = w.header->terminator.true_block;
	auto* vote  = w.latch->ImmPredecessors()[0];
	w.skip      = Producer(enter->condition, ValueOpcode::ConditionRef);
	w.leave     = Producer(vote->condition, ValueOpcode::ConditionRef);
	if (w.skip == nullptr || w.leave == nullptr ||
	    enter->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
	    enter->terminator.false_block != vote ||
	    vote->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
	    vote->terminator.false_block != w.latch || Inside(w, vote->terminator.true_block)) {
		return false;
	}
	const auto inside = [&](const Block* block) { return Inside(w, block); };
	for (const auto* block: w.loop) {
		if (block != vote && !std::ranges::all_of(block->ImmSuccessors(), inside)) return false;
		for (const auto& inst: *block) {
			if (HasSideEffects(inst.GetOpcode()) && inst.GetOpcode() != ValueOpcode::Reference) {
				return false;
			}
		}
	}
	const auto* body = Producer(w.skip->Arg(0), ValueOpcode::LogicalAnd);
	for (uint32_t arg = 0; body != nullptr && arg < 2; ++arg) {
		const auto* lanes = body->Arg(arg).Resolve().TryInstruction();
		if (lanes != nullptr && !Inside(w, lanes->Parent())) w.lanes = body->Arg(arg);
	}
	auto* exec = HeaderPhi(records, bvh.Arg(2));
	if (w.lanes.IsEmpty() || exec == nullptr) return false;
	w.bvh      = &bvh;
	w.lanes_in = {w.skip->Arg(0).Resolve().TryInstruction(), exec,
	              Entry(records, *exec).Resolve().TryInstruction()};

	uint32_t found = 0;
	for (const auto* other: intersections) {
		if (other->Flags<uint32_t>() != 1u || EnclosingLoop(program, other->Parent()) != w.header) {
			continue;
		}
		const auto* words = Producer(other->Arg(0), ValueOpcode::CompositeConstructU32x4);
		const auto* tlas  = words != nullptr ? DescriptorAddress(*words) : nullptr;
		const auto* ray   = Producer(other->Arg(1), ValueOpcode::MakeImageAddress);
		const auto  low   = tlas != nullptr ? Slot(tlas->Arg(0)) : std::nullopt;
		const auto  high  = tlas != nullptr ? Slot(tlas->Arg(1)) : std::nullopt;
		if (tlas == nullptr || (tracked && (!low || !high)) || ray == nullptr || ++found > 1) {
			return false;
		}
		w.tlas  = {low.value_or(0u), high.value_or(0u)};
		w.walk  = other;
		w.words = {tlas->Arg(0), tlas->Arg(1)};
		for (const auto word: w.words) {
			const auto* value = tracked ? nullptr : word.Resolve().TryInstruction();
			if (value != nullptr &&
			    BlockIndex(program, value->Parent()) >= BlockIndex(program, w.header)) {
				return false;
			}
		}
		for (uint32_t arg = 1; arg < 8; ++arg) {
			const auto* value = ray->Arg(arg).Resolve().TryInstruction();
			if (value != nullptr &&
			    BlockIndex(program, value->Parent()) >= BlockIndex(program, w.header)) {
				return false;
			}
			w.ray[(arg + 5) % 7] = ray->Arg(arg);
		}
	}
	if (found == 0) return false;

	for (auto* block: records.loop) {
		const auto* ref = Producer(block->condition, ValueOpcode::ConditionRef);
		if (ref == nullptr || block->terminator.kind != CFG::TerminatorKind::ConditionalBranch) {
			continue;
		}
		std::vector<std::pair<const Inst*, Value>> selects;
		bool                                       kept = true;
		for (auto& inst: *block->terminator.true_block) {
			if (inst.GetOpcode() != ValueOpcode::SelectU32 || inst.Arg(0) != ref->Arg(0)) continue;
			const auto* old = HeaderPhi(records, inst.Arg(2));
			kept            = kept && old != nullptr;
			selects.emplace_back(old, inst.Arg(1));
		}
		if (selects.empty() || !kept) continue;
		if (w.commit != nullptr) return false;
		w.commit = ref->Arg(0).Resolve().TryInstruction();
		for (const auto& [old, committed]: selects) {
			const auto* outer = HeaderPhi(blases, Entry(records, *old));
			auto*       value = outer != nullptr ? HeaderPhi(w, Entry(blases, *outer)) : nullptr;
			if (value == nullptr) return false;
			Inst* exit = nullptr;
			for (auto& inst: *vote) {
				if (inst.GetOpcode() != ValueOpcode::Phi ||
				    IncomingValue(inst, enter) != Value(value)) {
					continue;
				}
				if (exit != nullptr) return false;
				exit = &inst;
			}
			if (exit == nullptr) return false;
			w.commits.emplace_back(exit, committed);
		}
	}
	if (w.commit == nullptr) return false;

	for (auto* block: w.loop) {
		for (auto& inst: *block) {
			if (AddressWord(program, &inst, InstanceFlags) == nullptr ||
			    (!tracked && block->terminator.true_block != blases.preheader)) {
				continue;
			}
			if (w.flags != nullptr) return false;
			w.flags = &inst;
		}
	}
	const auto* address = AddressWord(program, w.flags, InstanceFlags);
	const auto* ref     = address != nullptr
	                          ? Producer(w.flags->Parent()->condition, ValueOpcode::ConditionRef)
	                          : nullptr;
	if (ref == nullptr || w.flags->Parent()->terminator.true_block != blases.preheader) {
		return false;
	}
	w.instance = {Producer(address->Arg(0), ValueOpcode::IAdd32),
	              address->Arg(1).Resolve().TryInstruction()};
	w.accepted = ref->Arg(0).Resolve().TryInstruction();
	if (w.instance[0] == nullptr || w.instance[1] == nullptr || w.accepted == nullptr ||
	    !InstanceArray(*w.instance[0], w.tlas[0], w.words[0], tracked)) {
		return false;
	}

	const auto instance = [&](const Inst* at) {
		return at != nullptr && at->Arg(0) == address->Arg(0) && at->Arg(1) == address->Arg(1);
	};
	const auto blas = [&](std::array<const Inst*, 2> base) {
		if (instance(AddressWord(program, base[0], InstanceBlas)) &&
		    instance(AddressWord(program, base[1], InstanceBlas + 4))) {
			return true;
		}
		const auto* wide = !tracked && base[0] != nullptr &&
		                           base[0]->GetOpcode() == ValueOpcode::CompositeExtractU64
		                       ? base[0]->Arg(0).Resolve().TryInstruction()
		                       : nullptr;
		const auto words = WideBlasWords(program, wide);
		return instance(words.instance) && words.Base(base[0], 0) && words.Base(base[1], 1);
	};
	for (auto& phi: *records.header) {
		if (phi.GetOpcode() != ValueOpcode::Phi) continue;
		bool word1 = phi.NumArgs() != 0;
		for (size_t arg = 0; arg < phi.NumArgs(); ++arg) {
			const auto* read = Producer(phi.Arg(arg), ValueOpcode::ReadConstBuffer);
			const auto  base =
			    read != nullptr ? RecordBase(read->Arg(0)) : std::array<const Inst*, 2> {};
			word1 = word1 && blas(base) &&
			        program.memory_info[read->Flags<MemoryFlags>().index].offset == 4u;
		}
		if (!word1) continue;
		if (w.word1 != nullptr) return false;
		w.word1 = &phi;
	}

	w.blas = blases;
	return w.word1 != nullptr;
}

bool WalkCopies(const Walk& w) {
	std::unordered_map<const Inst*, Value> copies;
	const auto copied = [&](Value value) { return !Copy(w, value, copies).IsEmpty(); };
	return copied(Value(w.accepted)) && copied(Value(w.commit)) &&
	       std::ranges::all_of(w.commits,
	                           [&](const auto& commit) { return copied(commit.second); });
}

void Rewrite(Program& program, Traversal& t) {
	using Op = ValueOpcode;
	const Inserter entry {t.preheader, t.preheader->end()};
	const auto     lanes   = Entry(t, *t.exec);
	const auto     valid   = entry(Op::RayQueryValid, {});
	const auto     invalid = entry(Op::LogicalNot, {valid});
	const auto     start   = entry(Op::LogicalAnd, {valid, lanes});
	entry(Op::RayQueryInitialize, {t.ray[0], t.ray[1], t.ray[2], t.ray[3], t.ray[4], t.ray[5],
	                               t.ray[6], Value(t.cull_mask), start});
	const auto entering = entry(Op::SelectU1, {valid, entry(Op::RayQueryProceed, {start}), lanes});
	for (auto* lane: t.lanes) {
		lane->SetArg(Incoming(*lane, t.preheader), entering);
	}

	auto*          body = t.header->terminator.true_block;
	const Inserter at {body, std::ranges::find_if(*body, [](const Inst& inst) {
		                   return inst.GetOpcode() != Op::Phi;
	                   })};
	const auto     current   = at(Op::LogicalAnd, {valid, Value(t.exec)});
	const auto     candidate = at(Op::RayQueryCandidate, {current});
	const auto     instance  = at(Op::RayQueryCandidateInstance, {current});
	const auto     part      = [&](Op op, Value value, uint32_t index) {
		return at(op, {value, Value(index)});
	};
	const auto instance_low  = part(Op::CompositeExtractU64, instance, 0);
	const auto instance_high = part(Op::CompositeExtractU64, instance, 1);
	const auto offset        = at(
	    Op::ShiftLeftLogical64,
	    {at(Op::CompositeConstructU64, {part(Op::CompositeExtractU32x4, candidate, 3), Value(0u)}),
		 Value(3u)});
	const auto choose = [&](Inst& phi, Value native) {
		const auto chosen = at(Op::SelectU32, {valid, native, Value(&phi)});
		Redirect(phi, chosen, *chosen.Instruction());
	};
	const auto info = [&](uint32_t word) {
		if (t.instance[0] == nullptr) return t.loads[word]->Flags<uint64_t>();
		const MemoryFlags flags {.index = static_cast<uint32_t>(program.memory_info.size()),
		                         .pc    = t.instance[2]->Flags<MemoryFlags>().pc};
		program.memory_info.push_back(
		    {.kind = ResourceKind::Global, .offset = word * 4u, .address_is_full = true});
		return std::bit_cast<uint64_t>(flags);
	};
	const std::array words {info(0), info(1)};
	const auto       load = [&](Value low, Value high, uint64_t flags) {
		return at(Op::LoadAddressU32, {at(Op::GetAddressResource, {low, high}), low, high, current},
		          flags);
	};
	const std::array blas {load(instance_low, instance_high, words[0]),
	                       load(instance_low, instance_high, words[1])};
	const auto record = at(Op::IAdd64, {at(Op::CompositeConstructU64, {blas[0], blas[1]}), offset});
	const auto low    = part(Op::CompositeExtractU64, record, 0);
	const auto high   = part(Op::CompositeExtractU64, record, 1);
	choose(*t.state[0], load(low, high, words[0]));
	choose(*t.state[1], load(low, high, words[1]));
	if (t.instance[0] == nullptr) {
		choose(*t.state[2], instance_low);
		choose(*t.state[3], instance_high);
		choose(*t.state[4], load(instance_low, instance_high, t.loads[2]->Flags<uint64_t>()));
	} else {
		const std::array native {instance_low, instance_high, blas[0], blas[1]};
		for (size_t index = 0; index < native.size(); ++index) {
			auto&      value  = *t.instance[index];
			const auto chosen = at(Op::SelectU32, {valid, native[index], Value(&value)});
			const auto uses   = value.Uses();
			for (const auto& use: uses) {
				if (use.user != chosen.Instruction() && Replaced(t, *use.user)) {
					use.user->SetArg(use.operand, chosen);
				}
			}
		}
		for (auto* scalar: t.scalar) {
			auto& memory           = program.memory_info[scalar->Flags<MemoryFlags>().index];
			memory.kind            = ResourceKind::Global;
			memory.component_index = 0;
			memory.component_count = 1;
			const Inserter before {scalar->Parent(), Position(*scalar)};
			scalar->SetArg(3, before(Op::SelectU1, {valid, current, scalar->Arg(3)}));
		}
		const Inserter before {t.record->Parent(), Position(*t.record)};
		t.record->SetArg(4, before(Op::LogicalAnd, {t.record->Arg(4), invalid}));
	}

	const std::array result {part(Op::CompositeExtractU32x4, candidate, 0), Value(One),
	                         part(Op::CompositeExtractU32x4, candidate, 1),
	                         part(Op::CompositeExtractU32x4, candidate, 2)};
	const Inserter   intersect {t.bvh->Parent(), Position(*t.bvh)};
	t.bvh->SetArg(2, intersect(Op::LogicalAnd, {Value(t.exec), invalid}));
	const auto uses = t.bvh->Uses();
	for (const auto& use: uses) {
		auto&          extract = *use.user;
		const Inserter after {extract.Parent(), std::next(Position(extract))};
		const auto     chosen =
		    after(Op::SelectU32, {valid, result[extract.Arg(1).U32()], Value(&extract)});
		Redirect(extract, chosen, *chosen.Instruction());
	}

	const Inserter vote {t.vote->Parent(), Position(*t.vote)};
	const auto     moved = vote(Op::INotEqual32, {Back(t, *t.extent), Value(t.extent)});
	vote(Op::RayQueryConfirm, {vote(Op::LogicalAnd, {current, moved})});
	const auto alive =
	    vote(Op::SelectU1, {valid, vote(Op::RayQueryProceed, {current}), t.vote->Arg(0)});
	t.vote->SetArg(0, alive);
	for (auto* lane: t.lanes) {
		lane->SetArg(Incoming(*lane, t.latch), alive);
	}
}

void RewriteWalk(Program& program, Walk& w) {
	using Op = ValueOpcode;
	const Inserter entry {w.preheader, w.preheader->end()};
	const auto     valid = entry(Op::RayQueryValid, {});
	const auto     start = entry(Op::LogicalAnd, {valid, w.lanes});
	entry(Op::RayQueryInitialize, {w.ray[0], w.ray[1], w.ray[2], w.ray[3], w.ray[4], w.ray[5],
	                               w.ray[6], CullMask(w, Value(w.accepted)), start});
	const Inserter skip {w.skip->Parent(), Position(*w.skip)};
	w.skip->SetArg(0, skip(Op::LogicalAnd, {w.skip->Arg(0), skip(Op::LogicalNot, {valid})}));
	auto& alive = *w.header->PrependNewInst(w.header->begin(), Op::Phi);
	alive.SetFlags(Type::U1);
	alive.AddPhiOperand(w.preheader, start);

	auto*          vote = w.leave->Parent();
	const Inserter at {vote, std::ranges::find_if(*vote, [](const Inst& inst) {
		                   return inst.GetOpcode() != Op::Phi;
	                   })};
	const auto     current   = at(Op::RayQueryProceed, {Value(&alive)});
	const auto     candidate = at(Op::RayQueryCandidate, {current});
	const auto     with      = [&](Op op, Value value, uint32_t immediate) {
		return at(op, {value, Value(immediate)});
	};
	const auto halves = [&](Value address) {
		return std::array {with(Op::CompositeExtractU64, address, 0),
		                   with(Op::CompositeExtractU64, address, 1)};
	};
	const auto load = [&](const std::array<Value, 2>& address, uint32_t offset) {
		const MemoryFlags flags {.index = static_cast<uint32_t>(program.memory_info.size()),
		                         .pc    = w.flags->Flags<MemoryFlags>().pc};
		program.memory_info.push_back(
		    {.kind = ResourceKind::Global, .offset = offset, .address_is_full = true});
		const auto& [low, high] = address;
		return at(Op::LoadAddressU32, {at(Op::GetAddressResource, {low, high}), low, high, current},
		          std::bit_cast<uint64_t>(flags));
	};
	const auto u64 = [&](Value low) { return at(Op::CompositeConstructU64, {low, Value(0u)}); };
	const auto instance = halves(at(Op::RayQueryCandidateInstance, {current}));
	const auto blas     = at(Op::CompositeConstructU64,
	                         {load(instance, InstanceBlas), load(instance, InstanceBlas + 4)});
	const auto record =
	    halves(at(Op::IAdd64, {blas, with(Op::ShiftLeftLogical64,
		                                  u64(with(Op::CompositeExtractU32x4, candidate, 3)), 3)}));
	const auto triangle = load(record, 0);
	const auto vertices =
	    halves(at(Op::IAdd64, {blas, with(Op::ShiftLeftLogical64,
		                                  u64(with(Op::ShiftRightLogical32, triangle, 3)), 6)}));
	const auto shift = with(Op::ShiftLeftLogical32, with(Op::BitwiseAnd32, triangle, 7), 3);
	const auto order =
	    with(Op::BitwiseAnd32, at(Op::ShiftRightLogical32, {load(vertices, 60), shift}), 0xf);
	const auto odd =
	    with(Op::INotEqual32,
		     with(Op::BitwiseAnd32, at(Op::ShiftRightLogical32, {Value(OddRemaps), order}), 1), 0);
	const auto faces = at(Op::LogicalXor, {at(Op::RayQueryCandidateFrontFace, {current}), odd});
	const auto t     = with(Op::CompositeExtractU32x4, candidate, 0);
	const std::array<Value, None> natives {
	    current,
	    at(Op::SelectU32, {faces, t, with(Op::BitwiseOr32, t, 0x80000000u)}),
	    at(Op::SelectU32, {faces, Value(One), Value(0xbf800000u)}),
	    load(instance, InstanceFlags),
	    load(record, 4),
	    instance[0],
	    instance[1]};

	std::unordered_map<const Inst*, Value> copies;
	const auto copy = [&](Value value) { return Copy(w, value, copies, &at, natives); };
	const auto accept =
	    at(Op::LogicalAnd,
		   {current, at(Op::LogicalAnd, {copy(Value(w.accepted)), copy(Value(w.commit))})});
	for (const auto& [exit, value]: w.commits) {
		const auto chosen = at(Op::SelectU32, {accept, copy(value), Value(exit)});
		Redirect(*exit, chosen, *chosen.Instruction());
	}
	const auto later = at(Op::LogicalAnd, {current, at(Op::LogicalNot, {accept})});
	alive.AddPhiOperand(w.latch, later);

	const Inserter leave {vote, Position(*w.leave)};
	const auto     any =
	    leave(Op::ConditionRef, {later}, static_cast<uint64_t>(CFG::BranchCondition::ExecNonZero));
	w.leave->SetArg(0, leave(Op::SelectU1, {valid, leave(Op::LogicalNot, {any}), w.leave->Arg(0)}));
}

}

uint32_t TraceTraversalsWithRayQueries(Program& program) {
	if (program.stage != ShaderType::Compute) return 0;
	std::vector<Inst*> intersections;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::BvhIntersect) intersections.push_back(&inst);
		}
	}
	uint32_t traced = 0;
	for (auto* bvh: intersections) {
		if ((bvh->Flags<uint32_t>() & BvhIntersectNative) != 0) continue;
		Traversal   traversal;
		Walk        walk;
		const bool  nested  = bvh->Flags<uint32_t>() == 1u;
		const bool  matched = nested ? MatchNested(program, *bvh, intersections, traversal) &&
		                                   NestedReads(program, traversal)
		                             : Match(program, *bvh, traversal);
		const bool  walked  = !matched && nested &&
		                      MatchWalk(program, *bvh, intersections, walk) && WalkCopies(walk);
		const auto& tlas    = walked ? walk.tlas : traversal.tlas;
		if ((matched || walked) && (traced == 0 || tlas == program.info.ray_query_tlas)) {
			walked ? RewriteWalk(program, walk) : Rewrite(program, traversal);
			program.info.ray_query_tlas = tlas;
			++traced;
		} else if (EnclosingLoop(program, bvh->Parent()) != nullptr) {
			Log::WriteToConsoleAndLog(
			    fmt::format("ShaderRecompiler BVH traversal left to emulation: hash=0x{:016x}\n",
				            program.shader_hash));
		}
	}
	return traced;
}

namespace {


const MemoryInfo& MemoryOf(const Program& program, const Inst& inst) {
	return program.memory_info[inst.Flags<MemoryFlags>().index];
}

Value Canonical(Value value) {
	for (uint32_t depth = 0; depth < 16; ++depth) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) break;
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::BitCastF32U32 || op == ValueOpcode::BitCastU32F32 ||
		    op == ValueOpcode::SelectU32) {
			value = inst->Arg(op == ValueOpcode::SelectU32 ? 1 : 0);
			continue;
		}
		break;
	}
	return value;
}

template <typename First>
std::optional<std::pair<Value, Value>> Operands(Value value, ValueOpcode op, First&& first) {
	const auto* inst = Producer(Canonical(value), op);
	if (inst == nullptr) return std::nullopt;
	for (uint32_t side = 0; side < 2; ++side) {
		if (first(inst->Arg(side))) return std::pair {inst->Arg(side), inst->Arg(1 - side)};
	}
	return std::nullopt;
}

Value Term(Value value) {
	value            = Canonical(value);
	const auto* inst = value.TryInstruction();
	return inst != nullptr && inst->GetOpcode() == ValueOpcode::FPNeg32 ? Canonical(inst->Arg(0))
	                                                                    : value;
}

struct AddressLoad {
	Value    low;
	Value    high;
	uint32_t offset = 0;
};

std::optional<AddressLoad> LoadOf(const Program& program, Value value) {
	const auto* load = Producer(value, ValueOpcode::LoadAddressU32);
	if (load == nullptr || Immediate(load->Arg(1)) != 0u || Immediate(load->Arg(2)) != 0u)
		return std::nullopt;
	const auto* handle = Producer(load->Arg(0), ValueOpcode::GetAddressResource);
	if (handle == nullptr) return std::nullopt;
	return AddressLoad {handle->Arg(0).Resolve(), handle->Arg(1).Resolve(),
	                    MemoryOf(program, *load).offset};
}

const Inst* DescriptorAddress(Value word) {
	const auto* low  = Producer(word, ValueOpcode::CompositeExtractU64);
	const auto* mask = low != nullptr ? Producer(low->Arg(0), ValueOpcode::BitwiseAnd64) : nullptr;
	const auto* shift =
	    mask != nullptr ? Producer(mask->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
	if (shift == nullptr || Immediate(low->Arg(1)) != 0u || Immediate(shift->Arg(1)) != 8u)
		return nullptr;
	return shift->Arg(0).Resolve().TryInstruction();
}

std::optional<uint32_t> UserRegister(Value value) {
	const auto* user = Producer(value, ValueOpcode::GetUserData);
	if (user == nullptr) return std::nullopt;
	return static_cast<uint32_t>(user->Arg(0).ScalarRegister());
}

const Inst* BallotOf(Value mask) {
	const auto* low = Producer(mask, ValueOpcode::CompositeExtractU32x4);
	if (low == nullptr || Immediate(low->Arg(1)) != 0u) return nullptr;
	return Producer(low->Arg(0), ValueOpcode::Ballot);
}

Inst* LoopPhi(Value value, const Block* header) {
	auto* inst = Producer(value, ValueOpcode::Phi);
	return inst != nullptr && inst->Parent() == header ? inst : nullptr;
}

Block::iterator AfterPhis(Block& block) {
	return std::ranges::find_if(
	    block, [](const Inst& inst) { return inst.GetOpcode() != ValueOpcode::Phi; });
}

struct BlasWalk {
	Block*             preheader = nullptr;
	Block*             header    = nullptr;
	Block*             latch     = nullptr;
	Inst*              node      = nullptr;
	Inst*              word      = nullptr;
	const Inst*        load      = nullptr;
	Inst*              vote      = nullptr;
	std::vector<Inst*> lanes;
};

struct LaneMaskWalk {
	Block*                preheader = nullptr;
	Block*                header    = nullptr;
	Block*                latch     = nullptr;
	Block*                merge     = nullptr;
	Block*                exit      = nullptr;
	Inst*                 bvh       = nullptr;
	Inst*                 node      = nullptr;
	Inst*                 pair_mask = nullptr;
	Inst*                 exit_ref  = nullptr;
	Value                 lanes;
	Value                 lanes_without;
	std::array<Value, 7>  ray {};
	std::array<Value, 2>  tlas {};
	uint32_t              cull_mask = 0xff;
	Value                 mask_value;
	bool                  masked        = false;
	bool                  mask_optional = false;
	std::vector<BlasWalk> blases;
	Block*                wave_header    = nullptr;
	Block*                wave_latch     = nullptr;
	Value                 wave_root;
	Inst*                 wave_node      = nullptr;
	Inst*                 wave_box       = nullptr;
	Inst*                 wave_triangles = nullptr;
	Inst*                 wave_copy      = nullptr;
	AddressLoad           record;
	RayTracingLayout      layout;
	NativeTraversals      source;
};

Value RayMask(const LaneMaskWalk& w) {
	return w.mask_value.IsEmpty() ? Value(w.cull_mask) : w.mask_value;
}

std::pair<Inst*, bool> BranchCondition(const Block& block) {
	if (auto* ref = Producer(block.condition, ValueOpcode::ConditionRef); ref != nullptr) return {ref, false};
	const auto* negated = Producer(block.condition, ValueOpcode::LogicalNot);
	return {negated != nullptr ? Producer(negated->Arg(0), ValueOpcode::ConditionRef) : nullptr, true};
}

Value LaneMaskOf(const Inst* lane) {
	if (lane == nullptr || lane->GetOpcode() != ValueOpcode::INotEqual32) return {};
	for (uint32_t side = 0; side < 2; ++side) {
		const auto* bit = Producer(lane->Arg(side), ValueOpcode::BitwiseAnd32);
		if (bit == nullptr || Immediate(lane->Arg(1 - side)) != 0u) continue;
		for (uint32_t b = 0; b < 2; ++b) {
			const auto* shift = Producer(bit->Arg(b), ValueOpcode::ShiftRightLogical32);
			const auto* index = shift != nullptr ? Producer(shift->Arg(1), ValueOpcode::BitwiseAnd32) : nullptr;
			if (index == nullptr || Immediate(bit->Arg(1 - b)) != 1u) continue;
			for (uint32_t i = 0; i < 2; ++i) {
				if (Immediate(index->Arg(1 - i)) == 31u && Producer(index->Arg(i), ValueOpcode::LaneId) != nullptr)
					return shift->Arg(0).Resolve();
			}
		}
	}
	return {};
}

std::optional<std::pair<uint32_t, bool>> TypeTest(Value condition, const Inst* node, uint32_t depth = 0) {
	const auto* test = condition.Resolve().TryInstruction();
	if (test != nullptr && test->GetOpcode() == ValueOpcode::LogicalAnd && depth < 4) {
		const auto a = TypeTest(test->Arg(0), node, depth + 1), b = TypeTest(test->Arg(1), node, depth + 1);
		return a && !b ? a : b && !a ? b : std::nullopt;
	}
	if (const auto mask = LaneMaskOf(test); !mask.IsEmpty() && depth < 4) {
		std::vector<Value> parts {mask};
		std::optional<std::pair<uint32_t, bool>> found;
		for (size_t at = 0; at < parts.size() && at < 16; ++at) {
			const auto* part = parts[at].Resolve().TryInstruction();
			if (part == nullptr) continue;
			if (part->GetOpcode() == ValueOpcode::BitwiseAnd32) {
				parts.push_back(part->Arg(0));
				parts.push_back(part->Arg(1));
			} else if (part->GetOpcode() == ValueOpcode::SelectU32 && Immediate(part->Arg(2)) == 0u) {
				if (const auto type = TypeTest(part->Arg(0), node, depth + 1)) {
					if (found) return std::nullopt;
					found = type;
				}
			}
		}
		return found;
	}
	if (test == nullptr ||
	    (test->GetOpcode() != ValueOpcode::IEqual32 && test->GetOpcode() != ValueOpcode::INotEqual32))
		return std::nullopt;
	for (uint32_t side = 0; side < 2; ++side) {
		const auto  type = Immediate(test->Arg(side));
		const auto* bits = Producer(test->Arg(1 - side), ValueOpcode::BitwiseAnd32);
		if (!type || *type >= 8u || bits == nullptr) continue;
		for (uint32_t b = 0; b < 2; ++b) {
			if (Immediate(bits->Arg(b)) == 7u && bits->Arg(1 - b).Resolve() == Value(const_cast<Inst*>(node)))
				return std::pair {*type, test->GetOpcode() == ValueOpcode::IEqual32};
		}
	}
	return std::nullopt;
}

std::optional<uint32_t> EdgeType(const Block* to, const Block* from, const Inst* node) {
	for (uint32_t depth = 0; depth < 4 && from != nullptr; ++depth) {
		if (from->terminator.kind == CFG::TerminatorKind::ConditionalBranch) {
			const auto [ref, negated] = BranchCondition(*from);
			const auto test           = ref != nullptr ? TypeTest(ref->Arg(0), node) : std::nullopt;
			const bool taken = (to == from->terminator.true_block) != negated;
			return test && taken == test->second ? std::optional(test->first) : std::nullopt;
		}
		const auto preds = from->ImmPredecessors();
		if (preds.size() != 1) return std::nullopt;
		to   = from;
		from = preds[0];
	}
	return std::nullopt;
}

bool InLoop(const Program& program, const Block* block, const Block* header, const Block* latch) {
	const auto at = BlockIndex(program, block);
	return at >= BlockIndex(program, header) && at <= BlockIndex(program, latch);
}

bool DefinedBefore(const Program& program, Value value, const Block* header) {
	const auto* inst = value.Resolve().TryInstruction();
	return !value.IsEmpty() &&
	       (inst == nullptr || BlockIndex(program, inst->Parent()) < BlockIndex(program, header));
}

bool DescriptorDefinedBefore(const Program& program, Value value, const Block* header) {
	if (DefinedBefore(program, value, header)) return true;
	const auto* load   = Producer(value, ValueOpcode::LoadAddressU32);
	const auto* handle = load != nullptr ? Producer(load->Arg(0), ValueOpcode::GetAddressResource) : nullptr;
	return handle != nullptr && Immediate(load->Arg(1)) == 0u && Immediate(load->Arg(2)) == 0u &&
	       DefinedBefore(program, handle->Arg(0), header) && DefinedBefore(program, handle->Arg(1), header);
}

Value DescriptorBefore(Program& program, Value value, const Block* header, const Inserter& at) {
	if (DefinedBefore(program, value, header)) return value;
	const auto* load   = Producer(value, ValueOpcode::LoadAddressU32);
	const auto* handle = Producer(load->Arg(0), ValueOpcode::GetAddressResource);
	const auto  again  = at(ValueOpcode::GetAddressResource, {handle->Arg(0), handle->Arg(1)}, handle->Flags<uint64_t>());
	auto        memory = load->Flags<MemoryFlags>();
	program.memory_info.push_back(program.memory_info[memory.index]);
	memory.index      = static_cast<uint32_t>(program.memory_info.size() - 1);
	const auto loaded = at(ValueOpcode::LoadAddressU32, {again, load->Arg(1), load->Arg(2), load->Arg(3)});
	loaded.Instruction()->SetFlags(memory);
	return loaded;
}

bool LoopShape(Block* header, Block*& preheader, Block*& latch, Block*& merge) {
	if (header == nullptr || header->terminator.continue_block == nullptr ||
	    header->terminator.merge_block == nullptr) {
		return false;
	}
	latch              = header->terminator.continue_block;
	merge              = header->terminator.merge_block;
	const auto entries = header->ImmPredecessors();
	if (entries.size() != 2 || std::ranges::find(entries, latch) == entries.end()) return false;
	preheader = entries[0] == latch ? entries[1] : entries[0];
	return true;
}

std::optional<Value> Hoist(const Program& program, Value value, const Block* header,
                           const Inserter& at, uint32_t depth = 0) {
	value      = value.Resolve();
	auto* inst = value.TryInstruction();
	if (inst == nullptr) return value;
	if (BlockIndex(program, inst->Parent()) < BlockIndex(program, header)) return value;
	if (depth > 4) return std::nullopt;
	const auto op = inst->GetOpcode();
	if (op == ValueOpcode::SelectU32) return Hoist(program, inst->Arg(1), header, at, depth + 1);
	if (op != ValueOpcode::BitCastU32F32 && op != ValueOpcode::BitCastF32U32 &&
	    op != ValueOpcode::ConvertF32S32 && op != ValueOpcode::ConvertF32U32) {
		return std::nullopt;
	}
	const auto arg = Hoist(program, inst->Arg(0), header, at, depth + 1);
	if (!arg) return std::nullopt;
	return at(op, {*arg}, inst->Flags<uint64_t>());
}

using LinearForm           = std::array<uint32_t, 7>;
constexpr uint32_t NoDword = 0xffffffffu;

template <typename Record>
bool Expand(Value value, const std::array<Value, 6>& world, const Record& record, LinearForm& form,
            uint32_t depth = 0) {
	if (depth > 16) return false;
	const auto canonical = Canonical(value);
	const auto term      = [&](Value v) -> std::optional<uint32_t> {
		const auto t = Term(v);
		for (uint32_t k = 0; k < 6; ++k) {
			if (t == world[k]) return k < 3 ? k : k + 1;
		}
		return std::nullopt;
	};
	const auto set = [&](uint32_t slot, uint32_t dword) {
		if (form[slot] != NoDword) return false;
		form[slot] = dword;
		return true;
	};
	const auto product = [&](Value a, Value b) {
		const auto da = record(Term(a)), db = record(Term(b));
		const auto ta = term(a), tb = term(b);
		if (da && tb) return set(*tb, *da);
		if (db && ta) return set(*ta, *db);
		return false;
	};
	if (const auto dword = record(Term(canonical)); dword) return set(3, *dword);
	const auto* inst = canonical.TryInstruction();
	if (inst == nullptr) return false;
	switch (inst->GetOpcode()) {
		case ValueOpcode::FPNeg32: return Expand(inst->Arg(0), world, record, form, depth + 1);
		case ValueOpcode::FPAdd32:
			return Expand(inst->Arg(0), world, record, form, depth + 1) &&
			       Expand(inst->Arg(1), world, record, form, depth + 1);
		case ValueOpcode::FPMul32: return product(inst->Arg(0), inst->Arg(1));
		case ValueOpcode::FPMad32:
			return product(inst->Arg(0), inst->Arg(1)) &&
			       Expand(inst->Arg(2), world, record, form, depth + 1);
		default: return false;
	}
}

bool DeriveLaneLayout(const Program& program, LaneMaskWalk& w) {
	auto& layout = w.layout;
	Value       root_value;
	const Inst* intersect = nullptr;
	if (!w.blases.empty()) {
		const auto& blas = w.blases.front();
		root_value       = IncomingValue(*blas.node, blas.preheader);
		for (const auto& use: blas.node->Uses()) {
			if (use.user->GetOpcode() == ValueOpcode::MakeImageAddress && use.operand == 0)
				intersect = use.user;
		}
	} else if (w.wave_box != nullptr) {
		root_value = w.wave_root;
		intersect  = Producer(w.wave_box->Arg(1), ValueOpcode::MakeImageAddress);
	}
	layout.inline_type = NoField;
	auto root          = LoadOf(program, Canonical(root_value));
	uint32_t inline_root = NoField;
	if (!root && !root_value.IsEmpty()) {
		const auto* inst = root_value.Resolve().TryInstruction();
		const auto  op   = inst != nullptr ? inst->GetOpcode() : ValueOpcode::Phi;
		for (uint32_t k = 0; inst != nullptr && k < 2 && !root; ++k) {
			const auto constant = op == ValueOpcode::SelectU32 ? Immediate(inst->Arg(1 + k))
			                      : op == ValueOpcode::Phi && inst->NumArgs() == 2 ? Immediate(inst->Arg(k))
			                                                                       : std::nullopt;
			if (!constant) continue;
			root        = LoadOf(program, Canonical(inst->Arg(op == ValueOpcode::SelectU32 ? 2 - k : 1 - k)));
			inline_root = *constant;
		}
	}
	if (!root || intersect == nullptr) return false;
	AddressLoad record_pair = *root;
	const auto  inline_phi  = [&](const Inst* phi, uint32_t k) {
		const auto* fixed = phi->NumArgs() == 2 ? Producer(phi->Arg(k), ValueOpcode::IAdd32) : nullptr;
		for (uint32_t side = 0; fixed != nullptr && side < 2; ++side) {
			const auto offset = Immediate(fixed->Arg(side));
			const auto type   = offset && fixed->Arg(1 - side).Resolve() == w.tlas[0]
			                        ? EdgeType(phi->Parent(), phi->PhiBlock(k), w.node)
			                        : std::nullopt;
			if (!type) continue;
			layout.inline_type   = *type;
			layout.inline_offset = *offset;
			layout.inline_root   = inline_root;
			return true;
		}
		return false;
	};
	const auto* sum = Producer(root->low, ValueOpcode::IAdd32);
	if (const auto* phi = Producer(root->low, ValueOpcode::Phi); phi != nullptr && inline_root != NoField) {
		for (uint32_t k = 0; k < 2 && sum == nullptr; ++k) {
			if (inline_phi(phi, k)) sum = Producer(phi->Arg(1 - k), ValueOpcode::IAdd32);
		}
	} else if (sum != nullptr && inline_root != NoField) {
		for (const auto& use: sum->Uses()) {
			if (use.user->GetOpcode() != ValueOpcode::Phi || !inline_phi(use.user, 1 - use.operand)) continue;
			for (const auto& address: use.user->Uses()) {
				if (address.user->GetOpcode() == ValueOpcode::GetAddressResource && address.operand == 0) {
					record_pair.low  = Value(use.user);
					record_pair.high = address.user->Arg(1).Resolve();
				}
			}
			break;
		}
	}
	if (sum == nullptr) return false;
	w.record   = record_pair;
	bool found = false;
	for (uint32_t side = 0; side < 2 && !found; ++side) {
		const auto* part = Producer(sum->Arg(side), ValueOpcode::CompositeExtractU64);
		const auto* scale =
		    part != nullptr ? Producer(part->Arg(0), ValueOpcode::ShiftLeftLogical64) : nullptr;
		const auto* wide = scale != nullptr
		                       ? Producer(scale->Arg(0), ValueOpcode::CompositeConstructU64)
		                       : nullptr;
		const auto* index =
		    wide != nullptr ? Producer(wide->Arg(0), ValueOpcode::ShiftRightLogical32) : nullptr;
		const auto bits   = scale != nullptr ? Immediate(scale->Arg(1)) : std::nullopt;
		uint32_t   stride = 0;
		if (index != nullptr && Immediate(part->Arg(1)) == 0u && Immediate(wide->Arg(1)) == 0u &&
		    Immediate(index->Arg(1)) == 3u && bits && *bits >= 7u && *bits <= 8u &&
		    LoopPhi(index->Arg(0), w.header) == w.node) {
			stride = 1u << *bits;
		} else if (const auto* mul = Producer(sum->Arg(side), ValueOpcode::IMul32); mul != nullptr) {
			for (uint32_t m = 0; m < 2; ++m) {
				const auto  scaled = Immediate(mul->Arg(m));
				const auto* by     = Producer(mul->Arg(1 - m), ValueOpcode::ShiftRightLogical32);
				if (scaled && by != nullptr && Immediate(by->Arg(1)) == 3u && LoopPhi(by->Arg(0), w.header) == w.node &&
				    *scaled % 32u == 0 && *scaled >= 128u && *scaled <= 224u)
					stride = *scaled;
			}
		}
		if (stride == 0) continue;
		const auto* base = Producer(sum->Arg(1 - side), ValueOpcode::IAdd32);
		if (base == nullptr) return false;
		for (uint32_t b = 0; b < 2 && !found; ++b) {
			const auto offset = LoadOf(program, base->Arg(b));
			if (offset && base->Arg(1 - b).Resolve() == w.tlas[0] && offset->low == w.tlas[0] &&
			    offset->high == w.tlas[1]) {
				layout.instance_offset = offset->offset;
				layout.record_stride   = stride;
				found                  = true;
			}
		}
	}
	if (!found || (root->offset & 3u) != 0) return false;
	const auto record = [&](Value value) -> std::optional<uint32_t> {
		const auto load = LoadOf(program, value);
		if (!load || !((load->low == root->low && load->high == root->high) ||
		               (load->low == record_pair.low && load->high == record_pair.high)) ||
		    (load->offset & 3u) != 0) {
			return std::nullopt;
		}
		return load->offset / 4u;
	};
	layout.root_dword = root->offset / 4u;
	std::array<Value, 6> world {};
	for (uint32_t k = 0; k < 6; ++k)
		world[k] = Term(w.ray[k]);
	for (uint32_t row = 0; row < 3; ++row) {
		LinearForm origin, direction;
		origin.fill(NoDword);
		direction.fill(NoDword);
		if (!Expand(intersect->Arg(2 + row), world, record, origin) ||
		    !Expand(intersect->Arg(5 + row), world, record, direction)) {
			return false;
		}
		for (uint32_t c = 0; c < 3; ++c) {
			if (origin[c] == NoDword || origin[4 + c] != NoDword || direction[4 + c] != origin[c] ||
			    direction[c] != NoDword) {
				return false;
			}
			layout.transform[row * 4 + c] = static_cast<uint8_t>(origin[c]);
		}
		if (origin[3] == NoDword || direction[3] != NoDword) return false;
		layout.transform[row * 4 + 3] = static_cast<uint8_t>(origin[3]);
	}
	const Inst* blas_bvh = nullptr;
	for (const auto& use: intersect->Uses()) {
		if (use.user->GetOpcode() == ValueOpcode::BvhIntersect) blas_bvh = use.user;
	}
	const auto* words = blas_bvh != nullptr
	                        ? Producer(blas_bvh->Arg(0), ValueOpcode::CompositeConstructU32x4)
	                        : nullptr;
	const auto* base  = words != nullptr ? DescriptorAddress(words->Arg(0)) : nullptr;
	if (base == nullptr) return false;
	const auto* pair = base;
	if (base->GetOpcode() == ValueOpcode::ShiftLeftLogical64) {
		if (Immediate(base->Arg(1)) != 3u) return false;
		pair               = base->Arg(0).Resolve().TryInstruction();
		layout.blas_packed = true;
	}
	if (pair == nullptr || pair->GetOpcode() != ValueOpcode::CompositeConstructU64) return false;
	const auto low  = record(pair->Arg(0));
	auto       high = pair->Arg(1);
	if (layout.blas_packed) {
		const auto* masked = Producer(high, ValueOpcode::BitwiseAnd32);
		if (masked == nullptr) return false;
		high = Immediate(masked->Arg(1)) == 0x3fffffu ? masked->Arg(0) : masked->Arg(1);
	}
	if (!low || record(high) != *low + 1u) return false;
	layout.blas_dword = *low;
	Value size = words->Arg(2);
	if (const auto* part = Producer(size, ValueOpcode::CompositeExtractU64);
	    part != nullptr && Immediate(part->Arg(1)) == 0u) {
		const auto* masked = Producer(part->Arg(0), ValueOpcode::BitwiseAnd64);
		const auto* wide =
		    masked != nullptr ? Producer(masked->Arg(0), ValueOpcode::CompositeConstructU64) : nullptr;
		if (wide != nullptr) size = wide->Arg(0);
	}
	for (uint32_t depth = 0; depth < 4; ++depth) {
		const auto* add = Producer(size, ValueOpcode::IAdd32);
		if (add == nullptr) break;
		size = Immediate(add->Arg(0)) ? add->Arg(1) : add->Arg(0);
	}
	const auto count = LoadOf(program, size);
	if (!count || (count->offset != 80u && count->offset != 88u)) return false;
	layout.node_count_80 = count->offset == 80u;
	bool masked  = false;
	w.mask_value = {};
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto dword = inst.GetOpcode() == ValueOpcode::LoadAddressU32 ? record(Value(&inst)) : std::nullopt;
			if (!dword || *dword == layout.blas_dword || *dword == layout.blas_dword + 1u) continue;
			for (const auto& use: inst.Uses()) {
				const auto* test  = use.user;
				const auto  shift = use.operand == 0 && ((test->GetOpcode() == ValueOpcode::BitFieldUExtract &&
				                                          Immediate(test->Arg(2)) == 8u) ||
				                                         test->GetOpcode() == ValueOpcode::ShiftRightLogical32)
				                        ? Immediate(test->Arg(1))
				                        : std::nullopt;
				for (const auto& u: test->Uses()) {
					const auto* both  = u.user;
					const auto  value = both->GetOpcode() == ValueOpcode::BitwiseAnd32 ? both->Arg(1 - u.operand) : Value {};
					if (!shift || *shift >= 32u || value.IsEmpty() || value.IsImmediate() ||
					    !DefinedBefore(program, value, w.header) || std::ranges::none_of(both->Uses(), [](const Use& t) {
						    return t.user->GetOpcode() == ValueOpcode::INotEqual32 && Immediate(t.user->Arg(1 - t.operand)) == 0u;
					    })) {
						continue;
					}
					if (masked && (layout.mask_dword != *dword || layout.mask_shift != *shift || w.mask_value != value.Resolve()))
						return false;
					layout.mask_dword = *dword;
					layout.mask_shift = *shift;
					layout.mask_width = 0;
					w.cull_mask       = 0xffu;
					w.mask_value      = value.Resolve();
					masked            = true;
				}
			}
		}
	}
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto dword = inst.GetOpcode() == ValueOpcode::LoadAddressU32
			                       ? record(Value(&inst))
			                       : std::nullopt;
			if (!dword || *dword == layout.blas_dword || *dword == layout.blas_dword + 1u || masked) continue;
			for (const auto& use: inst.Uses()) {
				const auto* test = use.user;
				auto        bits = test->GetOpcode() == ValueOpcode::BitwiseAnd32
				                       ? Immediate(test->Arg(1 - use.operand))
				                       : std::nullopt;
				bool bit = false;
				if (test->GetOpcode() == ValueOpcode::BitFieldUExtract && use.operand == 0 &&
				    Immediate(test->Arg(2)) == 1u && Immediate(test->Arg(1)).value_or(32u) < 32u) {
					bits = 1u << *Immediate(test->Arg(1));
					bit  = true;
				}
				bool above = false;
				if ((test->GetOpcode() == ValueOpcode::UGreaterThan32 ||
				     test->GetOpcode() == ValueOpcode::ULessThanEqual32) &&
				    use.operand == 0) {
					const auto below = Immediate(test->Arg(1));
					if (below && *below != 0u && *below != 0xffffffffu && (*below & (*below + 1u)) == 0) {
						bits  = ~*below;
						above = true;
					}
				}
				const auto in_leaf = [&](const Inst* user) {
					return w.wave_header != nullptr && InLoop(program, user->Parent(), w.wave_header, w.wave_latch);
				};
				if (!bits || *bits == 0u ||
				    (!above && std::ranges::none_of(test->Uses(), [&](const Use& u) {
					     const auto op = u.user->GetOpcode();
					     if (op != ValueOpcode::INotEqual32 && !(bit && op == ValueOpcode::IEqual32)) return false;
					     const auto other = Immediate(u.user->Arg(1 - u.operand));
					     return ((op == ValueOpcode::INotEqual32 && other == 0u) ||
					             (bit && op == ValueOpcode::IEqual32 && other == 1u)) &&
					            !in_leaf(u.user);
				     }))) {
					continue;
				}
				const auto shift = static_cast<uint32_t>(std::countr_zero(*bits));
				auto       mask  = *bits >> shift;
				uint32_t   width = 0;
				if (mask > 0xffu && (mask & (mask + 1u)) == 0) {
					width = static_cast<uint32_t>(std::popcount(mask));
					mask  = 0xffu;
				}
				if ((masked && (layout.mask_dword != *dword || layout.mask_shift != shift ||
				                layout.mask_width != width || w.cull_mask != mask)) ||
				    mask > 0xffu) {
					return false;
				}
				layout.mask_dword = *dword;
				layout.mask_shift = shift;
				layout.mask_width = width;
				w.cull_mask       = mask;
				masked            = true;
			}
		}
	}
	w.masked = masked;
	if (!masked) {
		if (!w.mask_optional) return false;
		layout.mask_dword = NoField;
		layout.mask_shift = 0;
		layout.mask_width = 0;
	}
	layout.valid = true;
	return true;
}

using BranchFacts = std::vector<std::pair<Value, bool>>;

std::pair<Value, bool> Literal(Value value) {
	bool negated = false;
	for (uint32_t depth = 0; depth < 8; ++depth) {
		const auto* inst = Producer(value, ValueOpcode::LogicalNot);
		if (inst == nullptr) break;
		value   = inst->Arg(0);
		negated = !negated;
	}
	return {value.Resolve(), negated};
}

std::optional<bool> Decided(Value value, const BranchFacts& facts, uint32_t depth = 0) {
	value = value.Resolve();
	if (value.IsEmpty() || depth > 8) return std::nullopt;
	if (value.IsImmediate()) return value.GetType() == Type::U1 ? std::optional(value.U1()) : std::nullopt;
	for (const auto& [condition, holds]: facts) {
		if (condition == value) return holds;
	}
	const auto* inst = value.TryInstruction();
	switch (inst->GetOpcode()) {
		case ValueOpcode::LogicalNot: {
			const auto arg = Decided(inst->Arg(0), facts, depth + 1);
			return arg ? std::optional(!*arg) : std::nullopt;
		}
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd: {
			const bool any = inst->GetOpcode() == ValueOpcode::LogicalOr;
			const auto a = Decided(inst->Arg(0), facts, depth + 1), b = Decided(inst->Arg(1), facts, depth + 1);
			if (a == any || b == any) return any;
			if (a && b) return !any;
			return std::nullopt;
		}
		case ValueOpcode::ConditionRef: return Decided(inst->Arg(0), facts, depth + 1);
		default: return std::nullopt;
	}
}

bool TakesEdge(const Block* from, const Block* to, BranchFacts& facts) {
	const auto& terminator = from->terminator;
	if (terminator.kind != CFG::TerminatorKind::ConditionalBranch || terminator.true_block == terminator.false_block)
		return true;
	const bool needed  = to == terminator.true_block;
	const auto decided = Decided(from->condition, facts);
	if (decided) return *decided == needed;
	const auto [condition, negated] = Literal(from->condition);
	facts.emplace_back(condition, needed != negated);
	return true;
}

bool BlockRuns(const Block* block, BranchFacts& facts) {
	for (uint32_t depth = 0; depth < 64; ++depth) {
		const auto preds = block->ImmPredecessors();
		if (preds.size() != 1) return true;
		if (!TakesEdge(preds[0], block, facts)) return false;
		block = preds[0];
	}
	return true;
}

bool ReachingValues(Value value, const BranchFacts& facts, std::vector<Value>& found, uint32_t depth = 0) {
	value           = value.Resolve();
	const auto* phi = Producer(value, ValueOpcode::Phi);
	if (phi == nullptr) {
		if (std::ranges::find(found, value) == found.end()) found.push_back(value);
		return true;
	}
	const auto* block = phi->Parent();
	if (depth > 32 || block->terminator.loop_header) return false;
	for (size_t arg = 0; arg < phi->NumArgs(); ++arg) {
		const auto* from = phi->PhiBlock(arg);
		BranchFacts edge;
		bool        holds = true;
		for (const auto& [condition, value_there]: facts) {
			const auto* fact = Producer(condition, ValueOpcode::Phi);
			if (fact == nullptr || fact->Parent() != block) {
				edge.emplace_back(condition, value_there);
				continue;
			}
			const auto [incoming, negated] = Literal(IncomingValue(*fact, from));
			const auto decided             = Decided(incoming, {});
			holds                          = holds && (!decided || *decided == (value_there != negated));
			edge.emplace_back(incoming, value_there != negated);
		}
		if (!holds || !TakesEdge(from, block, edge) || !BlockRuns(from, edge)) continue;
		if (!ReachingValues(phi->Arg(arg), edge, found, depth + 1)) return false;
	}
	return true;
}

Value ReachingValue(Value value, const Inst& inst) {
	BranchFacts        facts;
	std::vector<Value> found;
	if (!BlockRuns(inst.Parent(), facts) || !ReachingValues(value, facts, found) || found.size() != 1) return {};
	return found[0];
}

bool DeriveHeaderSource(const Program& program, LaneMaskWalk& w) {
	auto&       source = w.source;
	const auto  value  = Canonical(w.tlas[0]);
	const auto* read   = Producer(value, ValueOpcode::ReadConstBuffer);
	if (read != nullptr) {
		const auto* buffer = Producer(read->Arg(0), ValueOpcode::GetBufferResource);
		const auto  base   = buffer != nullptr ? LoadOf(program, buffer->Arg(0)) : std::nullopt;
		const auto  user   = base ? UserRegister(base->low) : std::nullopt;
		if (!user || UserRegister(base->high) != *user + 1u) return false;
		source.user_sgpr     = *user;
		source.header_vsharp = base->offset;
		source.header_offset = MemoryOf(program, *read).offset;
		return true;
	}
	if (const auto low = UserRegister(value); low) {
		const auto* cut  = Producer(w.tlas[1], ValueOpcode::BitwiseAnd32);
		auto        high = UserRegister(w.tlas[1]);
		for (uint32_t arg = 0; cut != nullptr && arg < 2; ++arg) {
			if (Immediate(cut->Arg(1 - arg)) == 0xffffu) high = UserRegister(cut->Arg(arg));
		}
		if (high != *low + 1u) return false;
		source.user_sgpr     = *low;
		source.header_offset = ShaderInfo::UserDataRayQueryHeader;
		return true;
	}
	auto load = LoadOf(program, value);
	auto user = load ? UserRegister(load->low) : std::nullopt;
	if (load && !user) {
		const auto& at = *value.Resolve().TryInstruction();
		load->low      = ReachingValue(load->low, at);
		load->high     = ReachingValue(load->high, at);
		user           = !load->low.IsEmpty() ? UserRegister(load->low) : std::nullopt;
	}
	if (!user || load->high.IsEmpty() || UserRegister(load->high) != *user + 1u) return false;
	source.user_sgpr     = *user;
	source.header_offset = load->offset;
	return true;
}

bool NestedBlasWalk(const Program& program, Inst& bvh, Inst& tlas, BlasWalk& b) {
	Traversal  nested;
	const auto walks = std::array {&tlas};
	if (!MatchNested(program, bvh, walks, nested, false) || nested.walk != &tlas) return false;
	b = {.preheader = nested.preheader,
	     .header    = nested.header,
	     .latch     = nested.latch,
	     .node      = nested.state[0],
	     .word      = nested.state[1],
	     .load      = nested.record,
	     .vote      = nested.vote,
	     .lanes     = nested.lanes};
	return true;
}

bool MatchLaneMaskWalk(Program& program, Inst& bvh, LaneMaskWalk& w) {
	const auto* descriptor = Producer(bvh.Arg(0), ValueOpcode::CompositeConstructU32x4);
	const auto* address    = Producer(bvh.Arg(1), ValueOpcode::MakeImageAddress);
	const auto* tlas = descriptor != nullptr ? DescriptorAddress(descriptor->Arg(0)) : nullptr;
	if (address == nullptr || tlas == nullptr ||
	    tlas->GetOpcode() != ValueOpcode::CompositeConstructU64)
		return false;
	w.tlas   = {tlas->Arg(0).Resolve(), tlas->Arg(1).Resolve()};
	w.bvh    = &bvh;
	w.header = EnclosingLoop(program, bvh.Parent());
	if (!LoopShape(w.header, w.preheader, w.latch, w.merge)) return false;
	const auto entry = [&](const Inst& phi) { return IncomingValue(phi, w.preheader); };
	const auto back  = [&](const Inst& phi) { return IncomingValue(phi, w.latch); };
	w.node          = LoopPhi(Canonical(address->Arg(0)), w.header);
	const auto root = w.node != nullptr ? Immediate(entry(*w.node)) : std::nullopt;
	if (!root) return false;
	const auto popped = [](Value value) -> const Inst* {
		const auto* phi = Producer(value, ValueOpcode::Phi);
		if (phi == nullptr) return nullptr;
		for (size_t arg = 0; arg < phi->NumArgs(); ++arg) {
			if (Producer(phi->Arg(arg), ValueOpcode::ReadLane) == nullptr) return nullptr;
		}
		return phi;
	};
	const auto* pop = popped(back(*w.node));
	if (pop == nullptr) return false;
	for (auto& inst: *w.header) {
		const auto* other =
		    inst.GetOpcode() == ValueOpcode::Phi && &inst != w.node ? popped(back(inst)) : nullptr;
		if (other == nullptr || other->Parent() != pop->Parent()) continue;
		if (w.pair_mask != nullptr) return false;
		w.pair_mask = &inst;
	}
	if (w.pair_mask == nullptr) return false;
	const auto  initial = entry(*w.pair_mask);
	const auto* both    = Producer(initial, ValueOpcode::BitwiseAnd32);
	const auto* minus   = both ? Producer(both->Arg(1), ValueOpcode::BitwiseNot32) : nullptr;
	const auto* ballot  = BallotOf(minus != nullptr ? both->Arg(0) : initial);
	const auto* without = minus != nullptr ? BallotOf(minus->Arg(0)) : nullptr;
	if (ballot == nullptr || (minus != nullptr && without == nullptr)) return false;
	w.lanes         = ballot->Arg(0).Resolve();
	w.lanes_without = without != nullptr ? without->Arg(0).Resolve() : Value {};

	const auto first = BlockIndex(program, w.header), last = BlockIndex(program, w.latch);
	for (auto at = first; at <= last; ++at) {
		auto* block = program.blocks[at];
		if (std::ranges::find(block->ImmSuccessors(), w.merge) == block->ImmSuccessors().end())
			continue;
		if (w.exit != nullptr) return false;
		w.exit = block;
	}
	w.exit_ref =
	    w.exit != nullptr ? Producer(w.exit->condition, ValueOpcode::ConditionRef) : nullptr;
	if (w.exit_ref == nullptr || BlockIndex(program, w.exit) >= BlockIndex(program, pop->Parent()))
		return false;
	for (uint32_t arg = 1; arg < 8; ++arg) {
		w.ray[(arg + 5) % 7] = address->Arg(arg);
	}
	for (auto at = first; at <= last; ++at) {
		for (auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != ValueOpcode::BvhIntersect || &inst == &bvh) continue;
			BlasWalk blas;
			if (!NestedBlasWalk(program, inst, bvh, blas)) return false;
			w.blases.push_back(std::move(blas));
		}
	}
	if (w.blases.empty() || !DeriveLaneLayout(program, w) || !DeriveHeaderSource(program, w))
		return false;
	w.source.root   = *root;
	w.source.layout = w.layout.Pack();
	uint32_t begin = UINT32_MAX;
	for (auto at = first; at <= last; ++at) {
		const auto* block = program.blocks[at];
		if (block->end_pc > block->start_pc) begin = std::min(begin, block->start_pc);
	}
	if (begin == UINT32_MAX || w.merge->start_pc <= begin) return false;
	w.source.reach        = {{begin, w.merge->start_pc}};
	w.source.source_valid = true;
	return true;
}

void RewriteBlasWalks(Program& program, std::vector<BlasWalk>& blases, Value any, uint32_t lane_leaf) {
	using Op = ValueOpcode;
	for (auto& b: blases) {
		const Inserter pre {b.preheader, b.preheader->end()};
		const auto     lanes = IncomingValue(*b.lanes.front(), b.preheader);
		const auto     index =
		    pre(Op::ShiftRightLogical32, {pre(Op::RayQuerySelectedLeaf, {}, lane_leaf), Value(3u)});
		const auto& guest = *Producer(b.load->Arg(0), ValueOpcode::GetBufferResource);
		const auto  buffer =
		    pre(Op::GetBufferResource, {guest.Arg(0), guest.Arg(1), guest.Arg(2), guest.Arg(3)},
		        guest.Flags<uint64_t>());
		auto memory = b.load->Flags<MemoryFlags>();
		program.memory_info.push_back(program.memory_info[memory.index]);
		memory.index      = static_cast<uint32_t>(program.memory_info.size() - 1);
		const auto loaded = pre(Op::LoadBufferU32x2, {buffer, index, b.load->Arg(2), b.load->Arg(3),
		                                              pre(Op::LogicalAnd, {lanes, any})});
		loaded.Instruction()->SetFlags(memory);
		for (auto [phi, word]: {std::pair {b.node, 0u}, std::pair {b.word, 1u}}) {
			const auto arg = Incoming(*phi, b.preheader);
			phi->SetArg(
			    arg, pre(Op::SelectU32, {any, pre(Op::CompositeExtractU32x2, {loaded, Value(word)}),
			                             phi->Arg(arg)}));
		}
		const Inserter vote {b.vote->Parent(), Position(*b.vote)};
		const auto     alive = vote(Op::SelectU1, {any, Value(false), b.vote->Arg(0)});
		b.vote->SetArg(0, alive);
		for (auto* lane: b.lanes) {
			lane->SetArg(Incoming(*lane, b.latch), alive);
		}
	}
}

void RewriteLaneMaskWalk(Program& program, LaneMaskWalk& w, uint32_t site) {
	using Op         = ValueOpcode;
	const auto flags = [site](auto variant) {
		return RayQueryFlags(site, static_cast<uint32_t>(variant));
	};
	const auto lane_leaf = flags(RayQueryKeyLaneLeaf);

	const Inserter entry {w.preheader, w.preheader->end()};
	if (!w.lanes_without.IsEmpty()) {
		w.lanes = entry(Op::LogicalAnd, {w.lanes, entry(Op::LogicalNot, {w.lanes_without})});
	}
	std::array<Value, 7> ray {};
	for (uint32_t k = 0; k < 7; ++k)
		ray[k] = *Hoist(program, w.ray[k], w.header, entry);
	const auto native =
	    entry(Op::RayQueryInit,
	          {ray[0], ray[1], ray[2], ray[3], ray[4], ray[5], ray[6], RayMask(w),
	           w.tlas[0], w.tlas[1], Value(0u), w.lanes, Value(true), Value(false), Value(true)},
	          flags(RayQueryInitBothFacings | RayQueryInitAllCandidates));
	const auto ballot = entry(Op::Ballot, {native});
	const auto any =
	    entry(Op::INotEqual32,
	          {entry(Op::BitwiseOr32, {entry(Op::CompositeExtractU32x4, {ballot, Value(0u)}),
	                                   entry(Op::CompositeExtractU32x4, {ballot, Value(1u)})}),
	           Value(0u)});
	const auto first = Incoming(*w.pair_mask, w.preheader);
	w.pair_mask->SetArg(first, entry(Op::SelectU32, {any, Value(0u), w.pair_mask->Arg(first)}));

	const Inserter at {w.exit, Position(*w.exit_ref)};
	at(Op::RayQueryMark, {}, flags(RayQueryMarkKind::Resolve));
	const auto next  = at(Op::RayQueryNext, {}, lane_leaf);
	const auto match = at(Op::RayQueryMatch, {}, lane_leaf);
	const auto has =
	    at(Op::INotEqual32, {at(Op::CompositeExtractU32x4, {next, Value(2u)}), Value(0u)});
	const auto instance = at(Op::CompositeExtractU32x4, {next, Value(0u)});
	const auto mask     = at(Op::CompositeExtractU32x4, {at(Op::Ballot, {match}), Value(0u)});
	const auto done     = at(Op::LogicalAnd, {w.lanes, at(Op::LogicalNot, {has})});
	w.exit_ref->SetArg(0, at(Op::SelectU1, {any, done, w.exit_ref->Arg(0)}));
	const Inserter back {w.latch, AfterPhis(*w.latch)};
	for (auto [phi, value]: {std::pair {w.node, instance}, std::pair {w.pair_mask, mask}}) {
		const auto index = Incoming(*phi, w.latch);
		phi->SetArg(index, back(Op::SelectU32, {any, value, phi->Arg(index)}));
	}
	const Inserter leave {w.merge, AfterPhis(*w.merge)};
	leave(Op::RayQueryMark, {}, flags(RayQueryMarkKind::Exit));

	RewriteBlasWalks(program, w.blases, any, lane_leaf);
}

struct LaneAlpha {
	bool                           valid = false;
	Value                          lod;
	Value                          spread;
	Value                          reference;
	std::array<Value, 4>           material {};
	std::array<Value, 4>           buffers {};
	uint32_t                       lod_offset = 0;
	RayTracingAlphaFields fields;
	const Inst*                    flag = nullptr;
	Value factor;
	Value level;
	bool scalar        = false;
	bool factor_always = false;
};

struct PacketWalk {
	Block*              preheader = nullptr;
	Block*              header    = nullptr;
	Block*              latch     = nullptr;
	Block*              merge     = nullptr;
	std::vector<Block*> path;
	LaneMaskWalk        shape;
	Value               lanes;
	Value               t_max;
	std::array<Value, 4>                              entries {};
	std::array<Value, 4>                              buffers {};
	std::vector<std::pair<Inst*, RayQueryResultKind>> results;
	Inst* outer_node = nullptr;
	Inst* packet     = nullptr;
	bool  both_facings = true;
	Value cull {true};
	LaneAlpha alpha;
	bool  any_hit = false;
	Value alpha_off;
	Value flip;
	bool walked = false;
	bool                                 leaf = false;
	std::vector<std::pair<Block*, bool>> list_exits;
	Value                                leaf_t;
	std::vector<Inst*> lane_phis;
};

constexpr uint64_t EntryTemplateHash = 0x777c0050ecd1ca9eull;

Value PathEntry(const PacketWalk& w, Value value) {
	for (uint32_t depth = 0; depth < 64; ++depth) {
		value            = value.Resolve();
		const auto* phi  = Producer(value, ValueOpcode::Phi);
		const auto  at   = phi != nullptr ? std::ranges::find(w.path, phi->Parent()) : w.path.end();
		if (at == w.path.end()) break;
		const auto* from = at == w.path.begin() ? w.preheader : *(at - 1);
		const auto  arg  = Incoming(*phi, from);
		if (arg == phi->NumArgs()) return {};
		value = phi->Arg(arg);
	}
	return value;
}

Value EntryLanes(const PacketWalk& w, Value lanes, uint32_t depth = 0) {
	lanes = PathEntry(w, lanes);
	if (depth > 8) return lanes;
	const auto all = [](Value value) {
		return !value.IsEmpty() && value.IsImmediate() && value.GetType() == Type::U1 && value.U1();
	};
	if (const auto* select = Producer(lanes, ValueOpcode::SelectU1); select != nullptr) {
		const auto first = EntryLanes(w, select->Arg(0), depth + 1);
		if (!first.IsEmpty() && first.IsImmediate() && first.GetType() == Type::U1)
			return EntryLanes(w, select->Arg(first.U1() ? 1 : 2), depth + 1);
	}
	if (const auto* both = Producer(lanes, ValueOpcode::LogicalAnd); both != nullptr) {
		const auto a = EntryLanes(w, both->Arg(0), depth + 1), b = EntryLanes(w, both->Arg(1), depth + 1);
		if (!a.IsEmpty() && (a == b || all(b))) return a;
		if (all(a)) return b;
	}
	return lanes;
}

bool WalkReach(const Program& program, const Block* header, const Block* latch, uint32_t end,
               NativeTraversals& source) {
	uint32_t begin = UINT32_MAX;
	for (auto at = BlockIndex(program, header); at <= BlockIndex(program, latch); ++at) {
		const auto* block = program.blocks[at];
		if (block->end_pc > block->start_pc) begin = std::min(begin, block->start_pc);
	}
	if (begin == UINT32_MAX || end <= begin) return false;
	source.reach        = {{begin, end}};
	source.source_valid = true;
	return true;
}

Value WaveRoot(const Program& program, const Inst* address, const Block* header, const Block* pre,
               const Block* latch, Inst** phi = nullptr) {
	std::vector<Value> nodes {Canonical(address->Arg(0))};
	const Inst*        node = nullptr;
	for (size_t at = 0; at < nodes.size() && at < 16 && node == nullptr; ++at) {
		const auto* phi = Producer(nodes[at], ValueOpcode::Phi);
		if (phi == nullptr || !InLoop(program, phi->Parent(), header, latch)) continue;
		if (phi->Parent() == header) {
			node = phi;
			break;
		}
		for (size_t arg = 0; arg < phi->NumArgs(); ++arg)
			nodes.push_back(Canonical(phi->Arg(arg)));
	}
	Value root;
	if (node == nullptr) {
		const auto* top   = Producer(nodes[0], ValueOpcode::ReadLane);
		const auto* stack = top != nullptr ? LoopPhi(top->Arg(0), header) : nullptr;
		const auto* push  = stack != nullptr ? Producer(IncomingValue(*stack, pre), ValueOpcode::WriteLane) : nullptr;
		const auto  low6  = [](Value value) {
			const auto pair = Operands(value, ValueOpcode::BitwiseAnd32, [](Value side) { return Immediate(side) == 63u; });
			return pair ? Canonical(pair->second) : Value {};
		};
		const auto  index = top != nullptr ? low6(top->Arg(1)) : Value {};
		const auto  below = !index.IsEmpty() ? Operands(index, ValueOpcode::IAdd32, [](Value side) { return Immediate(side) == ~0u; })
		                                     : std::nullopt;
		const auto* sp    = below ? LoopPhi(below->second, header) : nullptr;
		const auto  above = sp != nullptr ? Operands(IncomingValue(*sp, pre), ValueOpcode::IAdd32,
		                                             [](Value side) { return Immediate(side) == 1u; })
		                                  : std::nullopt;
		if (push == nullptr || !above || low6(push->Arg(2)).IsEmpty() || low6(push->Arg(2)) != Canonical(above->second))
			return {};
		root = push->Arg(1).Resolve();
	}
	if (phi != nullptr) *phi = const_cast<Inst*>(node);
	return node != nullptr ? IncomingValue(*node, pre) : root;
}

bool MatchWaveBlas(const Program& program, std::span<Inst* const> tests, uint32_t box, LaneMaskWalk& s) {
	const auto* a = Producer(tests[0]->Arg(0), ValueOpcode::CompositeConstructU32x4);
	for (const auto* test: tests) {
		const auto* b = Producer(test->Arg(0), ValueOpcode::CompositeConstructU32x4);
		if (a == nullptr || b == nullptr) return false;
		for (size_t k = 0; k < 4; ++k) {
			if (a->Arg(k).Resolve() != b->Arg(k).Resolve()) return false;
		}
	}
	const auto* address = Producer(tests[box]->Arg(1), ValueOpcode::MakeImageAddress);
	Block*      header  = EnclosingLoop(program, tests[box]->Parent());
	Block *     pre = nullptr, *latch = nullptr, *merge = nullptr;
	std::vector<Inst*> triangles;
	for (size_t k = 0; k < tests.size(); ++k) {
		if (k != box) triangles.push_back(tests[k]);
	}
	if (address == nullptr || header == s.header || !LoopShape(header, pre, latch, merge) ||
	    !InLoop(program, header, s.header, s.latch) || triangles.empty() || triangles.size() > 2 ||
	    std::ranges::any_of(triangles, [&](const Inst* tri) { return !InLoop(program, tri->Parent(), header, latch); }) ||
	    (triangles.size() == 2 &&
	     EnclosingLoop(program, triangles[0]->Parent()) == EnclosingLoop(program, triangles[1]->Parent())))
		return false;
	for (;;) {
		const auto entries = pre->ImmPredecessors();
		auto*      outer   = entries.size() == 1 ? entries[0] : nullptr;
		Block *    outer_pre = nullptr, *outer_latch = nullptr, *outer_merge = nullptr;
		if (outer == nullptr || outer == s.header || !outer->terminator.loop_header ||
		    !LoopShape(outer, outer_pre, outer_latch, outer_merge) || !InLoop(program, header, outer, outer_latch) ||
		    !InLoop(program, outer, s.header, s.latch))
			break;
		header = outer;
		pre    = outer_pre;
		latch  = outer_latch;
	}
	const auto root = WaveRoot(program, address, header, pre, latch, &s.wave_node);
	if (root.IsEmpty()) return false;
	s.wave_header    = header;
	s.wave_latch     = latch;
	s.wave_root      = root;
	s.wave_box       = tests[box];
	s.wave_triangles = triangles[0];
	s.wave_copy      = triangles.size() == 2 ? triangles[1] : nullptr;
	return true;
}

bool MatchPacketWalk(Program& program, Inst& bvh, PacketWalk& w) {
	const auto* descriptor = Producer(bvh.Arg(0), ValueOpcode::CompositeConstructU32x4);
	const auto* address    = Producer(bvh.Arg(1), ValueOpcode::MakeImageAddress);
	const auto* tlas = descriptor != nullptr ? DescriptorAddress(descriptor->Arg(0)) : nullptr;
	if (address == nullptr || tlas == nullptr ||
	    tlas->GetOpcode() != ValueOpcode::CompositeConstructU64)
		return false;
	auto& s = w.shape;
	s.tlas  = {tlas->Arg(0).Resolve(), tlas->Arg(1).Resolve()};
	s.bvh   = &bvh;
	s.header = EnclosingLoop(program, bvh.Parent());
	if (!LoopShape(s.header, s.preheader, s.latch, s.merge)) return false;
	w.header    = s.header;
	w.preheader = s.preheader;
	w.latch     = s.latch;
	w.merge     = s.merge;
	for (;;) {
		const auto entries = w.preheader->ImmPredecessors();
		auto*      outer   = entries.size() == 1 ? entries[0] : nullptr;
		Block *    pre = nullptr, *latch = nullptr, *merge = nullptr;
		if (outer == nullptr || !outer->terminator.loop_header ||
		    !LoopShape(outer, pre, latch, merge) || !InLoop(program, w.header, outer, latch))
			break;
		w.header    = outer;
		w.preheader = pre;
		w.latch     = latch;
		w.merge     = merge;
	}
	const auto first = BlockIndex(program, w.header), last = BlockIndex(program, w.latch);
	const auto in_walk = [&](const Block* block) {
		const auto at = BlockIndex(program, block);
		return at >= first && at <= last;
	};
	const auto intersects = [](const Block* block) {
		return std::ranges::any_of(*block, [](const Inst& inst) {
			return inst.GetOpcode() == ValueOpcode::BvhIntersect;
		});
	};
	const auto shortest = [&](Block* from, auto&& goal) {
		std::unordered_map<const Block*, Block*> parent {{from, nullptr}};
		std::vector<Block*>                      queue {from}, way;
		Block*                                   end = nullptr;
		for (size_t at = 0; at < queue.size() && end == nullptr; ++at) {
			auto* block = queue[at];
			for (auto* next: block->ImmSuccessors()) {
				if (goal(next)) {
					end = block;
					break;
				}
				if (!in_walk(next) || BlockIndex(program, next) <= BlockIndex(program, block) ||
				    intersects(next) || parent.contains(next))
					continue;
				parent.emplace(next, block);
				queue.push_back(next);
			}
		}
		for (auto* block = end; block != nullptr; block = parent.at(block))
			way.insert(way.begin(), block);
		return way;
	};
	const auto preds = bvh.Parent()->ImmPredecessors();
	if (preds.size() != 1) return false;
	w.path = shortest(w.header, [&](const Block* next) { return next == preds[0]; });
	if (w.path.empty()) return false;
	w.path.push_back(preds[0]);
	const auto rest = shortest(preds[0], [&](const Block* next) { return next == w.merge; });
	if (rest.empty()) return false;
	w.path.insert(w.path.end(), rest.begin() + 1, rest.end());
	for (size_t at = 0; at < w.path.size(); ++at) {
		const auto* block = w.path[at];
		const auto  kind  = block->terminator.kind;
		if (kind != CFG::TerminatorKind::Branch &&
		    (kind != CFG::TerminatorKind::ConditionalBranch || BranchCondition(*block).first == nullptr))
			return false;
	}
	for (uint32_t arg = 1; arg < 8; ++arg) {
		s.ray[(arg + 5) % 7] = address->Arg(arg);
	}
	const auto* origin = Producer(s.ray[0], ValueOpcode::SelectU32);
	for (uint32_t arg = 8; arg < 11 && origin == nullptr; ++arg) {
		const auto* inverse = Producer(address->Arg(arg), ValueOpcode::SelectU32);
		if (inverse == nullptr || (arg > 8 && inverse->Arg(0).Resolve() != w.lanes)) return false;
		w.lanes = inverse->Arg(0).Resolve();
		if (arg == 10) origin = inverse;
	}
	w.lanes = origin->Arg(0).Resolve();
	w.t_max = PathEntry(w, s.ray[6]);
	if (!DefinedBefore(program, w.lanes, w.header)) {
		for (auto& component: std::span(s.ray).first(6)) {
			const auto* copy = Producer(component, ValueOpcode::SelectU32);
			if (!DefinedBefore(program, component, w.header) && copy != nullptr && copy->Arg(0).Resolve() == w.lanes)
				component = copy->Arg(1).Resolve();
		}
		w.lanes = EntryLanes(w, w.lanes);
	}
	if (!DefinedBefore(program, w.lanes, w.header) || !DefinedBefore(program, w.t_max, w.header) ||
	    std::ranges::any_of(std::span(s.ray).first(6),
	                        [&](Value v) { return !DefinedBefore(program, v, w.header); }))
		return false;
	std::vector<Value> nodes {Canonical(address->Arg(0))};
	for (size_t at = 0; at < nodes.size() && at < 8 && s.node == nullptr; ++at) {
		auto* phi = Producer(nodes[at], ValueOpcode::Phi);
		if (phi == nullptr) continue;
		if (phi->Parent() == s.header) {
			s.node = phi;
			break;
		}
		for (size_t arg = 0; arg < phi->NumArgs(); ++arg)
			nodes.push_back(phi->Arg(arg));
	}
	const auto root = s.node != nullptr ? Immediate(PathEntry(w, Value(s.node))) : std::nullopt;
	if (!root) return false;
	std::vector<Inst*> others;
	for (auto at = first; at <= last; ++at) {
		for (auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() == ValueOpcode::BvhIntersect && &inst != &bvh) others.push_back(&inst);
		}
	}
	s.mask_optional = true;
	if (others.size() == 1) {
		BlasWalk blas;
		if (!NestedBlasWalk(program, *others[0], bvh, blas)) return false;
		s.blases.push_back(std::move(blas));
		if (!DeriveLaneLayout(program, s)) return false;
	} else if (others.size() == 2) {
		const auto tlas  = std::array {&bvh};
		bool       found = false;
		for (uint32_t box = 0; box < 2 && !found; ++box) {
			Walk shared;
			if (!MatchWalk(program, *others[1 - box], tlas, shared, false) || shared.walk != &bvh) continue;
			const auto* address = Producer(others[box]->Arg(1), ValueOpcode::MakeImageAddress);
			const auto  root    = address != nullptr ? WaveRoot(program, address, shared.blas.header,
			                                                    shared.blas.preheader, shared.blas.latch)
			                                         : Value {};
			if (root.IsEmpty() || EnclosingLoop(program, others[box]->Parent()) != shared.blas.header) break;
			s.wave_header    = shared.blas.header;
			s.wave_latch     = shared.blas.latch;
			s.wave_root      = root;
			s.wave_box       = others[box];
			s.wave_triangles = others[1 - box];
			found = w.walked = DeriveLaneLayout(program, s);
		}
		if (!found && !(MatchWaveBlas(program, others, 0, s) && DeriveLaneLayout(program, s)) &&
		    !(MatchWaveBlas(program, others, 1, s) && DeriveLaneLayout(program, s))) {
			return false;
		}
	} else if (others.size() == 3) {
		bool found = false;
		for (uint32_t box = 0; box < 3 && !found; ++box) {
			found = MatchWaveBlas(program, others, box, s) && DeriveLaneLayout(program, s);
		}
		if (!found) return false;
	} else {
		return false;
	}
	if (!DeriveHeaderSource(program, s)) return false;
	s.source.root   = *root;
	s.source.layout = s.layout.Pack();
	return WalkReach(program, w.header, w.latch, w.merge->start_pc, s.source);
}

template <typename KindOf>
std::optional<std::array<bool, 8>> HitResults(const Program& program, PacketWalk& w, KindOf&& kind_of,
                                              Value hit = {}, bool negated = false, bool select_kinds = false,
                                              std::span<const Value> others = {}) {
	const auto first = BlockIndex(program, w.header), last = BlockIndex(program, w.latch);
	const auto in_walk = [&](const Block* block) {
		const auto at = BlockIndex(program, block);
		return at >= first && at <= last;
	};
	const bool          strict = hit.IsEmpty();
	std::array<bool, 8> found {};
	const auto hit_mask = select_kinds ? LaneMaskOf(hit.Resolve().TryInstruction()) : Value {};
	const auto same     = [&](Value condition) {
		condition = condition.Resolve();
		return condition == hit.Resolve() || std::ranges::find(others, condition) != others.end() ||
		       (!hit_mask.IsEmpty() && LaneMaskOf(condition.TryInstruction()) == hit_mask);
	};
	for (auto at = first; at <= last; ++at) {
		for (auto& inst: *program.blocks[at]) {
			if (std::ranges::none_of(inst.Uses(), [&](const Use& use) { return !in_walk(use.user->Parent()); }))
				continue;
			std::unordered_set<const Inst*>          seen[2];
			std::vector<std::pair<const Inst*, bool>> pending {{&inst, strict}};
			std::optional<RayQueryResultKind>         kind;
			bool                                      mixed = false;
			while (!pending.empty() && !mixed) {
				const auto [value, accepted] = pending.back();
				pending.pop_back();
				if (!seen[accepted ? 1 : 0].insert(value).second) continue;
				const auto op = value->GetOpcode();
				const auto phi_kind = accepted && (op == ValueOpcode::Phi || (select_kinds && op == ValueOpcode::SelectU32))
				                          ? kind_of(value)
				                          : std::nullopt;
				if (phi_kind) {
					mixed = kind && *kind != *phi_kind;
					kind  = phi_kind;
					continue;
				}
				if (op == ValueOpcode::Phi || op == ValueOpcode::SelectU32) {
					for (size_t arg = op == ValueOpcode::Phi ? 0 : 1; arg < (op == ValueOpcode::Phi ? value->NumArgs() : 3); ++arg) {
						const auto* source = value->Arg(arg).Resolve().TryInstruction();
						const bool  write  = op == ValueOpcode::SelectU32 && arg == (negated ? 2u : 1u) && same(value->Arg(0));
						if (source != nullptr && in_walk(source->Parent())) pending.emplace_back(source, accepted || write);
					}
					continue;
				}
				if (!accepted) continue;
				const auto source = kind_of(value);
				if (!strict && !source) continue;
				mixed = !source || (kind && *kind != *source);
				kind  = source;
			}
			if (mixed || !kind) continue;
			if (std::ranges::any_of(inst.Uses(), [&](const Use& use) {
				    return use.user->Parent() == w.merge && use.user->GetOpcode() == ValueOpcode::Phi;
			    }))
				return std::nullopt;
			w.results.emplace_back(&inst, *kind);
			found[static_cast<uint32_t>(*kind)] = true;
		}
	}
	return found;
}

bool MatchEntryWalk(Program& program, const NativeTraversalOptions& options, PacketWalk& w) {
	auto& s = w.shape;
	if (options.decoded == nullptr || !s.masked || s.blases.empty()) return false;
	const auto& blas = s.blases.front();
	uint32_t begin = UINT32_MAX, end = 0;
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		const auto* block = program.blocks[at];
		if (block->end_pc > block->start_pc) {
			begin = std::min(begin, block->start_pc);
			end   = std::max(end, block->end_pc);
		}
	}
	uint64_t hash = 1469598103934665603ull;
	for (const auto& inst: options.decoded->instructions) {
		if (inst.pc < begin || inst.pc >= end) continue;
		for (uint32_t k = 0; k < inst.word_count && k < Decoder::MaxInstructionRawWords; ++k)
			hash = (hash ^ inst.raw[k]) * 1099511628211ull;
	}
	if (hash != EntryTemplateHash) return false;
	std::vector<const Inst*> tables;
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		for (const auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != ValueOpcode::ReadConstBuffer) continue;
			const auto* table = Producer(inst.Arg(0), ValueOpcode::GetBufferResource);
			if (table == nullptr) return false;
			const auto same = [&](const Inst* other) {
				for (size_t k = 0; k < 4; ++k) {
					if (other->Arg(k).Resolve() != table->Arg(k).Resolve()) return false;
				}
				return true;
			};
			if (std::ranges::none_of(tables, same)) tables.push_back(table);
		}
	}
	if (tables.size() < 2) return false;
	for (size_t k = 0; k < 4; ++k) {
		w.entries[k] = tables[0]->Arg(k).Resolve();
		w.buffers[k] = tables[1]->Arg(k).Resolve();
		if (!DefinedBefore(program, w.entries[k], w.header) ||
		    !DefinedBefore(program, w.buffers[k], w.header))
			return false;
	}
	s.layout.flags_dword = 15;
	s.layout.flags_shift = 0;
	s.source.layout      = s.layout.Pack();
	const auto record = LoadOf(program, Canonical(IncomingValue(*blas.node, blas.preheader)));
	if (!record) return false;
	const auto kind_of = [&](const Inst* value) -> std::optional<RayQueryResultKind> {
		if (value == record->low.TryInstruction()) return RayQueryResultKind::InstanceLo;
		if (value == record->high.TryInstruction()) return RayQueryResultKind::InstanceHi;
		if (value->GetOpcode() == ValueOpcode::CompositeExtractU32x2 && Immediate(value->Arg(1)) == 1u &&
		    value->Arg(0).Resolve().TryInstruction() == blas.load)
			return RayQueryResultKind::Word;
		if (value->GetOpcode() == ValueOpcode::IAdd32) {
			for (uint32_t side = 0; side < 2; ++side) {
				if (Immediate(value->Arg(side)) == 1u && value->Arg(1 - side).Resolve() == blas.load->Arg(1).Resolve())
					return RayQueryResultKind::Node;
			}
		}
		return std::nullopt;
	};
	const auto found = HitResults(program, w, kind_of);
	return found && (*found)[static_cast<uint32_t>(RayQueryResultKind::Word)] &&
	       (*found)[static_cast<uint32_t>(RayQueryResultKind::Node)] &&
	       (*found)[static_cast<uint32_t>(RayQueryResultKind::InstanceLo)];
}

uint32_t Components(const Program& program, const PacketWalk& w, Value value, const Inst* bvh) {
	uint32_t                        components = 0;
	std::unordered_set<const Inst*> seen;
	std::vector<Value>              pending {value};
	while (!pending.empty() && seen.size() < 64) {
		auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || !InLoop(program, inst->Parent(), w.header, w.latch) || !seen.insert(inst).second) continue;
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::BitCastF32U32 || op == ValueOpcode::BitCastU32F32) {
			pending.push_back(inst->Arg(0));
		} else if (op == ValueOpcode::SelectU32 || op == ValueOpcode::Phi) {
			for (size_t arg = op == ValueOpcode::Phi ? 0 : 1; arg < inst->NumArgs(); ++arg)
				pending.push_back(inst->Arg(arg));
		} else if (op == ValueOpcode::CompositeExtractU32x4 && inst->Arg(0).Resolve().TryInstruction() == bvh) {
			components |= 1u << Immediate(inst->Arg(1)).value_or(0u);
		}
	}
	return components;
}

bool Reciprocal(const Program& program, const PacketWalk& w, Value value, const Inst* bvh) {
	std::unordered_set<const Inst*> seen;
	std::vector<Value>              pending {value};
	while (!pending.empty() && seen.size() < 64) {
		auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || !InLoop(program, inst->Parent(), w.header, w.latch) || !seen.insert(inst).second) continue;
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::BitCastF32U32 || op == ValueOpcode::BitCastU32F32) {
			pending.push_back(inst->Arg(0));
		} else if (op == ValueOpcode::SelectU32 || op == ValueOpcode::Phi) {
			for (size_t arg = op == ValueOpcode::Phi ? 0 : 1; arg < inst->NumArgs(); ++arg)
				pending.push_back(inst->Arg(arg));
		} else if (op == ValueOpcode::FPRecip32 && (Components(program, w, inst->Arg(0), bvh) & 2u) != 0) {
			return true;
		}
	}
	return false;
}

std::optional<RayQueryResultKind> WrittenHitDistance(Value value, const Inst* bvh) {
	const auto* mul = Producer(Canonical(value), ValueOpcode::FPMul32);
	if (mul == nullptr) return std::nullopt;
	const auto component = [&](Value operand) -> std::optional<uint32_t> {
		const auto* extract = Producer(Canonical(operand), ValueOpcode::CompositeExtractU32x4);
		if (extract == nullptr || extract->Arg(0).Resolve().TryInstruction() != bvh) return std::nullopt;
		return Immediate(extract->Arg(1));
	};
	for (uint32_t side = 0; side < 2; ++side) {
		const auto* recip = Producer(Canonical(mul->Arg(1 - side)), ValueOpcode::FPRecip32);
		if (recip == nullptr || component(recip->Arg(0)) != 1u) continue;
		const auto direct = component(mul->Arg(side));
		if (direct == 0u) return RayQueryResultKind::T;
		if (direct == 2u) return RayQueryResultKind::U;
		if (direct == 3u) return RayQueryResultKind::V;
	}
	return std::nullopt;
}

std::optional<RayQueryResultKind> HitDistance(const Program& program, const PacketWalk& w, const Inst* value,
                                              const Inst* bvh) {
	const auto* mul = Producer(Canonical(Value(const_cast<Inst*>(value))), ValueOpcode::FPMul32);
	if (mul == nullptr) return std::nullopt;
	for (uint32_t side = 0; side < 2; ++side) {
		if (!Reciprocal(program, w, mul->Arg(1 - side), bvh)) continue;
		switch (Components(program, w, mul->Arg(side), bvh) & ~2u) {
			case 1u << 0: return RayQueryResultKind::T;
			case 1u << 2: return RayQueryResultKind::U;
			case 1u << 3: return RayQueryResultKind::V;
			default: return std::nullopt;
		}
	}
	return std::nullopt;
}

bool TlasNodeCount80(const Program& program, const LaneMaskWalk& s) {
	Value size = Producer(s.bvh->Arg(0), ValueOpcode::CompositeConstructU32x4)->Arg(2);
	if (const auto* part = Producer(size, ValueOpcode::CompositeExtractU64);
	    part != nullptr && Immediate(part->Arg(1)) == 0u) {
		const auto* masked = Producer(part->Arg(0), ValueOpcode::BitwiseAnd64);
		const auto* wide =
		    masked != nullptr ? Producer(masked->Arg(0), ValueOpcode::CompositeConstructU64) : nullptr;
		if (wide != nullptr) size = wide->Arg(0);
	}
	const auto* less     = Producer(size, ValueOpcode::IAdd32);
	const bool  less_one = less != nullptr && (Immediate(less->Arg(0)) == 0xffffffffu ||
	                                          Immediate(less->Arg(1)) == 0xffffffffu);
	const auto  count    = LoadOf(program, less_one ? (Immediate(less->Arg(0)) ? less->Arg(1) : less->Arg(0)) : size);
	return less_one && count && count->offset == 80u;
}

Value Strip(Value value) {
	for (uint32_t depth = 0; depth < 4; ++depth) {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr ||
		    (inst->GetOpcode() != ValueOpcode::BitCastF32U32 && inst->GetOpcode() != ValueOpcode::BitCastU32F32))
			break;
		value = inst->Arg(0);
	}
	return value.Resolve();
}

std::optional<float> ImmediateF32(Value value) {
	value = value.Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::F32) return std::nullopt;
	return value.F32Value();
}

std::optional<Value> TimesThree(Value value) {
	if (const auto product = Operands(value, ValueOpcode::IMul32, [](Value side) { return Immediate(side) == 3u; }))
		return Canonical(product->second);
	std::optional<Value> base;
	const auto           pair = Operands(value, ValueOpcode::IAdd32, [&](Value side) {
		const auto* shift = Producer(Canonical(side), ValueOpcode::ShiftLeftLogical32);
		if (shift == nullptr || Immediate(shift->Arg(1)) != 1u) return false;
		base = Canonical(shift->Arg(0));
		return true;
	});
	if (!pair || Canonical(pair->second) != *base) return std::nullopt;
	return base;
}

std::optional<std::pair<uint32_t, Value>> ElementOfThree(Value index) {
	if (const auto pair = Operands(index, ValueOpcode::IAdd32, [](Value side) { return Immediate(side).has_value(); })) {
		const auto base = TimesThree(pair->second);
		const auto k    = *Immediate(pair->first);
		if (base && k < 3) return std::pair {k, *base};
	}
	if (const auto base = TimesThree(index)) return std::pair {0u, *base};
	return std::nullopt;
}

std::optional<Value> WaterfallKey(Value offset, uint32_t shift, bool scalar = false) {
	const auto* lsl = Producer(Canonical(offset), ValueOpcode::ShiftLeftLogical32);
	if (lsl == nullptr || Immediate(lsl->Arg(1)) != shift) return std::nullopt;
	if (scalar) return Canonical(lsl->Arg(0));
	const auto* first = Producer(lsl->Arg(0), ValueOpcode::ReadFirstLane);
	if (first == nullptr) return std::nullopt;
	return Canonical(first->Arg(0));
}

std::array<Value, 4> Dwords(const Inst& resource) {
	return {resource.Arg(0).Resolve(), resource.Arg(1).Resolve(), resource.Arg(2).Resolve(), resource.Arg(3).Resolve()};
}

std::optional<std::pair<std::array<Value, 4>, Value>> TableDescriptor(std::span<const Value> dwords, uint32_t shift,
                                                                      bool scalar = false) {
	std::optional<std::array<Value, 4>> table;
	std::optional<Value>                key;
	for (const auto& dword: dwords) {
		const auto* read   = Producer(dword, ValueOpcode::ReadConstBuffer);
		const auto* vsharp = read != nullptr ? Producer(read->Arg(0), ValueOpcode::GetBufferResource) : nullptr;
		const auto  value  = read != nullptr ? WaterfallKey(read->Arg(1), shift, scalar) : std::nullopt;
		if (vsharp == nullptr || !value || (table && *table != Dwords(*vsharp)) || (key && *key != *value))
			return std::nullopt;
		table = Dwords(*vsharp);
		key   = value;
	}
	if (!table) return std::nullopt;
	return std::pair {*table, *key};
}

Value ShiftedRight(Value value, uint32_t shift) {
	const auto* lsr = Producer(Canonical(value), ValueOpcode::ShiftRightLogical32);
	return lsr != nullptr && Immediate(lsr->Arg(1)) == shift ? Canonical(lsr->Arg(0)) : Value {};
}

template <typename KindOf>
std::optional<RayQueryResultKind> SourceKind(const Program& program, const PacketWalk& w, Value value, KindOf&& kind_of) {
	std::optional<RayQueryResultKind> kind;
	std::unordered_set<const Inst*>   seen;
	std::vector<Value>                pending {value};
	while (!pending.empty() && seen.size() < 256) {
		const auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || !InLoop(program, inst->Parent(), w.header, w.latch) || !seen.insert(inst).second) continue;
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::Phi || op == ValueOpcode::SelectU32) {
			for (size_t arg = op == ValueOpcode::Phi ? 0 : 1; arg < inst->NumArgs(); ++arg)
				pending.push_back(inst->Arg(arg));
			continue;
		}
		const auto source = kind_of(inst);
		if (!source) continue;
		if (kind && *kind != *source) return std::nullopt;
		kind = source;
	}
	return kind;
}

template <typename IsNode, typename IsWord>
bool DeriveLaneAlpha(const Program& program, const PacketWalk& w, const BlasWalk& blas, const Inst* bvh,
                     const AddressLoad& record, IsNode&& node, IsWord&& word, LaneAlpha& a) {
	using Op = ValueOpcode;
	const Inst* sample = nullptr;
	const Inst* base   = nullptr;
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		for (const auto& inst: *program.blocks[at]) {
			const auto op = inst.GetOpcode();
			if (op == Op::ImageGatherRaw || op == Op::ImageRead) return false;
			if (op != Op::ImageSampleRaw) continue;
			if (sample != nullptr) return false;
			sample = &inst;
		}
	}
	if (sample == nullptr) return false;
	const auto& sampled = MemoryOf(program, *sample);
	if (sampled.dmask != 8 || sampled.image_sample_flags != Decoder::ImageSampleFlagLod ||
	    sampled.image_dimension != Decoder::ImageDimension::Dim2D)
		return false;
	const auto defined = [&](Value value) { return DefinedBefore(program, value, w.header); };
	const auto* address = Producer(sample->Arg(2), Op::MakeImageAddress);
	if (address == nullptr) return false;
	auto                            level_value = address->Arg(2);
	std::unordered_set<const Block*> other;
	if (const auto* join = Producer(Canonical(level_value), Op::Phi); join != nullptr) {
		level_value = {};
		for (size_t arg = 0; arg < join->NumArgs() && level_value.IsEmpty(); ++arg) {
			const auto* from  = join->PhiBlock(arg);
			const auto  preds = from->ImmPredecessors();
			const auto* split = preds.size() == 1 ? preds[0] : nullptr;
			if (split == nullptr || split->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
			    split->terminator.merge_block != join->Parent())
				continue;
			const auto [ref, negated] = BranchCondition(*split);
			const auto c = ref != nullptr && (from == split->terminator.true_block) != negated
			                   ? Operands(ref->Arg(0), Op::INotEqual32, [](Value side) { return Immediate(side) == 0u; })
			                   : std::nullopt;
			if (!c || !defined(c->second)) continue;
			a.level     = c->second.Resolve();
			level_value = join->Arg(arg);
			std::vector<const Block*> pending {from == split->terminator.true_block ? split->terminator.false_block
			                                                                         : split->terminator.true_block};
			while (!pending.empty()) {
				const auto* block = pending.back();
				pending.pop_back();
				if (block == join->Parent() || !InLoop(program, block, split, join->Parent()) || !other.insert(block).second)
					continue;
				for (const auto* next: block->ImmSuccessors())
					pending.push_back(next);
			}
		}
		if (level_value.IsEmpty()) return false;
	}
	std::vector<const Inst*> loads, buffer_loads, pairs, reads;
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		if (other.contains(program.blocks[at])) continue;
		for (const auto& inst: *program.blocks[at]) {
			switch (inst.GetOpcode()) {
				case Op::LoadAddressU32: loads.push_back(&inst); break;
				case Op::LoadBufferU32: buffer_loads.push_back(&inst); break;
				case Op::LoadBufferU32x2: pairs.push_back(&inst); break;
				case Op::ReadConstBuffer: reads.push_back(&inst); break;
				default: break;
			}
		}
	}
	const auto alpha = [&](Value value) {
		const auto* w3 = Producer(Canonical(value), Op::CompositeExtractU32x4);
		return w3 != nullptr && w3->Arg(0).Resolve().TryInstruction() == sample && Immediate(w3->Arg(1)) == 3u;
	};
	const Inst* compare = nullptr;
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		for (const auto& inst: *program.blocks[at]) {
			const auto op   = inst.GetOpcode();
			const int  side = op == Op::FPOrdGreaterThan32 && alpha(inst.Arg(0)) ? 1 : op == Op::FPOrdLessThan32 && alpha(inst.Arg(1)) ? 0 : -1;
			if (side < 0) continue;
			if (compare != nullptr) return false;
			compare     = &inst;
			a.reference = Strip(inst.Arg(side));
		}
	}
	if (compare == nullptr) return false;
	Value factor_m0;
	if (const auto* first = a.scalar ? Producer(Canonical(a.reference), Op::ReadFirstLane) : nullptr; first != nullptr)
		a.reference = Strip(Canonical(first->Arg(0)));
	if (!defined(a.reference)) {
		const auto product = Operands(a.reference, Op::FPMul32, [](Value side) { return ImmediateF32(side).has_value(); });
		if (!product) return false;
		const auto* choice = Producer(Strip(product->second), Op::SelectU32);
		for (uint32_t depth = 0; choice != nullptr && depth < 4 && Immediate(choice->Arg(2)) != 0x3f800000u; ++depth)
			choice = Producer(Strip(choice->Arg(1)), Op::SelectU32);
		if (a.scalar && defined(Strip(product->second))) {
			a.factor        = Strip(product->second);
			a.factor_always = true;
		} else {
			if (choice == nullptr || Immediate(choice->Arg(2)) != 0x3f800000u || !defined(Strip(choice->Arg(1)))) return false;
			std::vector<Value> conditions {choice->Arg(0)};
			if (const auto* both = Producer(choice->Arg(0), Op::LogicalAnd); both != nullptr) {
				conditions.push_back(both->Arg(0));
				conditions.push_back(both->Arg(1));
			}
			for (const auto& condition: conditions) {
				const auto* ne  = Producer(condition, Op::INotEqual32);
				const auto  low = ne != nullptr ? Operands(condition, Op::INotEqual32, [](Value side) { return Immediate(side) == 0u; })
				                                : std::nullopt;
				const auto  bit = low ? Operands(low->second, Op::BitwiseAnd32, [](Value side) { return Immediate(side) == 1u; })
				                      : std::nullopt;
				if (bit) factor_m0 = Canonical(bit->second);
				const auto* eq  = Producer(condition, Op::IEqual32);
				const auto  one = eq != nullptr ? Operands(condition, Op::IEqual32, [](Value side) { return Immediate(side) == 1u; })
				                                : std::nullopt;
				const auto* bfe = one ? Producer(Canonical(one->second), Op::BitFieldUExtract) : nullptr;
				if (bfe != nullptr && Immediate(bfe->Arg(1)) == 0u && Immediate(bfe->Arg(2)) == 1u)
					factor_m0 = Canonical(bfe->Arg(0));
			}
			if (factor_m0.IsEmpty()) return false;
			a.factor = Strip(choice->Arg(1));
		}
		a.reference = Value(std::bit_cast<uint32_t>(*ImmediateF32(product->first)));
	}
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch) && a.flag == nullptr; ++at) {
		for (const auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != Op::SelectU32 || Immediate(inst.Arg(1)) != 1u || Immediate(inst.Arg(2)) != 0u) continue;
			const auto* condition = inst.Arg(0).Resolve().TryInstruction();
			const auto* both      = Producer(inst.Arg(0), Op::LogicalAnd);
			if (condition == compare ||
			    (both != nullptr && (both->Arg(0).Resolve().TryInstruction() == compare ||
			                         both->Arg(1).Resolve().TryInstruction() == compare))) {
				a.flag = &inst;
				break;
			}
		}
	}
	if (a.flag == nullptr) return false;
	const auto* image = Producer(sample->Arg(0), Op::GetImageResource);
	if (image == nullptr) return false;
	std::array<Value, 8> tsharp;
	for (size_t k = 0; k < 8; ++k)
		tsharp[k] = image->Arg(k);
	const auto textures = TableDescriptor(tsharp, 5, a.scalar);
	const auto m2       = textures ? ShiftedRight(textures->second, 4) : Value {};
	const auto* m2_load = Producer(m2, a.scalar ? Op::ReadConstBuffer : Op::LoadBufferU32);
	const auto descriptor_defined = [&](Value value) { return DescriptorDefinedBefore(program, value, w.header); };
	if (m2_load == nullptr || !std::ranges::all_of(textures->first, descriptor_defined)) return false;
	const auto material_element = [&](const Inst* load) -> std::optional<std::pair<uint32_t, Value>> {
		if (!a.scalar) return ElementOfThree(load->Arg(1));
		auto       at    = MemoryOf(program, *load).offset;
		auto       index = Canonical(load->Arg(1));
		const auto plus  = Operands(index, Op::IAdd32, [](Value side) { return Immediate(side).has_value(); });
		if (plus) {
			at += *Immediate(plus->first);
			index = Canonical(plus->second);
		}
		const auto* lsl = Producer(index, Op::ShiftLeftLogical32);
		const auto  m   = lsl != nullptr && Immediate(lsl->Arg(1)) == 2u ? TimesThree(lsl->Arg(0)) : std::nullopt;
		if (!m || at % 4 != 0 || at / 4 >= 3) return std::nullopt;
		return std::pair {at / 4, *m};
	};
	const auto* material = Producer(m2_load->Arg(0), Op::GetBufferResource);
	const auto  element  = material_element(m2_load);
	if (material == nullptr || !element || element->first != 2) return false;
	a.material = Dwords(*material);
	if (!std::ranges::all_of(a.material, descriptor_defined)) return false;
	std::array<Value, 3> m {};
	for (const auto* load: a.scalar ? reads : buffer_loads) {
		const auto* vsharp = Producer(load->Arg(0), Op::GetBufferResource);
		const auto  at     = material_element(load);
		if (vsharp == nullptr || !at || at->second != element->second || Dwords(*vsharp) != a.material) continue;
		if (!m[at->first].IsEmpty() && m[at->first] != Value(const_cast<Inst*>(load))) return false;
		m[at->first] = Value(const_cast<Inst*>(load));
	}
	if (m[0].IsEmpty() || m[1].IsEmpty() || m[2].IsEmpty() || (!factor_m0.IsEmpty() && factor_m0 != m[0])) return false;
	const auto  descriptor = Producer(bvh->Arg(0), Op::CompositeConstructU32x4);
	const auto* blas_base  = descriptor != nullptr ? DescriptorAddress(descriptor->Arg(0)) : nullptr;
	const auto same = [&](Value a, Value b) {
		if (a.Resolve() == b.Resolve()) return true;
		const auto x = LoadOf(program, a), y = LoadOf(program, b);
		return x && y && x->low == y->low && x->high == y->high && x->offset == y->offset;
	};
	const auto  of_blas    = [&](const Inst* handle) {
		if (handle != nullptr && blas_base != nullptr && blas_base->GetOpcode() == Op::CompositeConstructU64 &&
		    same(blas_base->Arg(0), handle->Arg(0)) && same(blas_base->Arg(1), handle->Arg(1)))
			return true;
		const auto* low  = handle != nullptr ? Producer(handle->Arg(0), Op::CompositeExtractU64) : nullptr;
		const auto* high = handle != nullptr ? Producer(handle->Arg(1), Op::CompositeExtractU64) : nullptr;
		return low != nullptr && high != nullptr && blas_base != nullptr && Immediate(low->Arg(1)) == 0u &&
		       Immediate(high->Arg(1)) == 1u && low->Arg(0).Resolve().TryInstruction() == blas_base &&
		       high->Arg(0).Resolve().TryInstruction() == blas_base;
	};
	const auto dword_of = [](uint32_t offset, uint32_t limit) -> std::optional<uint32_t> {
		if (offset % 4 != 0 || offset / 4 >= limit) return std::nullopt;
		return offset / 4;
	};
	std::optional<uint32_t> material_dword, table_dword, first_dword;
	const Inst*             geometry = nullptr;
	const auto              sum      = Operands(element->second, Op::IAdd32, [&](Value side) {
		const auto load = LoadOf(program, Canonical(side));
		if (!load || load->low != record.low || load->high != record.high) return false;
		material_dword = dword_of(load->offset, 32);
		return material_dword.has_value();
	});
	if (sum) geometry = Producer(Canonical(sum->second), Op::LoadAddressU32);
	if (geometry == nullptr || MemoryOf(program, *geometry).offset != 0 ||
	    !of_blas(Producer(geometry->Arg(0), Op::GetAddressResource)))
		return false;
	const auto of_blas_field = [&](Value value, std::optional<uint32_t>& dword) {
		const auto* load = Producer(Canonical(value), Op::LoadAddressU32);
		if (load == nullptr || Immediate(load->Arg(1)) != 0u || Immediate(load->Arg(2)) != 0u ||
		    !of_blas(Producer(load->Arg(0), Op::GetAddressResource)))
			return false;
		dword = dword_of(MemoryOf(program, *load).offset, 64);
		return dword.has_value();
	};
	const auto offset = Operands(geometry->Arg(1), Op::IAdd32, [&](Value side) { return of_blas_field(side, table_dword); });
	const auto* scaled = offset ? Producer(Canonical(offset->second), Op::ShiftLeftLogical32) : nullptr;
	if (scaled == nullptr || Immediate(scaled->Arg(1)) != 2u) return false;
	auto local = Operands(scaled->Arg(0), Op::IAdd32, [&](Value side) {
		const auto* negate = Producer(Canonical(side), Op::ISub32);
		return negate != nullptr && Immediate(negate->Arg(0)) == 0u && of_blas_field(negate->Arg(1), first_dword);
	});
	if (const auto* sub = Producer(Canonical(scaled->Arg(0)), Op::ISub32);
	    !local && sub != nullptr && of_blas_field(sub->Arg(1), first_dword))
		local = std::pair {sub->Arg(1), sub->Arg(0)};
	const auto* lsr = local ? Producer(Canonical(local->second), Op::ShiftRightLogical32) : nullptr;
	if (lsr == nullptr || Immediate(lsr->Arg(1)) != 3u || !node(lsr->Arg(0))) return false;
	std::array<Value, 3> indices {};
	for (const auto* load: buffer_loads) {
		const auto* vsharp = Producer(load->Arg(0), Op::GetBufferResource);
		if (vsharp == nullptr) continue;
		const auto dwords = Dwords(*vsharp);
		const auto table  = TableDescriptor(dwords, 4, a.scalar);
		if (!table || ShiftedRight(table->second, 8) != m[1]) continue;
		const auto at = ElementOfThree(load->Arg(1));
		const auto primitive =
		    at ? Operands(at->second, Op::BitwiseAnd32, [](Value side) { return Immediate(side) == 0x3fffffffu; }) : std::nullopt;
		if (!primitive || !word(primitive->second) || !indices[at->first].IsEmpty() ||
		    (!a.buffers[0].IsEmpty() && table->first != a.buffers) || !std::ranges::all_of(table->first, descriptor_defined))
			return false;
		a.buffers           = table->first;
		indices[at->first]  = Value(const_cast<Inst*>(load));
	}
	if (indices[0].IsEmpty() || indices[1].IsEmpty() || indices[2].IsEmpty()) return false;
	const auto vertex_table = [&](const Inst* load) {
		const auto* vsharp = Producer(load->Arg(0), Op::GetBufferResource);
		const auto  table  = vsharp != nullptr ? TableDescriptor(Dwords(*vsharp), 4, true) : std::nullopt;
		return table && table->first == a.buffers && ShiftedRight(table->second, 8) == m[0];
	};
	for (const auto* pair: a.scalar ? std::vector<const Inst*> {} : pairs) {
		const auto* table = Producer(pair->Arg(0), Op::GetBufferResource);
		if (table == nullptr || ShiftedRight(pair->Arg(1), 8) != m[0] || Dwords(*table) != a.buffers) continue;
		if (base != nullptr) return false;
		base = pair;
	}
	if (base == nullptr && !a.scalar) return false;
	const auto masked = [&](Value value, Value of, uint32_t& mask) {
		value = Canonical(value);
		if (const auto* bits = Producer(value, Op::BitFieldUExtract);
		    bits != nullptr && Immediate(bits->Arg(1)) == 0u && Immediate(bits->Arg(2)) == 8u && Canonical(bits->Arg(0)) == of) {
			mask = 0xffu;
			return true;
		}
		const auto pair = Operands(value, Op::BitwiseAnd32, [](Value side) {
			return Immediate(side) == 0xfcu || Immediate(side) == 0xffu;
		});
		if (!pair || Canonical(pair->second) != of) return false;
		mask = *Immediate(pair->first);
		return true;
	};
	const auto at_base = [&](Value side) {
		const auto* low = Producer(Canonical(side), Op::CompositeExtractU32x2);
		return low != nullptr && low->Arg(0).Resolve().TryInstruction() == base && Immediate(low->Arg(1)) == 0u;
	};
	uint32_t vertices = 0, seconds = 0, vertex_mask = 0;
	bool     vertex_align = false;
	for (const auto* load: a.scalar ? reads : loads) {
		Value value;
		bool  second = false;
		if (a.scalar) {
			const auto at = MemoryOf(program, *load).offset;
			if (!vertex_table(load) || (at != 0 && at != 4)) continue;
			value  = Canonical(load->Arg(1));
			second = at == 4;
		} else {
			const auto* handle = Producer(load->Arg(0), Op::GetAddressResource);
			const auto  sum_of = handle != nullptr ? Operands(handle->Arg(0), Op::IAdd32, at_base) : std::nullopt;
			if (!sum_of || MemoryOf(program, *load).offset != 0) continue;
			value           = Canonical(sum_of->second);
			const auto plus = Operands(value, Op::IAdd32, [](Value side) { return Immediate(side) == 4u; });
			if (plus) value = Canonical(plus->second);
			second = plus.has_value();
		}
		const auto align = Operands(value, Op::BitwiseAnd32, [](Value side) { return Immediate(side) == 0xfffffffcu; });
		if (align) value = Canonical(align->second);
		uint32_t   uv_mask = 0, stride_mask = 0;
		const auto parts   = Operands(value, Op::IAdd32, [&](Value side) { return masked(side, m[1], uv_mask); });
		const auto product = parts ? Operands(parts->second, Op::IMul32, [&](Value side) { return masked(side, m[0], stride_mask); })
		                           : std::nullopt;
		auto       index   = product ? Canonical(product->second) : Value {};
		if (const auto* first = a.scalar && product ? Producer(index, Op::ReadFirstLane) : nullptr; first != nullptr)
			index = Canonical(first->Arg(0));
		if (!product || stride_mask != uv_mask ||
		    std::ranges::none_of(indices, [&](Value candidate) { return index == candidate; }) ||
		    (vertices != 0 && (uv_mask != vertex_mask || align.has_value() != vertex_align)))
			return false;
		vertex_mask  = uv_mask;
		vertex_align = align.has_value();
		seconds += second ? 1u : 0u;
		++vertices;
	}
	if (vertices != 6 || seconds != 3) return false;
	const auto level = Operands(level_value, Op::FPMin32, [&](Value side) {
		const auto* convert = Producer(Canonical(side), Op::ConvertF32U32);
		const auto  pair    = convert != nullptr ? Operands(convert->Arg(0), Op::BitwiseAnd32, [](Value v) { return Immediate(v) == 15u; })
		                                         : std::nullopt;
		return pair && Canonical(pair->second) == m[2];
	});
	const auto offset_add = level ? Operands(level->second, Op::FPAdd32, [](Value side) { return ImmediateF32(side).has_value(); })
	                              : std::nullopt;
	const auto* log       = offset_add ? Producer(Canonical(offset_add->second), Op::FPLog2) : nullptr;
	const auto  twice     = log != nullptr ? Operands(log->Arg(0), Op::FPMul32, [](Value side) { return ImmediateF32(side) == 2.0f; })
	                                       : std::nullopt;
	const auto* mad       = twice ? Producer(Canonical(twice->second), Op::FPMad32) : nullptr;
	if (mad == nullptr) return false;
	for (uint32_t side = 0; side < 2; ++side) {
		const auto* t = Canonical(mad->Arg(side)).TryInstruction();
		if (t != nullptr && WrittenHitDistance(Value(const_cast<Inst*>(t)), bvh) == RayQueryResultKind::T) {
			a.spread = Strip(mad->Arg(1 - side));
			a.lod    = Strip(mad->Arg(2));
			break;
		}
	}
	if (a.spread.IsEmpty() || !defined(a.spread) || !defined(a.lod)) return false;
	a.lod_offset                = std::bit_cast<uint32_t>(*ImmediateF32(offset_add->first));
	a.fields.material_dword     = *material_dword;
	a.fields.table_dword        = *table_dword;
	a.fields.first_dword        = *first_dword;
	a.fields.vertex_mask        = vertex_mask;
	a.fields.vertex_align       = vertex_align;
	a.valid                     = true;
	return true;
}

template <typename KindOf>
Value FlaggedAccept(const Program& program, const BlasWalk& blas, const LaneAlpha& alpha, RayQueryResultKind kind,
                    KindOf&& kind_of, Value* lanes = nullptr, Value* direct = nullptr) {
	Value      one;
	const auto arm = [&](Value value) {
		for (uint32_t depth = 0; depth < 8; ++depth) {
			value = value.Resolve();
			if (value.TryInstruction() == alpha.flag) return 1;
			if (Immediate(value) == 1u) return 2;
			const auto* select = Producer(value, ValueOpcode::SelectU32);
			if (select == nullptr) return 0;
			if (Immediate(select->Arg(1)) == 1u && Immediate(select->Arg(2)) == 0u) one = select->Arg(0).Resolve();
			value = select->Arg(1);
		}
		return 0;
	};
	const auto flagged = [&](Value condition) {
		const auto* both = Producer(condition, ValueOpcode::LogicalAnd);
		for (uint32_t side = 0; both != nullptr && side < 2; ++side) {
			const auto* ne = Producer(both->Arg(side), ValueOpcode::INotEqual32);
			for (uint32_t k = 0; ne != nullptr && k < 2; ++k) {
				const auto* phi = Immediate(ne->Arg(k)) == 0u ? Producer(ne->Arg(1 - k), ValueOpcode::Phi) : nullptr;
				if (phi == nullptr) continue;
				uint32_t arms = 0;
				one           = {};
				for (size_t arg = 0; arg < phi->NumArgs(); ++arg)
					arms |= 1u << arm(phi->Arg(arg));
				if (arms != 6u) continue;
				if (lanes != nullptr) *lanes = both->Arg(1 - side).Resolve();
				if (direct != nullptr) *direct = one;
				return true;
			}
		}
		return false;
	};
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		for (const auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != ValueOpcode::SelectU32) continue;
			const auto* written = Canonical(inst.Arg(1)).TryInstruction();
			if (written != nullptr && kind_of(written) == kind && flagged(inst.Arg(0))) return inst.Arg(0).Resolve();
		}
	}
	return {};
}

bool MatchClosestOpaqueWalk(Program& program, PacketWalk& w) {
	auto& s = w.shape;
	if (s.blases.empty()) return false;
	const auto& blas = s.blases.front();
	const Inst* bvh  = nullptr;
	bool        textures = false;
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		for (const auto& inst: *program.blocks[at]) {
			const auto op = inst.GetOpcode();
			textures |= op == ValueOpcode::ImageSampleRaw || op == ValueOpcode::ImageGatherRaw || op == ValueOpcode::ImageRead;
			if (op == ValueOpcode::BvhIntersect) {
				if (bvh != nullptr) return false;
				bvh = &inst;
			}
		}
	}
	const auto record = LoadOf(program, Canonical(IncomingValue(*blas.node, blas.preheader)));
	if (bvh == nullptr || !record) return false;
	const auto kind_of = [&](const Inst* value) -> std::optional<RayQueryResultKind> {
		if (value == record->low.TryInstruction()) return RayQueryResultKind::InstanceLo;
		if (value == record->high.TryInstruction()) return RayQueryResultKind::InstanceHi;
		if (value->GetOpcode() == ValueOpcode::CompositeExtractU32x2 &&
		    value->Arg(0).Resolve().TryInstruction() == blas.load) {
			const auto word = Immediate(value->Arg(1));
			if (word == 0u) return RayQueryResultKind::Node;
			if (word == 1u) return RayQueryResultKind::Word;
			return std::nullopt;
		}
		if (const auto written = w.alpha.valid ? WrittenHitDistance(Value(const_cast<Inst*>(value)), bvh) : std::nullopt)
			return written;
		return HitDistance(program, w, value, bvh);
	};
	const auto of_kind = [&](RayQueryResultKind kind) {
		return [&, kind](Value value) { return SourceKind(program, w, value, kind_of) == kind; };
	};
	w.alpha = {};
	if (textures && !DeriveLaneAlpha(program, w, blas, bvh, *record, of_kind(RayQueryResultKind::Node),
	                                 of_kind(RayQueryResultKind::Word), w.alpha))
		return false;
	Value accept;
	if (w.alpha.valid) {
		accept = FlaggedAccept(program, blas, w.alpha, RayQueryResultKind::U, kind_of);
	} else {
		for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch) && accept.IsEmpty(); ++at) {
			for (const auto& inst: *program.blocks[at]) {
				if (inst.GetOpcode() != ValueOpcode::SelectU32) continue;
				const auto* written = inst.Arg(1).Resolve().TryInstruction();
				if (written != nullptr && written->GetOpcode() != ValueOpcode::Phi && written->GetOpcode() != ValueOpcode::SelectU32 &&
				    kind_of(written) == RayQueryResultKind::U) {
					accept = inst.Arg(0).Resolve();
					break;
				}
			}
		}
	}
	const auto found = !accept.IsEmpty() ? HitResults(program, w, kind_of, accept) : std::nullopt;
	if (found) {
		s.layout.node_count_80 = TlasNodeCount80(program, s);
		s.source.layout        = s.layout.Pack();
	}
	using Kind = RayQueryResultKind;
	if (!found || !std::ranges::all_of(std::array {Kind::Word, Kind::Node, Kind::InstanceLo, Kind::InstanceHi, Kind::U, Kind::V},
	                                   [&](Kind kind) { return (*found)[static_cast<uint32_t>(kind)]; })) {
		w.alpha = {};
		return false;
	}
	return true;
}

struct LaneTable {
	static constexpr std::array<uint32_t, 5> Tables {0xaaaaaaaau, 0xccccccccu, 0xf0f0f0f0u, 0xff00ff00u,
	                                                 0xffff0000u};

	const Inst*              exec = nullptr;
	Value                    exec_mask;
	std::vector<const Inst*> atoms;
	bool                     flags   = false;
	const Program*           program = nullptr;
	const Block*             before  = nullptr;

	std::optional<uint32_t> Atom(const Inst* inst) {
		const auto at = std::ranges::find(atoms, inst);
		if (at != atoms.end()) return Tables[at - atoms.begin()];
		if (atoms.size() == Tables.size()) return std::nullopt;
		atoms.push_back(inst);
		return Tables[atoms.size() - 1];
	}

	std::optional<uint32_t> Condition(Value value, uint32_t depth = 0) {
		value = value.Resolve();
		if (value.IsImmediate())
			return value.GetType() == Type::U1 ? std::optional(value.U1() ? ~0u : 0u) : std::nullopt;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 32) return std::nullopt;
		if (inst == exec) return ~0u;
		const auto op = inst->GetOpcode();
		if (before != nullptr && (op == ValueOpcode::LogicalNot || op == ValueOpcode::LogicalAnd || op == ValueOpcode::LogicalOr) &&
		    DefinedBefore(*program, value, before))
			return Atom(inst);
		if (op == ValueOpcode::LogicalNot) {
			const auto a = Condition(inst->Arg(0), depth + 1);
			return a ? std::optional(~*a) : std::nullopt;
		}
		if (op == ValueOpcode::LogicalAnd || op == ValueOpcode::LogicalOr || op == ValueOpcode::LogicalXor) {
			const auto a = Condition(inst->Arg(0), depth + 1), b = Condition(inst->Arg(1), depth + 1);
			if (!a || !b) return std::nullopt;
			return op == ValueOpcode::LogicalAnd ? *a & *b : op == ValueOpcode::LogicalOr ? *a | *b : *a ^ *b;
		}
		if (const auto mask = LaneMaskOf(inst); !mask.IsEmpty()) return Mask(mask, depth + 1);
		if (flags && (op == ValueOpcode::IEqual32 || op == ValueOpcode::INotEqual32)) {
			for (uint32_t k = 0; k < 2; ++k) {
				if (Immediate(inst->Arg(k)) != 0u || inst->Arg(1 - k).Resolve().IsImmediate() || !IsFlag(inst->Arg(1 - k)))
					continue;
				const auto flag = Flag(inst->Arg(1 - k), depth + 1);
				if (!flag) return std::nullopt;
				return op == ValueOpcode::IEqual32 ? ~*flag : *flag;
			}
		}
		return Atom(inst);
	}

	[[nodiscard]] bool Covers(Value condition) const {
		condition = condition.Resolve();
		if (exec == nullptr || condition.TryInstruction() == exec) return exec != nullptr;
		if (exec->GetOpcode() != ValueOpcode::Phi) return false;
		bool entry = false;
		for (size_t arg = 0; arg < exec->NumArgs(); ++arg) {
			if (exec->Arg(arg).Resolve() == condition) {
				entry = true;
			} else if (!Within(exec->Arg(arg))) {
				return false;
			}
		}
		return entry;
	}

	[[nodiscard]] bool Within(Value value, uint32_t depth = 0) const {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == exec) return true;
		if (inst == nullptr || depth > 8 || inst->GetOpcode() != ValueOpcode::LogicalAnd) return false;
		return Within(inst->Arg(0), depth + 1) || Within(inst->Arg(1), depth + 1);
	}

	bool IsFlag(Value value, uint32_t depth = 0) {
		value = value.Resolve();
		if (const auto constant = Immediate(value)) return *constant <= 1u;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 16) return false;
		switch (inst->GetOpcode()) {
			case ValueOpcode::SelectU32:
				return IsFlag(inst->Arg(1), depth + 1) && (Covers(inst->Arg(0)) || IsFlag(inst->Arg(2), depth + 1));
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseOr32: return IsFlag(inst->Arg(0), depth + 1) && IsFlag(inst->Arg(1), depth + 1);
			default: return false;
		}
	}

	std::optional<uint32_t> Flag(Value value, uint32_t depth = 0) {
		value = value.Resolve();
		if (const auto constant = Immediate(value)) {
			return *constant == 1u ? std::optional(~0u) : *constant == 0u ? std::optional(0u) : std::nullopt;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 32) return std::nullopt;
		if (inst->GetOpcode() == ValueOpcode::BitwiseAnd32 || inst->GetOpcode() == ValueOpcode::BitwiseOr32) {
			const auto a = Flag(inst->Arg(0), depth + 1), b = Flag(inst->Arg(1), depth + 1);
			if (!a || !b) return std::nullopt;
			return inst->GetOpcode() == ValueOpcode::BitwiseAnd32 ? *a & *b : *a | *b;
		}
		const auto* select = Producer(value, ValueOpcode::SelectU32);
		if (select == nullptr) return std::nullopt;
		if (Covers(select->Arg(0))) return Flag(select->Arg(1), depth + 1);
		const auto c = Condition(select->Arg(0), depth + 1);
		const auto a = Flag(select->Arg(1), depth + 1), b = Flag(select->Arg(2), depth + 1);
		if (!c || !a || !b) return std::nullopt;
		return (*c & *a) | (~*c & *b);
	}

	std::optional<uint32_t> Mask(Value value, uint32_t depth) {
		value = value.Resolve();
		if (const auto constant = Immediate(value); constant) {
			return *constant == 0u ? std::optional(0u) : *constant == ~0u ? std::optional(~0u) : std::nullopt;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 32) return std::nullopt;
		if (!exec_mask.IsEmpty() && value == exec_mask) return ~0u;
		switch (inst->GetOpcode()) {
			case ValueOpcode::BitwiseNot32: {
				const auto a = Mask(inst->Arg(0), depth + 1);
				return a ? std::optional(~*a) : std::nullopt;
			}
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32: {
				const auto a = Mask(inst->Arg(0), depth + 1), b = Mask(inst->Arg(1), depth + 1);
				if (!a || !b) return std::nullopt;
				return inst->GetOpcode() == ValueOpcode::BitwiseAnd32  ? *a & *b
				       : inst->GetOpcode() == ValueOpcode::BitwiseOr32 ? *a | *b
				                                                       : *a ^ *b;
			}
			case ValueOpcode::CompositeExtractU32x4: {
				const auto* ballot = BallotOf(value);
				return ballot != nullptr ? Condition(ballot->Arg(0), depth + 1) : std::nullopt;
			}
			case ValueOpcode::SelectU32: {
				const auto c = Condition(inst->Arg(0), depth + 1);
				const auto a = Mask(inst->Arg(1), depth + 1), b = Mask(inst->Arg(2), depth + 1);
				if (!c || !a || !b) return std::nullopt;
				return (*c & *a) | (~*c & *b);
			}
			default: return std::nullopt;
		}
	}
};

std::optional<std::pair<uint32_t, uint32_t>> RecordField(const Program& program, const LaneMaskWalk& s, Value value) {
	value         = Canonical(value);
	uint32_t mask = ~0u;
	for (uint32_t depth = 0; depth < 3; ++depth) {
		const auto* bits = Producer(value, ValueOpcode::BitwiseAnd32);
		const auto  b    = bits == nullptr ? 2u : Immediate(bits->Arg(0)) ? 0u : Immediate(bits->Arg(1)) ? 1u : 2u;
		if (b == 2u) break;
		mask &= *Immediate(bits->Arg(b));
		value = Canonical(bits->Arg(1 - b));
	}
	const auto load = LoadOf(program, value);
	if (!load || load->low != s.record.low || load->high != s.record.high || (load->offset & 3u) != 0)
		return std::nullopt;
	return std::pair {load->offset / 4u, mask};
}

std::optional<std::pair<uint32_t, uint32_t>> RecordBit(const Program& program, const LaneMaskWalk& s, Value value) {
	const auto* inst = Canonical(value).TryInstruction();
	if (inst == nullptr) return std::nullopt;
	const auto op  = inst->GetOpcode();
	const auto bit = op == ValueOpcode::BitFieldUExtract || op == ValueOpcode::ShiftRightLogical32
	                     ? Immediate(inst->Arg(1))
	                     : std::nullopt;
	if (bit && *bit < 32u) {
		const auto f = RecordField(program, s, inst->Arg(0));
		if (f && ((f->second >> *bit) & 1u) != 0 &&
		    ((op == ValueOpcode::BitFieldUExtract && Immediate(inst->Arg(2)) == 1u) ||
		     (op == ValueOpcode::ShiftRightLogical32 && (f->second >> *bit) == 1u)))
			return std::pair {f->first, *bit};
	}
	for (uint32_t side = 0; op == ValueOpcode::BitwiseAnd32 && side < 2; ++side) {
		const auto* shift = Producer(inst->Arg(side), ValueOpcode::ShiftRightLogical32);
		const auto  by    = shift != nullptr ? Immediate(shift->Arg(1)) : std::nullopt;
		if (Immediate(inst->Arg(1 - side)) != 1u || !by || *by >= 32u) continue;
		if (const auto f = RecordField(program, s, shift->Arg(0)); f && ((f->second >> *by) & 1u) != 0)
			return std::pair {f->first, *by};
	}
	return std::nullopt;
}

std::optional<std::tuple<uint32_t, uint32_t, bool>> RecordBitTest(const Program& program, const LaneMaskWalk& s,
                                                                  const Inst* test) {
	const auto op = test->GetOpcode();
	if (op != ValueOpcode::IEqual32 && op != ValueOpcode::INotEqual32) return std::nullopt;
	for (uint32_t side = 0; side < 2; ++side) {
		const auto constant = Immediate(test->Arg(side));
		const auto other    = test->Arg(1 - side);
		if (constant == 0u) {
			if (const auto f = RecordField(program, s, other); f && std::popcount(f->second) == 1) {
				return std::tuple {f->first, static_cast<uint32_t>(std::countr_zero(f->second)),
				                   op == ValueOpcode::INotEqual32};
			}
			if (const auto bit = RecordBit(program, s, other))
				return std::tuple {bit->first, bit->second, op == ValueOpcode::INotEqual32};
		} else if (constant == 1u) {
			if (const auto bit = RecordBit(program, s, other))
				return std::tuple {bit->first, bit->second, op == ValueOpcode::IEqual32};
		}
	}
	return std::nullopt;
}

template <typename KindOf>
bool LeafFacing(const Program& program, PacketWalk& w, Value condition, bool negated, const Inst* tri, Value current,
                KindOf&& kind_of, bool flags = false, bool* leaf = nullptr) {
	auto&     s = w.shape;
	LaneTable lanes;
	lanes.exec       = tri->Arg(2).Resolve().TryInstruction();
	lanes.exec_mask  = LaneMaskOf(lanes.exec);
	lanes.flags      = flags;
	lanes.program    = flags ? &program : nullptr;
	lanes.before     = flags ? w.header : nullptr;
	const auto table = lanes.Condition(condition);
	if (!table) return false;
	const auto raw = [&](Value value, uint32_t component) {
		const auto* extract = Producer(Canonical(value), ValueOpcode::CompositeExtractU32x4);
		return extract != nullptr && extract->Arg(0).Resolve().TryInstruction() == tri &&
		       Immediate(extract->Arg(1)) == component;
	};
	const auto record_bit = [&](Value value) { return RecordBit(program, s, value); };
	const auto bit_test   = [&](const Inst* test) { return RecordBitTest(program, s, test); };
	constexpr size_t None = 8;
	size_t           hit = None, facing = None, far = None, both = None, enable = None, guest = None;
	bool             near = false, flip = false, both_set = true;
	uint32_t         flags_dword = 0, facing_bit = 0, both_dword = 0, both_bit = 0;
	for (size_t k = 0; k < lanes.atoms.size(); ++k) {
		const auto* atom = lanes.atoms[k];
		const auto  op   = atom->GetOpcode();
		if ((op == ValueOpcode::FPOrdGreaterThan32 || op == ValueOpcode::FPOrdLessThanEqual32) && far == None) {
			const auto* t = Canonical(atom->Arg(0)).TryInstruction();
			if (t != nullptr && kind_of(t) == RayQueryResultKind::T && Canonical(atom->Arg(1)) == current) {
				far  = k;
				near = op == ValueOpcode::FPOrdLessThanEqual32;
				continue;
			}
		}
		if ((op == ValueOpcode::IEqual32 || op == ValueOpcode::INotEqual32) && facing == None) {
			for (uint32_t side = 0; side < 2 && facing == None; ++side) {
				const auto* sign = Producer(Canonical(atom->Arg(side)), ValueOpcode::ShiftRightLogical32);
				const auto  bit  = sign != nullptr && Immediate(sign->Arg(1)) == 31u && raw(sign->Arg(0), 0)
				                       ? record_bit(atom->Arg(1 - side))
				                       : std::nullopt;
				if (bit && bit->second >= 3u) {
					facing      = k;
					flags_dword = bit->first;
					facing_bit  = bit->second;
					flip        = op == ValueOpcode::INotEqual32;
				}
			}
			if (facing == k) continue;
		}
		const bool compare = op == ValueOpcode::IEqual32 || op == ValueOpcode::INotEqual32 ||
		                     op == ValueOpcode::FPOrdNotEqual32 || op == ValueOpcode::FPUnordNotEqual32;
		if (compare && hit == None && (raw(atom->Arg(0), 0) || raw(atom->Arg(1), 0))) {
			hit = k;
			continue;
		}
		if (const auto test = bit_test(atom); test && both == None) {
			std::tie(both_dword, both_bit, both_set) = *test;
			both                                     = k;
			continue;
		}
		if (enable == None && DefinedBefore(program, Value(const_cast<Inst*>(atom)), w.header)) {
			enable = k;
			continue;
		}
		if (leaf != nullptr && guest == None) {
			guest = k;
			continue;
		}
		return false;
	}
	if (hit == None || facing == None || both == None || both_dword != flags_dword || both_bit + 1u != facing_bit)
		return false;
	const auto value = [&](size_t k, uint32_t c) { return k != None && ((c >> k) & 1u) != 0; };
	bool guest_set = true;
	for (uint32_t c = 0; c < 32 && guest != None; ++c) {
		if ((((*table >> c) & 1u) != 0) != negated) {
			guest_set = value(guest, c);
			break;
		}
	}
	for (uint32_t c = 0; c < 32; ++c) {
		const bool h = value(hit, c), f = value(facing, c), g = far != None && value(far, c) != near;
		const bool b2 = value(both, c) == both_set, e = enable == None || value(enable, c);
		const bool a  = guest == None || value(guest, c) == guest_set;
		if ((((*table >> c) & 1u) != 0) != ((h && !g && (f || b2 || !e) && a) != negated)) return false;
	}
	s.layout.flags_dword = flags_dword;
	s.layout.flags_shift = facing_bit - 3u;
	s.layout.flip_facing = flip;
	w.cull               = enable != None ? Value(const_cast<Inst*>(lanes.atoms[enable])) : Value(true);
	w.both_facings       = false;
	if (leaf != nullptr) *leaf = guest != None;
	return true;
}

const Inst* WaveBlasBase(const Inst* tri) {
	const auto* words = Producer(tri->Arg(0), ValueOpcode::CompositeConstructU32x4);
	return words != nullptr ? DescriptorAddress(words->Arg(0)) : nullptr;
}

std::optional<RayQueryResultKind> WaveResultKind(const Program& program, const LaneMaskWalk& s, const Inst* tri,
                                                 const Inst* value) {
	const auto* base = WaveBlasBase(tri);
	if (base == nullptr) return std::nullopt;
	const auto component = [&](Value operand) -> std::optional<uint32_t> {
		const auto* extract = Producer(Canonical(operand), ValueOpcode::CompositeExtractU32x4);
		return extract != nullptr && extract->Arg(0).Resolve().TryInstruction() == tri ? Immediate(extract->Arg(1))
		                                                                               : std::nullopt;
	};
	if (value == s.record.low.TryInstruction()) return RayQueryResultKind::InstanceLo;
	if (value == s.record.high.TryInstruction()) return RayQueryResultKind::InstanceHi;
	if (value->GetOpcode() == ValueOpcode::ReadConstBuffer) {
		const auto* resource = Producer(value->Arg(0), ValueOpcode::GetBufferResource);
		const auto* low =
		    resource != nullptr ? Producer(resource->Arg(0), ValueOpcode::CompositeExtractU64) : nullptr;
		const bool blas =
		    resource != nullptr &&
		    ((low != nullptr && Immediate(low->Arg(1)) == 0u && low->Arg(0).Resolve().TryInstruction() == base) ||
		     (base->GetOpcode() == ValueOpcode::CompositeConstructU64 &&
		      resource->Arg(0).Resolve() == base->Arg(0).Resolve()));
		if (!blas || !InLoop(program, value->Parent(), s.wave_header, s.wave_latch)) return std::nullopt;
		auto at = MemoryOf(program, *value).offset;
		if (const auto* add = Producer(value->Arg(1), ValueOpcode::IAdd32); add != nullptr)
			at += Immediate(add->Arg(0)).value_or(Immediate(add->Arg(1)).value_or(0u));
		at &= 7u;
		return at == 0u   ? std::optional(RayQueryResultKind::Node)
		       : at == 4u ? std::optional(RayQueryResultKind::Word)
		                  : std::nullopt;
	}
	const auto* mul = Producer(Canonical(Value(const_cast<Inst*>(value))), ValueOpcode::FPMul32);
	for (uint32_t side = 0; mul != nullptr && side < 2; ++side) {
		const auto* rcp = Producer(Canonical(mul->Arg(1 - side)), ValueOpcode::FPRecip32);
		if (rcp == nullptr || component(rcp->Arg(0)) != 1u) continue;
		switch (component(mul->Arg(side)).value_or(1u)) {
			case 0: return RayQueryResultKind::T;
			case 2: return RayQueryResultKind::U;
			case 3: return RayQueryResultKind::V;
			default: return std::nullopt;
		}
	}
	return std::nullopt;
}

std::optional<RayQueryResultKind> CopyResultKind(const Program& program, const LaneMaskWalk& s,
                                                 std::span<const Value> currents, const Inst* value) {
	for (const auto* tri: {s.wave_triangles, s.wave_copy}) {
		if (const auto kind = WaveResultKind(program, s, tri, value)) return kind;
	}
	const auto* min = Producer(Canonical(Value(const_cast<Inst*>(value))), ValueOpcode::FPMin32);
	for (uint32_t side = 0; min != nullptr && side < 2; ++side) {
		const auto* t = Canonical(min->Arg(side)).TryInstruction();
		if (t != nullptr && std::ranges::find(currents, Canonical(min->Arg(1 - side))) != currents.end() &&
		    CopyResultKind(program, s, {}, t) == RayQueryResultKind::T)
			return RayQueryResultKind::T;
	}
	return std::nullopt;
}

bool LeafFacingCopies(const Program& program, PacketWalk& w, std::vector<Value>& conditions,
                      std::array<Value, 2>& currents) {
	auto&      s    = w.shape;
	const auto tris = std::array {s.wave_triangles, s.wave_copy};
	std::array<Block*, 2> headers {}, pres {}, latches {};
	for (uint32_t k = 0; k < 2; ++k) {
		Block* merge = nullptr;
		headers[k]   = EnclosingLoop(program, tris[k]->Parent());
		if (!LoopShape(headers[k], pres[k], latches[k], merge) || pres[k]->ImmPredecessors().size() != 1) return false;
	}
	const auto* branch = pres[0]->ImmPredecessors()[0];
	const auto& term   = branch->terminator;
	if (pres[1]->ImmPredecessors()[0] != branch || term.kind != CFG::TerminatorKind::ConditionalBranch ||
	    !((term.true_block == pres[0] && term.false_block == pres[1]) ||
	      (term.true_block == pres[1] && term.false_block == pres[0])))
		return false;
	const auto [ref, inverted] = BranchCondition(*branch);
	const auto* test           = ref != nullptr ? ref->Arg(0).Resolve().TryInstruction() : nullptr;
	const auto  both           = test != nullptr ? RecordBitTest(program, s, test) : std::nullopt;
	if (!both) return false;
	const auto [both_dword, both_bit, both_set] = *both;
	const auto* when_set                        = both_set != inverted ? term.true_block : term.false_block;
	const uint32_t all                          = when_set == pres[0] ? 0u : 1u;
	for (uint32_t k = 0; k < 2; ++k) {
		for (auto at = BlockIndex(program, headers[k]); at <= BlockIndex(program, latches[k]); ++at) {
			for (const auto& inst: *program.blocks[at]) {
				const auto op = inst.GetOpcode();
				if (op != ValueOpcode::FPOrdGreaterThan32 && op != ValueOpcode::FPOrdLessThanEqual32) continue;
				const auto* t = Canonical(inst.Arg(0)).TryInstruction();
				if (t == nullptr || WaveResultKind(program, s, tris[k], t) != RayQueryResultKind::T) continue;
				if (!currents[k].IsEmpty() && currents[k] != Canonical(inst.Arg(1))) return false;
				currents[k] = Canonical(inst.Arg(1));
			}
		}
		if (currents[k].IsEmpty()) return false;
	}
	const auto kind_of = [&](const Inst* value) { return CopyResultKind(program, s, currents, value); };
	uint32_t   flags_dword = 0, facing_bit = 0;
	bool       flip        = false;
	for (uint32_t k = 0; k < 2; ++k) {
		const auto* tri = tris[k];
		LaneTable   lanes;
		lanes.exec      = tri->Arg(2).Resolve().TryInstruction();
		lanes.exec_mask = LaneMaskOf(lanes.exec);
		const auto raw  = [&](Value value, uint32_t component) {
			const auto* extract = Producer(Canonical(value), ValueOpcode::CompositeExtractU32x4);
			return extract != nullptr && extract->Arg(0).Resolve().TryInstruction() == tri &&
			       Immediate(extract->Arg(1)) == component;
		};
		std::optional<uint32_t> distance;
		std::vector<Value>      writes;
		const Inst*             far = nullptr;
		for (auto at = BlockIndex(program, headers[k]); at <= BlockIndex(program, latches[k]); ++at) {
			for (const auto& inst: *program.blocks[at]) {
				const auto op = inst.GetOpcode();
				if ((op == ValueOpcode::FPOrdGreaterThan32 || op == ValueOpcode::FPOrdLessThanEqual32) &&
				    Canonical(inst.Arg(1)) == currents[k]) {
					if (far != nullptr) return false;
					far = &inst;
				}
			}
		}
		for (auto at = BlockIndex(program, headers[k]); at <= BlockIndex(program, latches[k]); ++at) {
			for (const auto& inst: *program.blocks[at]) {
				if (inst.GetOpcode() != ValueOpcode::SelectU32 || inst.Arg(0).Resolve().TryInstruction() == lanes.exec)
					continue;
				const auto condition = inst.Arg(0).Resolve();
				bool       written   = false;
				for (uint32_t arm = 1; arm <= 2 && !written; ++arm) {
					const auto* t = Canonical(inst.Arg(arm)).TryInstruction();
					if (t == nullptr || Canonical(inst.Arg(3 - arm)) != currents[k] || kind_of(t) != RayQueryResultKind::T)
						continue;
					auto table = lanes.Condition(condition);
					if (!table || distance) return false;
					if (arm == 2) *table = ~*table;
					if (Producer(Canonical(Value(const_cast<Inst*>(t))), ValueOpcode::FPMin32) != nullptr) {
						const auto closer = far != nullptr ? lanes.Condition(Value(const_cast<Inst*>(far))) : std::nullopt;
						if (!closer) return false;
						*table &= far->GetOpcode() == ValueOpcode::FPOrdLessThanEqual32 ? *closer : ~*closer;
					}
					distance = table;
					written  = true;
				}
				if (!written) writes.push_back(condition);
				conditions.push_back(condition);
			}
		}
		if (!distance || writes.empty()) return false;
		for (const auto& write: writes) {
			if (lanes.Condition(write) != distance) return false;
		}
		constexpr size_t None = 8;
		size_t           hit = None, near = None, facing = None;
		bool             nearer = true;
		for (size_t a = 0; a < lanes.atoms.size(); ++a) {
			const auto* atom = lanes.atoms[a];
			const auto  op   = atom->GetOpcode();
			if (atom == far) {
				near   = a;
				nearer = op == ValueOpcode::FPOrdLessThanEqual32;
				continue;
			}
			if (op == ValueOpcode::FPCmpClass32 && Immediate(atom->Arg(1)) == 0x38u && hit == None) {
				const auto* negated = Producer(Canonical(atom->Arg(0)), ValueOpcode::FPNeg32);
				const auto* size    = negated != nullptr ? Producer(Canonical(negated->Arg(0)), ValueOpcode::FPAbs32) : nullptr;
				if (size == nullptr || !raw(size->Arg(0), 0)) return false;
				hit = a;
				continue;
			}
			if ((op == ValueOpcode::IEqual32 || op == ValueOpcode::INotEqual32) && facing == None && k != all) {
				for (uint32_t side = 0; side < 2 && facing == None; ++side) {
					const auto* sign = Producer(Canonical(atom->Arg(side)), ValueOpcode::ShiftRightLogical32);
					const auto  bit  = sign != nullptr && Immediate(sign->Arg(1)) == 31u && raw(sign->Arg(0), 0)
					                       ? RecordBit(program, s, atom->Arg(1 - side))
					                       : std::nullopt;
					if (bit && bit->second >= 3u) {
						facing      = a;
						flags_dword = bit->first;
						facing_bit  = bit->second;
						flip        = op == ValueOpcode::INotEqual32;
					}
				}
				if (facing == a) continue;
			}
			const bool compare = op == ValueOpcode::IEqual32 || op == ValueOpcode::INotEqual32 ||
			                     op == ValueOpcode::FPOrdNotEqual32 || op == ValueOpcode::FPUnordNotEqual32;
			if (compare && hit == None && (raw(atom->Arg(0), 0) || raw(atom->Arg(1), 0))) {
				hit = a;
				continue;
			}
			return false;
		}
		if (hit == None || near == None || (k != all && facing == None)) return false;
		const auto set = [](size_t a, uint32_t c) { return a != None && ((c >> a) & 1u) != 0; };
		for (uint32_t c = 0; c < 32; ++c) {
			const bool accept = set(hit, c) && set(near, c) == nearer && (k == all || set(facing, c));
			if ((((*distance >> c) & 1u) != 0) != accept) return false;
		}
	}
	if (both_dword != flags_dword || both_bit + 1u != facing_bit) return false;
	s.layout.flags_dword = flags_dword;
	s.layout.flags_shift = facing_bit - 3u;
	s.layout.flip_facing = flip;
	w.cull               = Value(true);
	w.both_facings       = false;
	return true;
}

bool MatchClosestWaveWalk(Program& program, PacketWalk& w) {
	auto& s = w.shape;
	if (s.wave_triangles == nullptr || !s.blases.empty()) return false;
	const auto first = BlockIndex(program, s.wave_header), last = BlockIndex(program, s.wave_latch);
	const Inst* tri = s.wave_triangles;
	const auto  kind_of = [&](const Inst* value) { return WaveResultKind(program, s, tri, value); };
	if (WaveBlasBase(tri) == nullptr) return false;
	if (s.wave_copy != nullptr) {
		std::vector<Value>   conditions;
		std::array<Value, 2> currents;
		if (!LeafFacingCopies(program, w, conditions, currents)) return false;
		for (auto at = first; at <= last; ++at) {
			for (const auto& inst: *program.blocks[at]) {
				const auto op = inst.GetOpcode();
				if (op == ValueOpcode::ImageSampleRaw || op == ValueOpcode::ImageGatherRaw || op == ValueOpcode::ImageRead)
					return false;
			}
		}
		const auto copy_kind = [&](const Inst* value) { return CopyResultKind(program, s, currents, value); };
		const auto found     = HitResults(program, w, copy_kind, conditions.front(), false, true, conditions);
		if (!found || !(*found)[static_cast<uint32_t>(RayQueryResultKind::Word)]) return false;
		s.layout.node_count_80 = TlasNodeCount80(program, s);
		s.source.layout        = s.layout.Pack();
		return true;
	}
	Value current;
	for (auto at = first; at <= last; ++at) {
		for (const auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != ValueOpcode::FPOrdGreaterThan32 && inst.GetOpcode() != ValueOpcode::FPOrdLessThanEqual32)
				continue;
			const auto* t = Canonical(inst.Arg(0)).TryInstruction();
			if (t == nullptr || kind_of(t) != RayQueryResultKind::T) continue;
			if (!current.IsEmpty() && current != Canonical(inst.Arg(1))) return false;
			current = Canonical(inst.Arg(1));
		}
	}
	if (current.IsEmpty()) return false;
	Value accept;
	bool  negated = false;
	for (auto at = first; at <= last; ++at) {
		for (const auto& inst: *program.blocks[at]) {
			for (uint32_t arm = 1; inst.GetOpcode() == ValueOpcode::SelectU32 && arm <= 2; ++arm) {
				const auto* t = Canonical(inst.Arg(arm)).TryInstruction();
				if (t == nullptr || Canonical(inst.Arg(3 - arm)) != current || kind_of(t) != RayQueryResultKind::T)
					continue;
				if (!accept.IsEmpty() && (accept != inst.Arg(0).Resolve() || negated != (arm == 2))) return false;
				accept  = inst.Arg(0).Resolve();
				negated = arm == 2;
			}
		}
	}
	if (accept.IsEmpty() || !LeafFacing(program, w, accept, negated, tri, current, kind_of, false, &w.leaf)) return false;
	for (auto at = first; at <= last && !w.leaf; ++at) {
		for (const auto& inst: *program.blocks[at]) {
			const auto op = inst.GetOpcode();
			if (op == ValueOpcode::ImageSampleRaw || op == ValueOpcode::ImageGatherRaw || op == ValueOpcode::ImageRead)
				return false;
		}
	}
	const auto found = HitResults(program, w, kind_of, accept, negated, true);
	if (!found || !(*found)[static_cast<uint32_t>(RayQueryResultKind::Word)]) return false;
	s.layout.node_count_80 = TlasNodeCount80(program, s);
	s.source.layout        = s.layout.Pack();
	return true;
}

bool MatchClosestLeafWalk(const Program& program, PacketWalk& w) {
	auto& s = w.shape;
	if (!w.leaf || s.wave_node == nullptr) return false;
	const auto from = [](const Inst& phi, const Block* block) { return IncomingValue(phi, block); };
	w.outer_node    = w.header == s.header ? s.node : LoopPhi(from(*s.node, s.preheader), w.header);
	if (w.outer_node == nullptr || Immediate(from(*w.outer_node, w.preheader)) != s.source.root) return false;
	const auto  popped = from(*w.outer_node, w.latch);
	const auto* pop    = Producer(popped, ValueOpcode::Phi);
	for (size_t arg = 0; pop != nullptr && arg < pop->NumArgs(); ++arg) {
		if (Producer(pop->Arg(arg), ValueOpcode::ReadLane) == nullptr) return false;
	}
	if (pop == nullptr && Producer(popped, ValueOpcode::ReadLane) == nullptr) return false;
	w.packet = nullptr;
	for (auto& inst: *w.header) {
		if (inst.GetOpcode() != ValueOpcode::Phi || inst.GetType() != Type::U1) continue;
		const auto entry = from(inst, w.preheader);
		if (entry.IsImmediate() || !DefinedBefore(program, entry, w.header)) continue;
		if (w.packet != nullptr) return false;
		w.packet = &inst;
	}
	const auto* exit = w.path.back();
	const auto& term = exit->terminator;
	if (w.packet == nullptr || term.kind != CFG::TerminatorKind::ConditionalBranch ||
	    !((term.true_block == w.merge && term.false_block == w.latch) ||
	      (term.false_block == w.merge && term.true_block == w.latch)))
		return false;
	w.leaf_t = {};
	for (auto& inst: *w.header) {
		if (inst.GetOpcode() != ValueOpcode::Phi || from(inst, w.preheader).Resolve() != w.t_max.Resolve()) continue;
		const auto* kept = from(inst, w.latch).Resolve().TryInstruction();
		if (kept == nullptr || kept->Parent() != exit || !w.leaf_t.IsEmpty()) return false;
		w.leaf_t = Value(const_cast<Inst*>(kept));
	}
	if (w.leaf_t.IsEmpty()) return false;
	auto*  list     = EnclosingLoop(program, s.wave_triangles->Parent());
	Block *list_pre = nullptr, *list_latch = nullptr, *list_merge = nullptr;
	if (list == nullptr || list == s.wave_header || !LoopShape(list, list_pre, list_latch, list_merge) ||
	    !InLoop(program, list, s.wave_header, s.wave_latch))
		return false;
	w.list_exits.clear();
	for (auto at = BlockIndex(program, list); at <= BlockIndex(program, list_latch); ++at) {
		auto*       block = program.blocks[at];
		const auto& next  = block->terminator;
		if (next.kind != CFG::TerminatorKind::ConditionalBranch ||
		    (next.true_block != list_latch && next.false_block != list_latch))
			continue;
		auto* other = next.true_block == list_latch ? next.false_block : next.true_block;
		if (InLoop(program, other, list, list_latch)) continue;
		if (BranchCondition(*block).first == nullptr) return false;
		w.list_exits.emplace_back(block, next.true_block == other);
	}
	return !w.list_exits.empty();
}

void RewriteClosestLeafWalk(Program& program, PacketWalk& w, uint32_t site) {
	using Op         = ValueOpcode;
	const auto flags = [site](auto variant) {
		return RayQueryFlags(site, static_cast<uint32_t>(variant));
	};
	auto&      s     = w.shape;
	const auto lanes = IncomingValue(*w.packet, w.preheader).Resolve();
	const auto first = BlockIndex(program, w.header), last = BlockIndex(program, w.latch);
	const auto in_walk = [&](const Block* block) {
		const auto at = BlockIndex(program, block);
		return at >= first && at <= last;
	};
	const auto outside = [&](Inst* inst) {
		auto uses = inst->Uses();
		std::erase_if(uses, [&](const Use& use) { return in_walk(use.user->Parent()); });
		return uses;
	};
	const auto walk_lanes = [&](const Inst* value) {
		std::vector<const Inst*>        pending {value};
		std::unordered_set<const Inst*> seen;
		while (!pending.empty()) {
			const auto* at = pending.back();
			pending.pop_back();
			if (at == w.packet || !seen.insert(at).second) continue;
			const auto op = at->GetOpcode();
			if (!in_walk(at->Parent()) || (op != Op::Phi && op != Op::SelectU1)) return false;
			for (size_t arg = op == Op::Phi ? 0 : 1; arg < (op == Op::Phi ? at->NumArgs() : 3); ++arg) {
				const auto source = at->Arg(arg).Resolve();
				if (source == lanes) continue;
				if (source.TryInstruction() == nullptr) return false;
				pending.push_back(source.TryInstruction());
			}
		}
		return true;
	};
	using Uses = decltype(outside(nullptr));
	std::vector<Uses>                       hit;
	std::vector<std::pair<Inst*, Uses>>     left;
	for (auto& [inst, kind]: w.results) {
		hit.push_back(outside(inst));
	}
	for (auto at = first; at <= last; ++at) {
		for (auto& inst: *program.blocks[at]) {
			if (inst.GetType() != Type::U1) continue;
			auto uses = outside(&inst);
			if (!uses.empty() && walk_lanes(&inst)) left.emplace_back(&inst, std::move(uses));
		}
	}
	const Inserter entry {w.preheader, w.preheader->end()};
	const auto     native =
	    entry(Op::RayQueryInit,
	          {s.ray[0], s.ray[1], s.ray[2], s.ray[3], s.ray[4], s.ray[5], w.t_max, RayMask(s), s.tlas[0], s.tlas[1],
	           Value(0u), lanes, w.cull, Value(false), Value(true)},
	          flags(w.both_facings ? RayQueryInitBothFacings : 0u));
	const auto ballot = entry(Op::Ballot, {native});
	const auto any =
	    entry(Op::INotEqual32,
	          {entry(Op::BitwiseOr32, {entry(Op::CompositeExtractU32x4, {ballot, Value(0u)}),
	                                   entry(Op::CompositeExtractU32x4, {ballot, Value(1u)})}),
	           Value(0u)});
	const auto round = [&](const Inserter& at, Block* from) {
		const auto     next   = at(Op::RayQueryNext, {}, flags(0u));
		const auto     match  = at(Op::RayQueryMatch, {}, flags(0u));
		const auto     node   = Incoming(*w.outer_node, from);
		const auto     packet = Incoming(*w.packet, from);
		const Inserter end {from, from->end()};
		w.outer_node->SetArg(node, end(Op::SelectU32, {any, at(Op::CompositeExtractU32x4, {next, Value(0u)}),
		                                               w.outer_node->Arg(node)}));
		w.packet->SetArg(packet, end(Op::SelectU1, {any, match, w.packet->Arg(packet)}));
		return at(Op::INotEqual32, {at(Op::CompositeExtractU32x4, {next, Value(2u)}), Value(0u)});
	};
	(void)round(entry, w.preheader);
	Block *pre = nullptr, *latch = nullptr, *merge = nullptr;
	LoopShape(s.wave_header, pre, latch, merge);
	const Inserter blas {pre, pre->end()};
	const auto     root = Incoming(*s.wave_node, pre);
	s.wave_node->SetArg(root, blas(Op::SelectU32, {any, blas(Op::RayQuerySelectedLeaf, {}, flags(0u)),
	                                               s.wave_node->Arg(root)}));
	for (const auto& [block, leaves_on_true]: w.list_exits) {
		const auto [ref, negated] = BranchCondition(*block);
		const Inserter in {ref->Parent(), Position(*ref)};
		ref->SetArg(0, in(Op::SelectU1, {any, Value(leaves_on_true != negated), ref->Arg(0)}));
	}
	auto*          exit       = w.path.back();
	const auto [ref, negated] = BranchCondition(*exit);
	const Inserter at {exit, Position(*ref)};
	at(Op::RayQueryMark, {}, flags(RayQueryMarkKind::Resolve));
	const auto has            = round(at, w.latch);
	const bool leaves_on_true = exit->terminator.true_block == w.merge;
	ref->SetArg(0, at(Op::SelectU1, {any, leaves_on_true != negated ? at(Op::LogicalNot, {has}) : has, ref->Arg(0)}));
	const Inserter after {w.merge, AfterPhis(*w.merge)};
	const auto     result = [&](RayQueryResultKind kind) { return after(Op::RayQueryResult, {}, flags(kind)); };
	const auto     commit =
	    after(Op::LogicalAnd, {after(Op::INotEqual32, {result(RayQueryResultKind::Has), Value(0u)}),
	                           after(Op::FPOrdLessThanEqual32, {after(Op::BitCastF32U32, {result(RayQueryResultKind::T)}),
	                                                            after(Op::BitCastF32U32, {w.leaf_t})})});
	for (size_t k = 0; k < hit.size(); ++k) {
		const auto [inst, kind] = w.results[k];
		const auto select       = after(Op::SelectU32, {commit, result(kind), Value(inst)});
		for (const auto& use: hit[k])
			use.user->SetArg(use.operand, select);
	}
	for (auto& [inst, uses]: left) {
		const auto select = after(Op::SelectU1, {any, lanes, Value(inst)});
		for (const auto& use: uses)
			use.user->SetArg(use.operand, select);
	}
	after(Op::RayQueryMark, {}, flags(RayQueryMarkKind::Exit));
}

struct LeafOpacity {
	size_t                        off = 0, opaque = 0, alpha_tested = 0;
	std::pair<uint32_t, uint32_t> flags;
	Value                         off_value;
	bool                          off_negated = false;

	[[nodiscard]] bool Taken(uint32_t c) const {
		const auto atom = [c](size_t k) { return ((c >> k) & 1u) != 0; };
		return atom(off) != off_negated || (atom(opaque) && !atom(alpha_tested));
	}
};

template <typename IsWord>
std::optional<LeafOpacity> DeriveLeafOpacity(const Program& program, const PacketWalk& w,
                                             std::span<const Inst* const> atoms, IsWord&& is_word) {
	using Op = ValueOpcode;
	if (atoms.size() != 3) return std::nullopt;
	const auto& s = w.shape;
	const auto flags_of = [&](Value value) -> std::optional<std::pair<uint32_t, uint32_t>> {
		value             = Canonical(value);
		uint32_t    shift = 0;
		const auto* lsr   = Producer(value, Op::ShiftRightLogical32);
		if (lsr != nullptr) {
			const auto by = Immediate(lsr->Arg(1));
			if (!by || *by >= 29u) return std::nullopt;
			shift = *by;
			value = lsr->Arg(0);
		}
		const auto f = RecordField(program, s, value);
		if (!f || f->second != ~0u) return std::nullopt;
		return std::pair {f->first, shift};
	};
	constexpr size_t                             None = 8;
	LeafOpacity                                  o {None, None, None};
	std::optional<std::pair<uint32_t, uint32_t>> flags;
	for (size_t k = 0; k < atoms.size(); ++k) {
		Value value;
		if (atoms[k]->GetOpcode() == Op::INotEqual32) {
			const auto pair = Operands(Value(const_cast<Inst*>(atoms[k])), Op::INotEqual32,
			                           [](Value side) { return Immediate(side) == 0u; });
			if (pair) value = pair->second;
		}
		Value bit_of;
		if (atoms[k]->GetOpcode() == Op::IEqual32) {
			const auto  pair = Operands(Value(const_cast<Inst*>(atoms[k])), Op::IEqual32,
			                            [](Value side) { return Immediate(side) == 1u; });
			const auto* bfe  = pair ? Producer(Canonical(pair->second), Op::BitFieldUExtract) : nullptr;
			if (bfe != nullptr && Immediate(bfe->Arg(1)) == 0u && Immediate(bfe->Arg(2)) == 1u) bit_of = Canonical(bfe->Arg(0));
			const auto zero = Operands(Value(const_cast<Inst*>(atoms[k])), Op::IEqual32,
			                           [](Value side) { return Immediate(side) == 0u; });
			if (bit_of.IsEmpty() && zero && o.off == None && DefinedBefore(program, zero->second, w.header)) {
				o.off         = k;
				o.off_value   = zero->second.Resolve();
				o.off_negated = true;
				continue;
			}
		}
		if (value.IsEmpty() && bit_of.IsEmpty()) return std::nullopt;
		if (!value.IsEmpty() && o.off == None && DefinedBefore(program, value, w.header)) {
			o.off       = k;
			o.off_value = value.Resolve();
			continue;
		}
		const auto bit = bit_of.IsEmpty() ? Operands(value, Op::BitwiseAnd32, [](Value side) { return Immediate(side) == 1u; })
		                                  : std::optional(std::pair {Value(1u), bit_of});
		const auto any = bit ? Operands(bit->second, Op::BitwiseOr32,
		                                [&](Value side) {
			                                const auto* word = Producer(Canonical(side), Op::ShiftRightLogical32);
			                                return word != nullptr && Immediate(word->Arg(1)) == 30u && is_word(word->Arg(0));
		                                })
		                     : std::nullopt;
		const auto two = !value.IsEmpty() ? Operands(value, Op::BitwiseAnd32, [](Value side) { return Immediate(side) == 2u; })
		                                  : std::nullopt;
		const auto f   = any ? flags_of(any->second) : two ? flags_of(two->second) : std::nullopt;
		if (!f || (flags && *flags != *f)) return std::nullopt;
		flags = f;
		if (any && o.opaque == None) {
			o.opaque = k;
		} else if (!any && o.alpha_tested == None) {
			o.alpha_tested = k;
		} else {
			return std::nullopt;
		}
	}
	if (o.off == None || o.opaque == None || o.alpha_tested == None) return std::nullopt;
	o.flags = *flags;
	return o;
}

bool MatchPacketLaneWalk(Program& program, PacketWalk& w) {
	using Op = ValueOpcode;
	auto& s  = w.shape;
	if (s.blases.size() != 1) return false;
	const auto& blas = s.blases.front();
	const Inst* tri  = nullptr;
	for (auto at = BlockIndex(program, blas.header); at <= BlockIndex(program, blas.latch); ++at) {
		for (const auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != Op::BvhIntersect) continue;
			if (tri != nullptr) return false;
			tri = &inst;
		}
	}
	const auto record = LoadOf(program, Canonical(IncomingValue(*blas.node, blas.preheader)));
	if (tri == nullptr || !record) return false;
	const auto kind_of = [&](const Inst* value) -> std::optional<RayQueryResultKind> {
		if (value == record->low.TryInstruction()) return RayQueryResultKind::InstanceLo;
		if (value == record->high.TryInstruction()) return RayQueryResultKind::InstanceHi;
		if (value->GetOpcode() == Op::CompositeExtractU32x2 && value->Arg(0).Resolve().TryInstruction() == blas.load) {
			const auto word = Immediate(value->Arg(1));
			if (word == 0u) return RayQueryResultKind::Node;
			if (word == 1u) return RayQueryResultKind::Word;
			return std::nullopt;
		}
		return WrittenHitDistance(Value(const_cast<Inst*>(value)), tri);
	};
	const auto of_kind = [&](RayQueryResultKind kind) {
		return [&, kind](Value value) { return SourceKind(program, w, value, kind_of) == kind; };
	};
	LaneAlpha alpha;
	if (!DeriveLaneAlpha(program, w, blas, tri, *record, of_kind(RayQueryResultKind::Node), of_kind(RayQueryResultKind::Word),
	                     alpha))
		return false;
	Value       lanes, direct;
	const auto  accept = FlaggedAccept(program, blas, alpha, RayQueryResultKind::T, kind_of, &lanes, &direct);
	const auto* within = !accept.IsEmpty() && !direct.IsEmpty() ? Producer(lanes, Op::LogicalAnd) : nullptr;
	Value       facing;
	for (uint32_t side = 0; within != nullptr && side < 2 && facing.IsEmpty(); ++side) {
		const auto* near = Producer(within->Arg(side), Op::FPOrdLessThanEqual32);
		const auto* t    = near != nullptr ? Canonical(near->Arg(0)).TryInstruction() : nullptr;
		if (t != nullptr && kind_of(t) == RayQueryResultKind::T &&
		    (Canonical(near->Arg(1)) == Canonical(s.ray[6]) || Canonical(near->Arg(1)) == Canonical(w.t_max)))
			facing = within->Arg(1 - side);
	}
	const auto* facing_lanes = facing.IsEmpty() ? nullptr : Producer(facing, Op::LogicalAnd);
	if (facing_lanes == nullptr) return false;
	const auto nonzero = [](const Inst* atom) -> Value {
		const auto pair = atom->GetOpcode() == Op::INotEqual32
		                      ? Operands(Value(const_cast<Inst*>(atom)), Op::INotEqual32, [](Value side) { return Immediate(side) == 0u; })
		                      : std::nullopt;
		return pair ? pair->second : Value {};
	};
	const auto atom = [](size_t k, uint32_t c) { return ((c >> k) & 1u) != 0; };
	const auto* compare = alpha.flag->Arg(0).Resolve().TryInstruction();
	const auto* tested  = Producer(alpha.flag->Arg(0), Op::LogicalAnd);
	if (tested == nullptr) return false;
	const auto path = tested->Arg(0).Resolve().TryInstruction() == compare ? tested->Arg(1) : tested->Arg(0);
	LaneTable  opacity;
	opacity.exec            = lanes.TryInstruction();
	const auto direct_table = opacity.Condition(direct);
	const auto path_table   = opacity.Condition(path);
	if (!direct_table || !path_table) return false;
	const auto leaf_opacity = DeriveLeafOpacity(program, w, opacity.atoms, of_kind(RayQueryResultKind::Word));
	if (!leaf_opacity) return false;
	for (uint32_t c = 0; c < 8; ++c) {
		const bool taken = leaf_opacity->Taken(c);
		if (atom(c, *direct_table) != taken || atom(c, *path_table) == taken) return false;
	}
	constexpr size_t None      = 8;
	const auto       flags     = std::optional(leaf_opacity->flags);
	const auto       off_value = leaf_opacity->off_value;
	const auto raw = [&](Value value, uint32_t component) {
		const auto* extract = Producer(Canonical(value), Op::CompositeExtractU32x4);
		return extract != nullptr && extract->Arg(0).Resolve().TryInstruction() == tri && Immediate(extract->Arg(1)) == component;
	};
	for (uint32_t side = 0; side < 2; ++side) {
		LaneTable  table;
		table.exec        = facing_lanes->Arg(side).Resolve().TryInstruction();
		const auto result = table.Condition(facing_lanes->Arg(1 - side));
		if (!result || table.atoms.size() != 4) continue;
		size_t                  hit = None, f = None;
		std::array<size_t, 2>   selector {None, None};
		std::array<Value, 2>    uniform {};
		std::array<uint32_t, 2> clear_bit {};
		uint32_t                facing_dword = 0, facing_bit = 0;
		bool                    flip = false, ok = true;
		for (size_t k = 0; k < table.atoms.size() && ok; ++k) {
			const auto* a  = table.atoms[k];
			const auto  op = a->GetOpcode();
			if ((op == Op::IEqual32 || op == Op::INotEqual32) && f == None) {
				for (uint32_t at = 0; at < 2 && f == None; ++at) {
					const auto* sign = Producer(Canonical(a->Arg(at)), Op::ShiftRightLogical32);
					const auto  bit  = sign != nullptr && Immediate(sign->Arg(1)) == 31u && raw(sign->Arg(0), 0)
					                       ? RecordBit(program, s, a->Arg(1 - at))
					                       : std::nullopt;
					if (bit && bit->second >= 3u) {
						f            = k;
						facing_dword = bit->first;
						facing_bit   = bit->second;
						flip         = op == Op::INotEqual32;
					}
				}
				if (f == k) continue;
			}
			const bool test = op == Op::IEqual32 || op == Op::INotEqual32 || op == Op::FPOrdNotEqual32 ||
			                  op == Op::FPUnordNotEqual32;
			if (test && hit == None && (raw(a->Arg(0), 0) || raw(a->Arg(1), 0))) {
				hit = k;
				continue;
			}
			const auto  value = nonzero(a);
			const auto* both  = !value.IsEmpty() ? Producer(Canonical(value), Op::BitwiseAnd32) : nullptr;
			const auto  slot  = selector[0] == None ? 0u : 1u;
			bool        found = false;
			for (uint32_t at = 0; both != nullptr && at < 2 && selector[1] == None && !found; ++at) {
				const auto  u     = both->Arg(at).Resolve();
				const auto* clear = Producer(both->Arg(1 - at), Op::SelectU32);
				const auto  zero  = clear != nullptr && Immediate(clear->Arg(1)) == 1u && Immediate(clear->Arg(2)) == 0u
				                        ? Operands(clear->Arg(0), Op::IEqual32, [](Value v) { return Immediate(v) == 0u; })
				                        : std::nullopt;
				const auto  bit   = zero ? RecordField(program, s, zero->second) : std::nullopt;
				if (!DefinedBefore(program, u, w.header) || !bit || std::popcount(bit->second) != 1) continue;
				selector[slot]  = k;
				uniform[slot]   = u;
				clear_bit[slot] = static_cast<uint32_t>(std::countr_zero(bit->second)) | bit->first << 8u;
				found           = true;
			}
			ok = found;
		}
		if (!ok || hit == None || f == None || selector[1] == None || clear_bit[0] != clear_bit[1] ||
		    clear_bit[0] != ((facing_bit - 1u) | facing_dword << 8u) || facing_dword != flags->first ||
		    facing_bit != flags->second + 3u)
			continue;
		std::array<std::optional<std::pair<Value, bool>>, 2> x;
		for (uint32_t k = 0; k < 2; ++k) {
			const auto* select = Producer(uniform[k], Op::SelectU32);
			const auto* cond   = select != nullptr && Immediate(select->Arg(1)) == 1u && Immediate(select->Arg(2)) == 0u
			                         ? select->Arg(0).Resolve().TryInstruction()
			                         : nullptr;
			if (cond == nullptr || (cond->GetOpcode() != Op::IEqual32 && cond->GetOpcode() != Op::INotEqual32)) continue;
			const auto pair = Operands(Value(const_cast<Inst*>(cond)), cond->GetOpcode(), [](Value v) { return Immediate(v) == 0u; });
			if (pair) x[k] = std::pair {pair->second.Resolve(), cond->GetOpcode() == Op::IEqual32};
		}
		if (!x[0] || !x[1] || x[0]->first != x[1]->first || x[0]->second == x[1]->second) continue;
		uint32_t first = 2;
		for (uint32_t order = 0; order < 2; ++order) {
			bool match = true;
			for (uint32_t c = 0; c < 16 && match; ++c) {
				const bool h = atom(hit, c), fa = atom(f, c), a = atom(selector[order], c), b = atom(selector[1 - order], c);
				match        = atom(c, *result) == (h && (!a || fa) && (!b || !fa));
			}
			if (match) first = order;
		}
		if (first == 2) continue;
		const auto found = HitResults(program, w, kind_of, accept);
		using Kind       = RayQueryResultKind;
		if (!found || !std::ranges::all_of(std::array {Kind::Word, Kind::InstanceLo, Kind::InstanceHi},
		                                   [&](Kind kind) { return (*found)[static_cast<uint32_t>(kind)]; }))
			return false;
		s.layout.flags_dword   = flags->first;
		s.layout.flags_shift   = flags->second;
		s.layout.flip_facing   = flip;
		s.layout.node_count_80 = TlasNodeCount80(program, s);
		s.source.layout        = s.layout.Pack();
		w.both_facings         = false;
		w.cull                 = Value(true);
		w.any_hit              = true;
		w.alpha                = alpha;
		w.alpha_off            = off_value;
		w.flip                 = Producer(uniform[1 - first], Op::SelectU32)->Arg(0).Resolve();
		return true;
	}
	return false;
}

template <typename KindOf, typename IsWord, typename Facing>
bool MatchOpaqueWaveLeaf(Program& program, PacketWalk& w, const BlasWalk& loop, const Inst* tri, KindOf&& kind_of,
                         IsWord&& is_word, Facing&& facing) {
	auto&      s   = w.shape;
	const auto far = [&](const Inst* compare) -> std::optional<bool> {
		const auto op = compare->GetOpcode();
		if (op != ValueOpcode::FPOrdLessThanEqual32 && op != ValueOpcode::FPOrdGreaterThan32) return std::nullopt;
		const auto* t     = Canonical(compare->Arg(0)).TryInstruction();
		const auto  bound = Canonical(compare->Arg(1));
		if (t == nullptr || kind_of(t) != RayQueryResultKind::T || (bound != Canonical(s.ray[6]) && bound != Canonical(w.t_max)))
			return std::nullopt;
		return op == ValueOpcode::FPOrdLessThanEqual32;
	};
	const auto near_in = [&](Value value, const auto& self, uint32_t depth) -> bool {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr || depth > 8) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd)
			return self(inst->Arg(0), self, depth + 1) || self(inst->Arg(1), self, depth + 1);
		return far(inst) == true;
	};
	for (auto at = BlockIndex(program, loop.header); at <= BlockIndex(program, loop.latch); ++at) {
		for (const auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != ValueOpcode::SelectU32) continue;
			const auto* written = Canonical(inst.Arg(1)).TryInstruction();
			const auto* both    = Producer(inst.Arg(0), ValueOpcode::LogicalAnd);
			if (written == nullptr || kind_of(written) != RayQueryResultKind::T || both == nullptr) continue;
			const auto accept = inst.Arg(0).Resolve();
			for (uint32_t side = 0; side < 2; ++side) {
				const auto lanes = both->Arg(side).Resolve();
				if (lanes.TryInstruction() == nullptr || !facing(lanes)) continue;
				LaneTable table;
				table.exec  = lanes.TryInstruction();
				table.flags = true;
				auto taken  = table.Condition(accept);
				const auto at = std::ranges::find_if(table.atoms, [&](const Inst* atom) { return far(atom).has_value(); });
				if (taken && at != table.atoms.end()) {
					const auto k    = static_cast<uint32_t>(at - table.atoms.begin());
					const bool near = *far(*at);
					const auto n    = static_cast<uint32_t>(table.atoms.size());
					uint32_t   rest = 0;
					bool       none = true;
					for (uint32_t c = 0; c < (1u << n); ++c) {
						const bool bit = ((c >> k) & 1u) != 0, value = ((*taken >> c) & 1u) != 0;
						if (bit != near) {
							none &= !value;
						} else if (value) {
							const uint32_t low = c & ((1u << k) - 1u), high = (c >> (k + 1u)) << k;
							for (uint32_t r = low | high; r < 32u; r += 1u << (n - 1u)) rest |= 1u << r;
						}
					}
					if (!none) continue;
					table.atoms.erase(at);
					taken = rest;
				} else if (!near_in(lanes, near_in, 0)) {
					continue;
				}
				if (!taken) continue;
				const auto opacity = DeriveLeafOpacity(program, w, table.atoms, is_word);
				if (!opacity || opacity->flags != std::pair {s.layout.flags_dword, s.layout.flags_shift}) continue;
				bool same = true;
				for (uint32_t c = 0; c < 8; ++c)
					same &= (((*taken >> c) & 1u) != 0) == opacity->Taken(c);
				if (!same) continue;
				const auto found = HitResults(program, w, kind_of, accept);
				using Kind       = RayQueryResultKind;
				if (!found || !std::ranges::all_of(std::array {Kind::Word, Kind::InstanceLo, Kind::InstanceHi},
				                                   [&](Kind kind) { return (*found)[static_cast<uint32_t>(kind)]; }))
					continue;
				s.layout.node_count_80 = TlasNodeCount80(program, s);
				s.source.layout        = s.layout.Pack();
				w.any_hit              = true;
				w.alpha_off            = opacity->off_value;
				return true;
			}
		}
	}
	return false;
}

bool MatchPacketWaveWalk(Program& program, PacketWalk& w) {
	using Op = ValueOpcode;
	auto& s  = w.shape;
	if (!w.walked || s.wave_triangles == nullptr || !s.blases.empty()) return false;
	const Inst* tri     = s.wave_triangles;
	const auto  kind_of = [&](const Inst* value) { return WaveResultKind(program, s, tri, value); };
	const auto  of_kind = [&](RayQueryResultKind kind) {
		return [&, kind](Value value) { return SourceKind(program, w, value, kind_of) == kind; };
	};
	BlasWalk loop;
	loop.header = s.wave_header;
	loop.latch  = s.wave_latch;
	LaneAlpha alpha;
	alpha.scalar = true;
	if (WaveBlasBase(tri) == nullptr) return false;
	const auto facing = [&](Value lanes) {
		for (const bool flags: {false, true}) {
			if (LeafFacing(program, w, lanes, false, tri, Canonical(s.ray[6]), kind_of, flags) ||
			    LeafFacing(program, w, lanes, false, tri, Canonical(w.t_max), kind_of, flags))
				return true;
		}
		return false;
	};
	if (!DeriveLaneAlpha(program, w, loop, tri, s.record, of_kind(RayQueryResultKind::Node), of_kind(RayQueryResultKind::Word),
	                     alpha))
		return MatchOpaqueWaveLeaf(program, w, loop, tri, kind_of, of_kind(RayQueryResultKind::Word), facing);
	Value      lanes;
	const auto accept = FlaggedAccept(program, loop, alpha, RayQueryResultKind::T, kind_of, &lanes);
	if (accept.IsEmpty() || !facing(lanes) || !w.cull.IsImmediate()) return false;
	const Inst* phi  = nullptr;
	const auto* both = Producer(accept, Op::LogicalAnd);
	for (uint32_t side = 0; both != nullptr && side < 2 && phi == nullptr; ++side) {
		const auto pair = Operands(both->Arg(side), Op::INotEqual32, [](Value v) { return Immediate(v) == 0u; });
		if (pair) phi = Producer(pair->second, Op::Phi);
	}
	if (phi == nullptr || phi->NumArgs() != 2) return false;
	const auto written = [&](Value value) {
		for (const Inst* select = nullptr; (select = Producer(value, Op::SelectU32)) != nullptr &&
		                                   select->Arg(0).Resolve() == lanes;)
			value = select->Arg(1).Resolve();
		return value.Resolve();
	};
	const auto alpha_arm = written(phi->Arg(0)).TryInstruction() == alpha.flag ? 0u : 1u;
	const auto value     = written(phi->Arg(1 - alpha_arm));
	auto*      split     = phi->PhiBlock(1 - alpha_arm);
	const auto* tested   = phi->PhiBlock(alpha_arm);
	if (written(phi->Arg(alpha_arm)).TryInstruction() != alpha.flag ||
	    split->terminator.kind != CFG::TerminatorKind::ConditionalBranch || split->terminator.merge_block != phi->Parent())
		return false;
	while (tested != nullptr && tested->ImmPredecessors().size() == 1 && tested->ImmPredecessors()[0] != split)
		tested = tested->ImmPredecessors()[0];
	const auto& t = split->terminator;
	if (tested == nullptr || tested->ImmPredecessors().size() != 1 || (tested != t.true_block && tested != t.false_block) ||
	    (t.true_block != phi->Parent() && t.false_block != phi->Parent()))
		return false;
	const auto [ref, negated] = BranchCondition(*split);
	const auto tests          = ref != nullptr ? Operands(ref->Arg(0), Op::INotEqual32, [](Value v) { return Immediate(v) == 0u; })
	                                           : std::nullopt;
	if (!tests || Canonical(tests->second) != Canonical(value) || (tested == t.true_block) != negated) return false;
	LaneTable table;
	table.exec       = lanes.TryInstruction();
	const auto taken = table.Flag(value);
	if (!taken) return false;
	const auto opacity = DeriveLeafOpacity(program, w, table.atoms, of_kind(RayQueryResultKind::Word));
	if (!opacity || opacity->flags != std::pair {s.layout.flags_dword, s.layout.flags_shift}) return false;
	for (uint32_t c = 0; c < 8; ++c) {
		if ((((*taken >> c) & 1u) != 0) != opacity->Taken(c)) return false;
	}
	const auto found = HitResults(program, w, kind_of, accept);
	using Kind       = RayQueryResultKind;
	if (!found || !std::ranges::all_of(std::array {Kind::Word, Kind::InstanceLo, Kind::InstanceHi},
	                                   [&](Kind kind) { return (*found)[static_cast<uint32_t>(kind)]; }))
		return false;
	s.layout.node_count_80 = TlasNodeCount80(program, s);
	s.source.layout        = s.layout.Pack();
	w.any_hit              = true;
	w.alpha                = alpha;
	w.alpha_off            = opacity->off_value;
	return true;
}

template <typename Record>
bool FlatFacing(const Program& program, const Traversal& t, PacketWalk& w, Value hit, Record&& record) {
	const auto first = BlockIndex(program, t.header), last = BlockIndex(program, t.latch);
	const auto component = [&](Value value) {
		const auto* extract = Producer(Canonical(value), ValueOpcode::CompositeExtractU32x4);
		return extract != nullptr && extract->Arg(0).Resolve().TryInstruction() == t.bvh ? Immediate(extract->Arg(1))
		                                                                                : std::nullopt;
	};
	const auto flags_of = [&](Value value) -> std::optional<std::pair<uint32_t, uint32_t>> {
		const auto* phi = HeaderPhi(t, Canonical(value));
		if (phi == nullptr) return std::nullopt;
		std::unordered_set<const Inst*> values;
		Collect(Back(t, *phi), t.header, values);
		std::optional<std::pair<uint32_t, uint32_t>> found;
		for (const auto* inst: values) {
			if (inst->GetOpcode() == ValueOpcode::Phi || inst->GetOpcode() == ValueOpcode::SelectU32) continue;
			const auto* shift = inst->GetOpcode() == ValueOpcode::ShiftRightLogical32 ? inst : nullptr;
			const auto  by    = shift != nullptr ? Immediate(shift->Arg(1)) : std::optional<uint32_t> {0u};
			auto        field = shift != nullptr ? shift->Arg(0).Resolve() : Value(const_cast<Inst*>(inst));
			if (const auto* kept = Producer(field, ValueOpcode::SelectU32); kept != nullptr) {
				const auto* load = Producer(kept->Arg(1), ValueOpcode::LoadAddressU32);
				if (load == nullptr || load->Arg(3).Resolve() != kept->Arg(0).Resolve()) return std::nullopt;
				field = kept->Arg(1).Resolve();
			}
			const auto dword = record(field);
			if (!by || *by > 28u || !dword || found) return std::nullopt;
			found = std::pair {*dword, *by};
		}
		return found;
	};
	const auto flag_bit = [&](Value value) -> const Inst* {
		const auto* bit = Producer(Canonical(value), ValueOpcode::BitFieldUExtract);
		return bit != nullptr && flags_of(bit->Arg(0)) && Immediate(bit->Arg(1)) && Immediate(bit->Arg(2)) == 1u ? bit
		                                                                                                         : nullptr;
	};
	const Inst* facing = nullptr;
	const Inst* lanes  = nullptr;
	const Inst* flags  = nullptr;
	uint32_t    k      = 0;
	for (auto at = first; at <= last; ++at) {
		for (const auto& inst: *program.blocks[at]) {
			const auto op = inst.GetOpcode();
			for (uint32_t side = 0; (op == ValueOpcode::IEqual32 || op == ValueOpcode::INotEqual32) && side < 2; ++side) {
				const auto* sign = Producer(Canonical(inst.Arg(side)), ValueOpcode::ShiftRightLogical32);
				const auto* bit  = flag_bit(inst.Arg(1 - side));
				const auto* kept = Producer(inst.Arg(side), ValueOpcode::SelectU32);
				if (sign == nullptr || Immediate(sign->Arg(1)) != 31u || component(sign->Arg(0)) != 0u || bit == nullptr ||
				    *Immediate(bit->Arg(1)) < 3u || kept == nullptr)
					continue;
				if (facing != nullptr) return false;
				facing = &inst;
				lanes  = kept->Arg(0).Resolve().TryInstruction();
				flags  = Canonical(bit->Arg(0)).TryInstruction();
				k      = *Immediate(bit->Arg(1));
			}
		}
	}
	if (facing == nullptr) return true;
	LaneTable table;
	table.exec       = lanes;
	const auto value = table.Condition(hit);
	if (!value || lanes == nullptr) return false;
	constexpr size_t None = 8;
	size_t           finite = None, face = None, cull = None, nearer = None;
	Value            culling;
	for (size_t a = 0; a < table.atoms.size(); ++a) {
		const auto* atom = table.atoms[a];
		const auto  op   = atom->GetOpcode();
		if (atom == facing) {
			face = a;
		} else if (op == ValueOpcode::FPCmpClass32 && Immediate(atom->Arg(1)) == 0x38u && finite == None) {
			const auto* negated = Producer(Canonical(atom->Arg(0)), ValueOpcode::FPNeg32);
			const auto* size    = negated != nullptr ? Producer(Canonical(negated->Arg(0)), ValueOpcode::FPAbs32) : nullptr;
			if (size == nullptr || component(size->Arg(0)) != 0u) return false;
			finite = a;
		} else if (op == ValueOpcode::FPOrdLessThanEqual32 && t.extent != nullptr && nearer == None) {
			const auto* far = Canonical(atom->Arg(0)).TryInstruction();
			if (far == nullptr || WrittenHitDistance(Value(const_cast<Inst*>(far)), t.bvh) != RayQueryResultKind::T ||
			    Canonical(atom->Arg(1)).TryInstruction() != t.extent)
				return false;
			nearer = a;
		} else if (op == ValueOpcode::INotEqual32 && cull == None) {
			const auto* both = Immediate(atom->Arg(1)) == 0u ? Producer(atom->Arg(0), ValueOpcode::BitwiseAnd32) : nullptr;
			for (uint32_t side = 0; both != nullptr && side < 2 && culling.IsEmpty(); ++side) {
				const auto* r      = Producer(both->Arg(side), ValueOpcode::SelectU32);
				const auto  kept   = ResolveActiveU32(both->Arg(1 - side), Value(const_cast<Inst*>(lanes)));
				const auto* select = Producer(kept, ValueOpcode::SelectU32);
				if (r == nullptr || select == nullptr || Immediate(r->Arg(1)) != 1u || Immediate(r->Arg(2)) != 0u ||
				    !DefinedBefore(program, r->Arg(0), t.header) || Immediate(select->Arg(1)) != 0u ||
				    Immediate(select->Arg(2)) != 1u)
					continue;
				LaneTable  inner;
				inner.exec       = lanes;
				const auto taken = inner.Condition(select->Arg(0));
				const auto* test = inner.atoms.size() == 1 ? inner.atoms[0] : nullptr;
				const auto* bits = test != nullptr && test->GetOpcode() == ValueOpcode::INotEqual32 && Immediate(test->Arg(1)) == 0u
				                       ? Producer(test->Arg(0), ValueOpcode::BitwiseAnd32)
				                       : nullptr;
				for (uint32_t b = 0; bits != nullptr && b < 2; ++b) {
					if (taken == LaneTable::Tables[0] && Immediate(bits->Arg(b)) == 1u << (k - 1u) &&
					    Canonical(bits->Arg(1 - b)).TryInstruction() == flags)
						culling = r->Arg(0).Resolve();
				}
			}
			if (culling.IsEmpty()) return false;
			cull = a;
		} else {
			return false;
		}
	}
	if (finite == None || face == None || cull == None) return false;
	const auto set = [](size_t a, uint32_t c) { return a != None && ((c >> a) & 1u) != 0; };
	for (uint32_t c = 0; c < 32; ++c) {
		const bool accept = set(finite, c) && (!set(cull, c) || set(face, c)) && (nearer == None || set(nearer, c));
		if ((((*value >> c) & 1u) != 0) != accept) return false;
	}
	const auto field = flags_of(Value(const_cast<Inst*>(flags)));
	if (!field || field->second + k - 3u > 31u) return false;
	auto& layout       = w.shape.layout;
	layout.flags_dword = field->first;
	layout.flags_shift = field->second + k - 3u;
	layout.flip_facing = facing->GetOpcode() == ValueOpcode::INotEqual32;
	w.cull             = culling;
	w.both_facings     = false;
	return true;
}

bool MatchFlatWalk(Program& program, Inst& bvh, PacketWalk& w) {
	Traversal t;
	if (!Match(program, bvh, t, false)) return false;
	auto& s = w.shape;
	if (!LoopShape(t.header, w.preheader, w.latch, w.merge) || w.preheader != t.preheader || w.latch != t.latch)
		return false;
	w.header = t.header;
	const auto lanes = Entry(t, *t.exec);
	const auto root  = Immediate(ResolveActiveU32(Entry(t, *t.state[0]), lanes));
	if (!root) return false;
	for (uint32_t k = 0; k < 7; ++k) {
		s.ray[k] = t.ray[k];
	}
	s.tlas = {t.words[0].Resolve(), t.words[1].Resolve()};
	auto&       layout = s.layout;
	const auto* sum    = Producer(Canonical(t.root->Arg(1)), ValueOpcode::IAdd32);
	bool        found  = false;
	for (uint32_t side = 0; sum != nullptr && side < 2 && !found; ++side) {
		const auto* scale = Producer(Canonical(sum->Arg(side)), ValueOpcode::ShiftLeftLogical32);
		const auto* index = scale != nullptr ? Producer(Canonical(scale->Arg(0)), ValueOpcode::ShiftRightLogical32) : nullptr;
		const auto  bits  = scale != nullptr ? Immediate(scale->Arg(1)) : std::nullopt;
		const auto* array = Producer(Canonical(sum->Arg(1 - side)), ValueOpcode::IAdd32);
		if (index == nullptr || Immediate(index->Arg(1)) != 3u || !bits || *bits < 7u || *bits > 8u || array == nullptr)
			continue;
		for (uint32_t b = 0; b < 2 && !found; ++b) {
			const auto offset = LoadOf(program, array->Arg(b));
			if (offset && array->Arg(1 - b).Resolve() == s.tlas[0] && offset->low == s.tlas[0] &&
			    offset->high == s.tlas[1]) {
				layout.instance_offset = offset->offset;
				layout.record_stride   = 1u << *bits;
				found                  = true;
			}
		}
	}
	if (!found) return false;
	const auto record = [&](Value value) -> std::optional<uint32_t> {
		const auto* load = Producer(value, ValueOpcode::LoadAddressU32);
		if (load == nullptr || load->Arg(1) != t.root->Arg(1) || load->Arg(2) != t.root->Arg(2)) return std::nullopt;
		const auto& memory = MemoryOf(program, *load);
		if (memory.kind != ResourceKind::Global || !memory.address_is_full || (memory.offset & 3u) != 0) return std::nullopt;
		return memory.offset / 4u;
	};
	const auto* address = Producer(bvh.Arg(1), ValueOpcode::MakeImageAddress);
	std::array<Value, 6> world {};
	for (uint32_t k = 0; k < 6; ++k) {
		world[k] = Term(s.ray[k]);
	}
	const auto object = [&](uint32_t arg, bool origin, LinearForm& out) {
		auto* phi = HeaderPhi(t, address->Arg(arg));
		if (phi == nullptr) return false;
		std::unordered_set<const Inst*> values;
		Collect(Back(t, *phi), t.header, values);
		bool any = false;
		for (const auto* value: values) {
			if (value->GetOpcode() == ValueOpcode::Phi || value->GetOpcode() == ValueOpcode::SelectU32) continue;
			LinearForm form;
			form.fill(NoDword);
			if (!Expand(Value(const_cast<Inst*>(value)), world, record, form)) continue;
			bool complete = (form[3] != NoDword) == origin;
			for (uint32_t c = 0; c < 3; ++c) {
				complete = complete && (form[c] != NoDword) == origin && (form[4 + c] != NoDword) != origin;
			}
			if (!complete) continue;
			if (any && form != out) return false;
			out = form;
			any = true;
		}
		return any;
	};
	for (uint32_t row = 0; row < 3; ++row) {
		LinearForm origin, direction;
		if (!object(3 + row, true, origin) || !object(6 + row, false, direction)) return false;
		for (uint32_t c = 0; c < 3; ++c) {
			if (direction[4 + c] != origin[c]) return false;
			layout.transform[row * 4 + c] = static_cast<uint8_t>(origin[c]);
		}
		layout.transform[row * 4 + 3] = static_cast<uint8_t>(origin[3]);
	}
	const auto root_dword = record(Value(const_cast<Inst*>(t.root)));
	const auto blas_dword = record(Value(const_cast<Inst*>(t.blas)));
	const auto mask_dword = record(Value(const_cast<Inst*>(t.loads[2])));
	if (!root_dword || !blas_dword || !mask_dword) return false;
	layout.root_dword  = *root_dword;
	layout.blas_dword  = *blas_dword;
	layout.blas_packed = t.packed;
	layout.mask_dword  = *mask_dword;
	layout.mask_shift  = t.mask_shift;
	layout.mask_width  = 0;
	layout.flags_dword = NoField;
	layout.inline_type = NoField;
	layout.node_count_80 = !t.packed;
	layout.valid         = true;
	s.cull_mask          = t.cull_mask;
	s.masked             = true;
	if (!DeriveHeaderSource(program, s)) return false;
	s.source.root   = *root;
	s.source.layout = layout.Pack();
	if (!WalkReach(program, w.header, w.latch, w.merge->start_pc, s.source)) return false;
	Value hit;
	for (auto at = BlockIndex(program, w.header); at <= BlockIndex(program, w.latch); ++at) {
		for (auto& inst: *program.blocks[at]) {
			if (inst.GetOpcode() != ValueOpcode::SelectU32 || inst.Arg(1).Resolve().TryInstruction() != t.state[1]) continue;
			if (!hit.IsEmpty() && hit != inst.Arg(0).Resolve()) return false;
			hit = inst.Arg(0).Resolve();
		}
	}
	if (hit.IsEmpty() || !FlatFacing(program, t, w, hit, record)) return false;
	s.source.layout = layout.Pack();
	const auto kind_of = [&](const Inst* value) -> std::optional<RayQueryResultKind> {
		if (value == t.state[0]) return RayQueryResultKind::Node;
		if (value == t.state[1]) return RayQueryResultKind::Word;
		if (value != nullptr && value == t.state[2]) return RayQueryResultKind::InstanceLo;
		if (value != nullptr && value == t.state[3]) return RayQueryResultKind::InstanceHi;
		return WrittenHitDistance(Value(const_cast<Inst*>(value)), &bvh);
	};
	w.results.clear();
	const auto results = HitResults(program, w, kind_of, hit);
	if (!results || w.results.empty()) return false;
	w.lanes     = lanes;
	w.t_max     = t.ray[6];
	w.any_hit   = t.extent == nullptr;
	w.alpha_off = Value(1u);
	w.lane_phis = t.lanes;
	return true;
}

void RewriteHitWalk(Program& program, PacketWalk& w, uint32_t site, bool entry_alpha) {
	using Op         = ValueOpcode;
	const auto flags = [site](auto variant) {
		return RayQueryFlags(site, static_cast<uint32_t>(variant));
	};
	const auto& s = w.shape;
	const Inserter entry {w.preheader, w.preheader->end()};
	const auto&    lane = w.alpha;
	if (lane.valid) {
		const auto reference =
		    lane.factor_always
		        ? entry(Op::BitCastU32F32, {entry(Op::FPMul32, {Value::F32(std::bit_cast<float>(lane.reference.U32())),
		                                                        entry(Op::BitCastF32U32, {lane.factor})})})
		        : lane.reference;
		const auto before = [&](Value value) { return DescriptorBefore(program, value, w.header, entry); };
		entry(Op::RayQueryAlpha,
		      {lane.lod, lane.spread, lane.factor.IsEmpty() || lane.factor_always ? Value(One) : lane.factor,
		       before(lane.material[0]), before(lane.material[1]), before(lane.material[2]), before(lane.material[3]),
		       before(lane.buffers[0]), before(lane.buffers[1]), before(lane.buffers[2]), before(lane.buffers[3]),
		       reference, Value(lane.lod_offset), Value(lane.fields.Pack(s.layout))},
		      flags(lane.scalar ? RayQueryAlphaScalar : RayQueryAlphaLane | (lane.factor.IsEmpty() ? 0u : RayQueryAlphaFactor)));
	}
	if (entry_alpha) {
		RayTracingAlphaFields fields;
		fields.material_dword = 0x70 / 4;
		entry(Op::RayQueryAlpha,
		      {Value(0u), Value(0u), Value(One), w.entries[0], w.entries[1], w.entries[2], w.entries[3],
		       w.buffers[0], w.buffers[1], w.buffers[2], w.buffers[3], Value(std::bit_cast<uint32_t>(0.5f)),
		       Value(std::bit_cast<uint32_t>(2.0f)), Value(fields.Pack(s.layout))},
		      flags(RayQueryAlphaEntry));
	}
	const auto nonzero = [&](Value value, bool otherwise) {
		return value.IsEmpty() ? Value(otherwise) : entry(Op::INotEqual32, {value, Value(0u)});
	};
	const auto alpha_off = w.any_hit ? w.alpha_off : Value(entry_alpha || lane.valid ? 0u : 1u);
	const auto native =
	    entry(Op::RayQueryInit,
	          {s.ray[0], s.ray[1], s.ray[2], s.ray[3], s.ray[4], s.ray[5], w.t_max, RayMask(s), s.tlas[0], s.tlas[1], alpha_off,
	           w.lanes, w.cull, w.flip.IsEmpty() ? Value(false) : w.flip, nonzero(lane.level, true)},
	          flags((w.both_facings ? RayQueryInitBothFacings : 0u) | (entry_alpha || w.any_hit ? RayQueryInitNoLeaf : 0u) |
	                (lane.valid && !w.any_hit ? RayQueryInitGeometryOpacity : 0u) |
	                (w.any_hit && !lane.valid ? RayQueryInitOpaqueOnly : 0u)));
	const auto ballot = entry(Op::Ballot, {native});
	const auto any =
	    entry(Op::INotEqual32,
	          {entry(Op::BitwiseOr32, {entry(Op::CompositeExtractU32x4, {ballot, Value(0u)}),
	                                   entry(Op::CompositeExtractU32x4, {ballot, Value(1u)})}),
	           Value(0u)});
	entry(Op::RayQueryNext, {}, flags(0u));
	if (!w.lane_phis.empty()) {
		const auto guest = entry(Op::LogicalAnd, {w.lanes, entry(Op::LogicalNot, {native})});
		for (auto* phi: w.lane_phis) {
			phi->SetArg(Incoming(*phi, w.preheader), guest);
		}
	}
	for (size_t at = 0; at < w.path.size(); ++at) {
		auto* block = w.path[at];
		if (block->terminator.kind != CFG::TerminatorKind::ConditionalBranch) continue;
		const auto [ref, negated] = BranchCondition(*block);
		const auto*    next       = at + 1 < w.path.size() ? w.path[at + 1] : w.merge;
		const Inserter in {ref->Parent(), Position(*ref)};
		ref->SetArg(0, in(Op::SelectU1, {any, Value((next == block->terminator.true_block) != negated), ref->Arg(0)}));
	}
	const Inserter after {w.merge, AfterPhis(*w.merge)};
	const auto     result = [&](RayQueryResultKind kind) { return after(Op::RayQueryResult, {}, flags(kind)); };
	const auto     has    = after(Op::INotEqual32, {result(RayQueryResultKind::Has), Value(0u)});
	const auto     commit = w.any_hit ? has
	                                  : after(Op::LogicalAnd, {has, after(Op::FPOrdLessThanEqual32,
	                                                                      {after(Op::BitCastF32U32, {result(RayQueryResultKind::T)}),
	                                                                       after(Op::BitCastF32U32, {w.t_max})})});
	for (auto& [inst, kind]: w.results) {
		const auto value  = result(kind);
		const auto select = after(Op::SelectU32, {commit, value, Value(inst)});
		const auto uses   = inst->Uses();
		for (const auto& use: uses) {
			if (use.user != select.Instruction() && !InLoop(program, use.user->Parent(), w.header, w.latch))
				use.user->SetArg(use.operand, select);
		}
	}
	after(Op::RayQueryMark, {}, flags(RayQueryMarkKind::Exit));
}

bool MatchClosestLaneWalk(Program& program, const NativeTraversalOptions& options, PacketWalk& w) {
	auto& s = w.shape;
	if (options.wave_size != 32 || w.header == s.header || !s.masked || s.blases.empty()) return false;
	const auto from = [](const Inst& phi, const Block* block) { return IncomingValue(phi, block); };
	w.outer_node = LoopPhi(from(*s.node, s.preheader), w.header);
	if (w.outer_node == nullptr || Immediate(from(*w.outer_node, w.preheader)) != s.source.root)
		return false;
	const auto* pop = Producer(from(*w.outer_node, w.latch), ValueOpcode::Phi);
	if (pop == nullptr) return false;
	for (size_t arg = 0; arg < pop->NumArgs(); ++arg) {
		if (Producer(pop->Arg(arg), ValueOpcode::ReadLane) == nullptr) return false;
	}
	const auto* exit = w.path.back();
	if (exit != s.merge || exit->terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
	    (exit->terminator.true_block != w.merge && exit->terminator.false_block != w.merge) ||
	    BlockIndex(program, pop->Parent()) <= BlockIndex(program, exit))
		return false;
	for (auto& inst: *s.header) {
		if (inst.GetOpcode() != ValueOpcode::Phi || inst.GetType() != Type::U32 || &inst == s.node)
			continue;
		std::unordered_set<const Inst*> values;
		Collect(from(inst, s.latch), s.header, values);
		if (std::ranges::none_of(values, [&](const Inst* value) {
			    return value->Parent() == s.bvh->Parent() &&
			           value->GetOpcode() == ValueOpcode::CompositeExtractU32x4 &&
			           Immediate(value->Arg(1)) == 0u && Producer(value->Arg(0), ValueOpcode::Ballot) != nullptr;
		    }))
			continue;
		if (w.packet != nullptr) return false;
		w.packet = &inst;
	}
	if (w.packet == nullptr || !DefinedBefore(program, from(*w.packet, s.preheader), w.header))
		return false;
	const auto first = BlockIndex(program, s.header), last = BlockIndex(program, s.latch);
	const auto successors = [](const Block* block) -> std::vector<Block*> {
		const auto& term  = block->terminator;
		const auto  taken = block->condition;
		if (term.kind == CFG::TerminatorKind::ConditionalBranch && taken.IsImmediate() &&
		    taken.GetType() == Type::U1)
			return {taken.U1() ? term.true_block : term.false_block};
		const auto next = block->ImmSuccessors();
		return {next.begin(), next.end()};
	};
	const auto forward = [&](Block* start, std::vector<const Block*>& seen) {
		bool                ends = false;
		std::vector<Block*> queue {start};
		seen = {start};
		for (size_t at = 0; at < queue.size(); ++at) {
			for (auto* next: successors(queue[at])) {
				const auto index = BlockIndex(program, next);
				if (next == exit) {
					ends = true;
					continue;
				}
				if (index <= BlockIndex(program, queue[at]) || index < first || index > last ||
				    std::ranges::find(seen, next) != seen.end())
					continue;
				seen.push_back(next);
				queue.push_back(next);
			}
		}
		return ends;
	};
	const auto& blas = s.blases.front();
	Block*      side = nullptr;
	for (const auto* block: w.path) {
		const auto index = BlockIndex(program, block);
		if (index < first || index > last) continue;
		for (auto* next: block->ImmSuccessors()) {
			std::vector<const Block*> seen;
			if (std::ranges::find(w.path, next) != w.path.end()) continue;
			forward(next, seen);
			if (std::ranges::find(seen, blas.header) == seen.end()) continue;
			if (side != nullptr && side != next) return false;
			side = next;
		}
	}
	std::vector<const Block*> instance;
	return side != nullptr && forward(side, instance) &&
	       std::ranges::find(instance, s.latch) == instance.end();
}

void RewriteClosestLaneWalk(Program& program, PacketWalk& w, uint32_t site) {
	using Op             = ValueOpcode;
	const auto flags     = [site](auto variant) {
		return RayQueryFlags(site, static_cast<uint32_t>(variant));
	};
	const auto lane_leaf = flags(RayQueryKeyLaneLeaf);
	auto&      s         = w.shape;
	const Inserter entry {w.preheader, w.preheader->end()};
	const auto     native =
	    entry(Op::RayQueryInit,
	          {s.ray[0], s.ray[1], s.ray[2], s.ray[3], s.ray[4], s.ray[5], w.t_max, RayMask(s),
	           s.tlas[0], s.tlas[1], Value(0u), w.lanes, Value(true), Value(false), Value(true)},
	          flags(RayQueryInitBothFacings | RayQueryInitAllCandidates));
	const auto ballot = entry(Op::Ballot, {native});
	const auto any =
	    entry(Op::INotEqual32,
	          {entry(Op::BitwiseOr32, {entry(Op::CompositeExtractU32x4, {ballot, Value(0u)}),
	                                   entry(Op::CompositeExtractU32x4, {ballot, Value(1u)})}),
	           Value(0u)});
	const auto first = entry(Op::RayQueryNext, {}, lane_leaf);
	const auto start = Incoming(*w.outer_node, w.preheader);
	w.outer_node->SetArg(start, entry(Op::SelectU32, {any, entry(Op::CompositeExtractU32x4, {first, Value(0u)}),
	                                                 w.outer_node->Arg(start)}));
	const Inserter round {s.preheader, s.preheader->end()};
	const auto     match  = round(Op::RayQueryMatch, {}, lane_leaf);
	const auto     lanes  = round(Op::CompositeExtractU32x4, {round(Op::Ballot, {match}), Value(0u)});
	const auto     packet = Incoming(*w.packet, s.preheader);
	w.packet->SetArg(packet, round(Op::SelectU32, {any, lanes, w.packet->Arg(packet)}));
	auto*          exit           = w.path.back();
	const auto [ref, negated]     = BranchCondition(*exit);
	const Inserter at {exit, Position(*ref)};
	at(Op::RayQueryMark, {}, flags(RayQueryMarkKind::Resolve));
	const auto next     = at(Op::RayQueryNext, {}, lane_leaf);
	const auto instance = at(Op::CompositeExtractU32x4, {next, Value(0u)});
	const auto has = at(Op::INotEqual32, {at(Op::CompositeExtractU32x4, {next, Value(2u)}), Value(0u)});
	const bool leaves_on_true = exit->terminator.true_block == w.merge;
	ref->SetArg(0, at(Op::SelectU1, {any, leaves_on_true != negated ? at(Op::LogicalNot, {has}) : has,
	                                 ref->Arg(0)}));
	const Inserter back {w.latch, AfterPhis(*w.latch)};
	const auto     index = Incoming(*w.outer_node, w.latch);
	w.outer_node->SetArg(index, back(Op::SelectU32, {any, instance, w.outer_node->Arg(index)}));
	const Inserter leave {w.merge, AfterPhis(*w.merge)};
	leave(Op::RayQueryMark, {}, flags(RayQueryMarkKind::Exit));
	RewriteBlasWalks(program, s.blases, any, lane_leaf);
}

}

NativeTraversals TraceNativeTraversals(Program& program, const NativeTraversalOptions& options) {
	NativeTraversals result;
	if (program.stage != ShaderType::Compute) return result;
	std::vector<Inst*> intersections;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() == ValueOpcode::BvhIntersect) intersections.push_back(&inst);
		}
	}
	const auto mark = [](Inst* bvh) {
		bvh->SetFlags(bvh->Flags<uint32_t>() | BvhIntersectNative);
	};
	enum class Kind { Entry, ClosestLane, ClosestOpaque, ClosestLeaf, PacketLane, Flat, LaneMask };
	struct Walk {
		Kind         kind = Kind::LaneMask;
		PacketWalk   packet;
		LaneMaskWalk lane;
		Block*       header = nullptr;
		Block*       latch  = nullptr;

		LaneMaskWalk& Shape() { return kind == Kind::LaneMask ? lane : packet.shape; }
	};
	std::vector<Walk> walks;
	const auto        inside = [&](const Block* block) {
		return std::ranges::any_of(walks, [&](const Walk& walk) {
			return InLoop(program, block, walk.header, walk.latch);
		});
	};
	for (auto* bvh: intersections) {
		if (inside(bvh->Parent())) continue;
		Walk walk;
		bool found = false;
		if (MatchPacketWalk(program, *bvh, walk.packet)) {
			walk.header = walk.packet.header;
			walk.latch  = walk.packet.latch;
			if (MatchEntryWalk(program, options, walk.packet)) {
				walk.kind = Kind::Entry;
				found     = true;
			} else if (MatchClosestLaneWalk(program, options, walk.packet)) {
				walk.kind = Kind::ClosestLane;
				found     = true;
			} else if (MatchClosestOpaqueWalk(program, walk.packet) || MatchClosestWaveWalk(program, walk.packet)) {
				walk.kind = walk.packet.leaf ? Kind::ClosestLeaf : Kind::ClosestOpaque;
				found     = !walk.packet.leaf || MatchClosestLeafWalk(program, walk.packet);
			} else if (MatchPacketLaneWalk(program, walk.packet) || MatchPacketWaveWalk(program, walk.packet)) {
				walk.kind = Kind::PacketLane;
				found     = true;
			}
		}
		if (!found && MatchFlatWalk(program, *bvh, walk.packet = {})) {
			walk.kind   = Kind::Flat;
			walk.header = walk.packet.header;
			walk.latch  = walk.packet.latch;
			found       = true;
		}
		if (!found && options.wave_size == 32 && MatchLaneMaskWalk(program, *bvh, walk.lane)) {
			walk.kind   = Kind::LaneMask;
			walk.header = walk.lane.header;
			walk.latch  = walk.lane.latch;
			found       = true;
		}
		if (found) walks.push_back(std::move(walk));
	}
	if (walks.empty()) return result;
	auto reference = walks.front().Shape().source;
	auto lowest    = walks.front().Shape().layout.mask_shift;
	for (auto& walk: walks) {
		if (walk.Shape().layout.mask_shift < lowest) {
			lowest    = walk.Shape().layout.mask_shift;
			reference = walk.Shape().source;
		}
	}
	const auto shares = [&](LaneMaskWalk& walk) {
		const auto& source = walk.source;
		if (source.user_sgpr != reference.user_sgpr || source.header_offset != reference.header_offset ||
		    source.header_vsharp != reference.header_vsharp || source.root != reference.root) {
			return false;
		}
		if (source.layout == reference.layout) return true;
		const auto     packed = reference.layout[2];
		const uint32_t shift  = packed >> 4u & 31u;
		auto           layout = walk.layout;
		if ((packed >> 19u & 1u) == 0 || (packed >> 25u & 31u) != 0 || layout.mask_width != 0 || !walk.mask_value.IsEmpty() ||
		    layout.mask_shift <= shift || (walk.cull_mask << (layout.mask_shift - shift)) > 0xffu) {
			return false;
		}
		const auto moved  = layout.mask_shift - shift;
		layout.mask_shift = shift;
		if (layout.Pack() != reference.layout) return false;
		walk.cull_mask <<= moved;
		walk.layout        = layout;
		walk.source.layout = reference.layout;
		return true;
	};
	std::erase_if(walks, [&](Walk& walk) { return !shares(walk.Shape()); });
	if (walks.empty()) return result;
	result       = walks.front().Shape().source;
	result.reach = {};
	const auto sites   = options.rewrite ? std::min<size_t>(walks.size(), MaxRayQuerySites) : 0;
	const auto covered = sites == walks.size() && std::ranges::all_of(intersections, [&](const Inst* other) {
		                     return inside(other->Parent());
	                     });
	for (auto& walk: walks) {
		if (covered) {
			const auto& reach = walk.Shape().source.reach;
			result.reach.insert(result.reach.end(), reach.begin(), reach.end());
		}
	}
	for (size_t at = 0; at < sites; ++at) {
		auto&      walk = walks[at];
		const auto site = static_cast<uint32_t>(at);
		switch (walk.kind) {
			case Kind::Entry: RewriteHitWalk(program, walk.packet, site, true); break;
			case Kind::ClosestOpaque:
			case Kind::PacketLane:
			case Kind::Flat: RewriteHitWalk(program, walk.packet, site, false); break;
			case Kind::ClosestLane: RewriteClosestLaneWalk(program, walk.packet, site); break;
			case Kind::ClosestLeaf: RewriteClosestLeafWalk(program, walk.packet, site); break;
			case Kind::LaneMask: RewriteLaneMaskWalk(program, walk.lane, site); break;
		}
		++result.sites;
		for (auto* bvh: intersections) {
			if (InLoop(program, bvh->Parent(), walk.header, walk.latch)) mark(bvh);
		}
	}
	return result;
}

}
