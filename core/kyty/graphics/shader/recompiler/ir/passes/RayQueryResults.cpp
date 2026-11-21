#include "graphics/shader/recompiler/ir/passes/RayQueryResults.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryOracle.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

using Op = ValueOpcode;

constexpr uint32_t NoSlot     = 0xffffffffu;
constexpr uint32_t Miss       = 0xffffffffu;
constexpr uint32_t InfBits    = 0x7f800000u;
constexpr uint32_t OneBits    = 0x3f800000u;
constexpr uint32_t MaxVisits  = 6000;
constexpr uint32_t MaxSeeds   = 12;
constexpr uint32_t MaxRows    = 1024;
constexpr uint64_t PrologueWork = 60000;
constexpr uint64_t WalkWork     = 150000;
constexpr uint64_t TotalWork    = 3000000;
constexpr size_t   TaintBits  = 320;
constexpr uint32_t TaintLive  = 0;
constexpr uint32_t TaintLives = 240;
constexpr uint32_t TaintRecord = 240;
constexpr uint32_t TaintList   = 288;
constexpr uint32_t TaintMemory = 290;
constexpr uint32_t TaintOracle = 291;

using Taint = std::bitset<TaintBits>;

uint64_t Mix(uint64_t x) {
	x += 0x9e3779b97f4a7c15ull;
	x = (x ^ (x >> 30u)) * 0xbf58476d1ce4e5b9ull;
	x = (x ^ (x >> 27u)) * 0x94d049bb133111ebull;
	return x ^ (x >> 31u);
}

float F(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

uint64_t B(float value) {
	return std::bit_cast<uint32_t>(value);
}

uint32_t Dword(uint32_t low, uint32_t high) {
	return (low & 15u) | ((high & 3u) << 4u);
}

struct Layout {
	uint32_t                 instance_offset = 0;
	uint32_t                 stride          = 128;
	uint32_t                 blas            = 0;
	bool                     packed          = false;
	uint32_t                 root            = 0;
	bool                     mask            = false;
	uint32_t                 mask_dword      = 0;
	uint32_t                 mask_shift      = 0;
	uint32_t                 mask_width      = 0;
	bool                     flags           = false;
	uint32_t                 flags_dword     = 0;
	uint32_t                 flags_shift     = 0;
	bool                     flip            = false;
	bool                     inline_record   = false;
	uint32_t                 inline_type     = 0;
	uint32_t                 inline_offset   = 0;
	uint32_t                 inline_root     = 0xff;
	std::array<uint32_t, 12> transform {};

	explicit Layout(const std::array<uint32_t, 4>& w) {
		instance_offset = ((w[1] >> 16u) & 15u) * 8u;
		stride          = 128u + ((w[1] >> 30u) & 3u) * 32u;
		blas            = Dword(w[1] >> 20u, w[3] >> 20u);
		packed          = ((w[1] >> 24u) & 1u) != 0;
		root            = Dword(w[1] >> 25u, w[3] >> 22u);
		mask            = ((w[2] >> 19u) & 1u) != 0;
		mask_dword      = Dword(w[2], w[3] >> 24u);
		mask_shift      = (w[2] >> 4u) & 31u;
		mask_width      = (w[2] >> 25u) & 31u;
		flags           = ((w[2] >> 13u) & 1u) != 0;
		flags_dword     = Dword(w[2] >> 9u, w[3] >> 26u);
		flags_shift     = (w[2] >> 14u) & 31u;
		flip            = ((w[1] >> 29u) & 1u) != 0;
		inline_record   = ((w[3] >> 19u) & 1u) != 0;
		inline_type     = (w[3] >> 16u) & 7u;
		inline_offset   = (w[3] & 0xffu) * 8u + ((w[3] >> 28u) & 1u) * 4u;
		inline_root     = (w[3] >> 8u) & 0xffu;
		const uint32_t base = (w[2] >> 20u) & 31u;
		for (uint32_t i = 0; i < 12; ++i) transform[i] = base + ((w[i / 8u] >> ((i % 8u) * 4u)) & 15u);
	}

	[[nodiscard]] bool Inline(uint32_t node) const { return inline_record && (node & 7u) == inline_type; }
	[[nodiscard]] uint32_t Words() const { return stride / 4u; }
};

class Machine {
public:
	struct Call {
		uint32_t                lane   = 0;
		uint64_t                base   = 0;
		uint64_t                node   = 0;
		uint32_t                extent = 0;
		std::array<uint32_t, 4> descriptor {};
	};
	using Oracle = std::function<bool(const Inst&, std::span<const Call>, std::span<std::array<uint32_t, 4>>)>;

	Machine(const Program& program, uint32_t wave, const Block* header, const Block* latch)
	    : m_program(program), m_wave(wave) {
		for (size_t index = 0; index < program.blocks.size(); ++index) m_block_index.emplace(program.blocks[index], index);
		m_first = m_block_index.at(header);
		m_last  = m_block_index.at(latch);
		m_header = header;
		uint32_t slots = 0;
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				m_slot.emplace(&inst, slots++);
				m_insts.push_back(const_cast<Inst*>(&inst));
			}
		}
		m_values.assign(static_cast<size_t>(slots) * m_wave * 4u, 0);
		m_taint.assign(slots, Taint {});
		m_decision.assign(slots, Taint {});
		m_merge = header->terminator.merge_block;
		m_active.assign(m_wave, 1u);
		m_varying.assign(slots, 0u);
		for (bool changed = true; changed;) {
			changed = false;
			for (const auto* block: program.blocks) {
				for (const auto& inst: *block) {
					const auto slot = m_slot.at(&inst);
					if (m_varying[slot] != 0) continue;
					const auto op = inst.GetOpcode();
					bool varying  = false;
					switch (op) {
						case Op::ReadFirstLane:
						case Op::ReadLane:
						case Op::Ballot:
						case Op::ConditionRef:
						case Op::SubgroupUMinU32:
						case Op::SubgroupUMaxU32:
						case Op::SubgroupOrU32:
						case Op::LiveMask:
						case Op::GetUserData: varying = false; break;
						case Op::LaneId: varying = true; break;
						case Op::GetBuiltin: {
							const auto kind = static_cast<StageInputKind>(inst.Arg(0).U32());
							varying = kind != StageInputKind::WorkgroupId && kind != StageInputKind::NumWorkgroups;
							break;
						}
						default: {
							const auto info = ImageOpcodeInfoOf(op);
							varying = info.access != ImageAccess::None || op == Op::ImageSampleRaw || op == Op::ImageGatherRaw ||
							          op == Op::ImageQueryLod || op == Op::GetAttribute || op == Op::GetAttributeF16 ||
							          op == Op::GetInterpolationParameter;
							for (size_t arg = 0; arg < inst.NumArgs() && !varying; ++arg) {
								const auto* used = inst.Arg(arg).Resolve().TryInstruction();
								varying = used != nullptr && m_varying[m_slot.at(used)] != 0;
							}
						}
					}
					if (varying) {
						m_varying[slot] = 1u;
						changed         = true;
					}
				}
			}
		}
		std::vector<const Block*> pending {header};
		m_reaches.insert(header);
		while (!pending.empty()) {
			const auto* block = pending.back();
			pending.pop_back();
			for (const auto* pred: block->ImmPredecessors()) {
				if (Inside(pred) || !m_reaches.insert(pred).second) continue;
				pending.push_back(pred);
			}
		}
		for (const auto* block: program.blocks) {
			if (!Inside(block)) continue;
			for (const auto& inst: *block) {
				for (size_t arg = 0; arg < inst.NumArgs(); ++arg) {
					const auto* used = inst.Arg(arg).Resolve().TryInstruction();
					if (used == nullptr || Inside(used->Parent())) continue;
					m_live_ids.try_emplace(m_slot.at(used), static_cast<uint32_t>(m_live_ids.size()));
				}
			}
		}
	}

	[[nodiscard]] bool Inside(const Block* block) const {
		const auto at = m_block_index.find(block);
		return at != m_block_index.end() && at->second >= m_first && at->second <= m_last;
	}

	[[nodiscard]] uint32_t Wave() const { return m_wave; }
	[[nodiscard]] const std::vector<const Block*>& Visited() const { return m_visited; }
	[[nodiscard]] const Block* Merge() const { return m_merge; }
	[[nodiscard]] uint32_t SlotOf(const Inst* inst) const { return m_slot.at(inst); }
	[[nodiscard]] Inst*    InstOf(uint32_t slot) const { return m_insts[slot]; }

	uint64_t& At(uint32_t slot, uint32_t lane, uint32_t k = 0) {
		return m_values[(static_cast<size_t>(slot) * m_wave + lane) * 4u + k];
	}

	[[nodiscard]] uint64_t Read(Value value, uint32_t lane, uint32_t k = 0) const {
		value = value.Resolve();
		if (const auto* inst = value.TryInstruction()) {
			return m_values[(static_cast<size_t>(m_slot.at(inst)) * m_wave + lane) * 4u + k];
		}
		return Immediate(value, k);
	}

	[[nodiscard]] Taint TaintOf(Value value) const { return ArgTaint(value); }
	[[nodiscard]] Taint DecisionTaint(const Inst* inst) const { return m_decision[m_slot.at(inst)]; }

	void Activate(std::vector<uint8_t> active) { m_active = std::move(active); }
	[[nodiscard]] uint64_t Work() const { return m_work; }

	void Reset(uint64_t seed) {
		m_seed = seed;
		m_written.clear();
		m_shared.clear();
		m_scratch.clear();
		m_user.clear();
		m_reads.clear();
		m_planted = nullptr;
	}

	void Plant(const std::unordered_map<uint64_t, uint32_t>* planted) {
		m_planted = planted;
		m_planted_reads.clear();
	}
	[[nodiscard]] bool PlantedRead(uint64_t address) const { return m_planted_reads.contains(address & ~uint64_t {3}); }
	[[nodiscard]] uint32_t UserData(uint32_t reg) const {
		if (const auto at = m_user.find(reg); at != m_user.end()) return at->second;
		return Hash(0x5553000000000000ull + reg) & 3u;
	}
	void SetUserData(uint32_t reg, uint32_t value) { m_user[reg] = value; }
	void Preset(uint64_t address, uint32_t value) { m_written[address & ~uint64_t {3}] = value; }
	[[nodiscard]] uint32_t Peek(uint64_t address) const {
		address &= ~uint64_t {3};
		if (const auto at = m_written.find(address); at != m_written.end()) return at->second;
		return Hash(address);
	}
	void TrackTaint(bool on) {
		m_tracking = on;
		if (on) m_control.reset();
	}
	[[nodiscard]] const Taint& Control() const { return m_control; }
	void SetRecords(std::vector<std::pair<uint64_t, uint32_t>> records) { m_records = std::move(records); }
	void SetLists(std::vector<uint64_t> lists) { m_lists = std::move(lists); }
	void Force(const std::unordered_map<const Block*, const Block*>* forced) { m_forced = forced; }

	[[nodiscard]] const std::unordered_set<uint64_t>& PreReads() const { return m_reads; }
	[[nodiscard]] const std::unordered_map<uint32_t, uint32_t>& LiveIds() const { return m_live_ids; }

	const char* RunPrologue(const Block*& from) {
		m_walk = false;
		const Block* previous = nullptr;
		const Block* block    = m_program.blocks.front();
		const auto   start    = m_work;
		for (uint32_t visits = 0; visits < 4 * MaxVisits; ++visits) {
			if (m_work - start > PrologueWork) return "the prologue exceeds its work bound";
			if (m_work > TotalWork) return "the derivation exceeds its work budget";
			if (block == m_header) {
				from = previous;
				return nullptr;
			}
			if (const char* failure = Visit(block, previous)) return failure;
			const Block* next = nullptr;
			if (block->terminator.kind == CFG::TerminatorKind::ConditionalBranch &&
			    m_reaches.contains(block->terminator.true_block) != m_reaches.contains(block->terminator.false_block)) {
				next = m_reaches.contains(block->terminator.true_block) ? block->terminator.true_block : block->terminator.false_block;
			} else if (const char* failure = Successor(block, next)) {
				return failure;
			}
			if (next == nullptr) {
				return "the program returns before the walk";
			}
			previous = block;
			block    = next;
		}
		return "the prologue does not reach the walk";
	}

	const char* RunWalk(const Block* from, const Oracle& oracle, const Block*& exit_from) {
		m_walk   = true;
		m_oracle = &oracle;
		m_visited.clear();
		const Block* previous = from;
		const Block* block    = m_header;
		const auto   start    = m_work;
		for (uint32_t visits = 0; visits < MaxVisits; ++visits) {
			if (m_work - start > WalkWork) return "the walk exceeds its work bound";
			if (m_work > TotalWork) return "the derivation exceeds its work budget";
			m_visited.push_back(block);
			if (const char* failure = Visit(block, previous)) return failure;
			const Block* next = nullptr;
			if (m_forced != nullptr) {
				if (const auto at = m_forced->find(block); at != m_forced->end()) next = at->second;
			}
			if (next == nullptr) {
				if (const char* failure = Successor(block, next)) return failure;
			}
			if (next == nullptr) return "the walk returns";
			if (!Inside(next)) {
				if (next != m_merge) return "the walk leaves through another block than its merge";
				exit_from = block;
				Phis(next, block);
				return nullptr;
			}
			previous = block;
			block    = next;
		}
		return "the walk does not terminate in the step bound";
	}

private:
	[[nodiscard]] uint64_t Immediate(Value value, uint32_t k) const {
		switch (value.GetType()) {
			case Type::U1: return value.U1() ? 1u : 0u;
			case Type::U8: return value.U8();
			case Type::U16: return value.U16();
			case Type::U32: return value.U32();
			case Type::F32: return B(value.F32Value());
			case Type::U64: return k == 0 ? value.U64() : 0u;
			default: return 0;
		}
	}

	uint32_t Hash(uint64_t key) const {
		const auto h    = Mix(key ^ (m_seed * 0x2545f4914f6cdd1dull));
		const auto pick = h >> 61u;
		if (pick < 2) return std::bit_cast<uint32_t>(8.0f + static_cast<float>(h & 0xfffu) * (120.0f / 4096.0f));
		if (pick < 4) return static_cast<uint32_t>(h & 15u);
		return static_cast<uint32_t>(h) & 0xff00u;
	}

	uint32_t Memory(uint64_t address) {
		address &= ~uint64_t {3};
		if (!m_walk) m_reads.insert(address);
		if (m_planted != nullptr) {
			if (const auto at = m_planted->find(address); at != m_planted->end()) {
				if (m_walk) m_planted_reads.insert(address);
				return at->second;
			}
		}
		if (const auto at = m_written.find(address); at != m_written.end()) return at->second;
		return Hash(address);
	}

	Taint MemoryTaint(uint64_t address) const {
		Taint taint;
		address &= ~uint64_t {3};
		for (const auto& [record, words]: m_records) {
			if (address >= record && address < record + words * 4u) {
				taint.set(TaintRecord + std::min<uint32_t>(static_cast<uint32_t>((address - record) / 4u), 47u));
				return taint;
			}
		}
		for (const auto list: m_lists) {
			if (address == list || address == list + 4u) {
				taint.set(TaintList + static_cast<uint32_t>((address - list) / 4u));
				return taint;
			}
		}
		taint.set(TaintMemory);
		return taint;
	}

	void Phis(const Block* block, const Block* from) {
		std::vector<std::pair<uint32_t, Value>> phis;
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != Op::Phi) continue;
			Value chosen;
			for (size_t index = 0; index < inst.NumArgs(); ++index) {
				if (inst.PhiBlock(index) == from) chosen = inst.Arg(index);
			}
			phis.emplace_back(m_slot.at(&inst), chosen);
		}
		auto& staged = m_staged;
		staged.assign(phis.size() * m_wave * 4u, 0u);
		std::vector<Taint> taints(m_tracking ? phis.size() : 0u);
		std::vector<Taint> decisions(m_tracking ? phis.size() : 0u);
		for (size_t index = 0; index < phis.size(); ++index) {
			const auto& [slot, value] = phis[index];
			if (value.IsEmpty()) continue;
			const auto resolved = value.Resolve();
			if (const auto* used = resolved.TryInstruction()) {
				const auto* source = &m_values[static_cast<size_t>(m_slot.at(used)) * m_wave * 4u];
				std::copy(source, source + m_wave * 4u, staged.begin() + static_cast<std::ptrdiff_t>(index * m_wave * 4u));
			} else {
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					for (uint32_t k = 0; k < 4; ++k) staged[(index * m_wave + lane) * 4u + k] = Immediate(resolved, k);
				}
			}
			if (m_tracking) {
				taints[index]    = ArgTaint(value);
				decisions[index] = ArgDecision(value);
			}
		}
		for (size_t index = 0; index < phis.size(); ++index) {
			const auto slot = phis[index].first;
			for (uint32_t lane = 0; lane < m_wave; ++lane) {
				for (uint32_t k = 0; k < 4; ++k) At(slot, lane, k) = staged[(index * m_wave + lane) * 4u + k];
			}
			if (m_tracking) {
				m_taint[slot]    = taints[index];
				m_decision[slot] = decisions[index];
			}
		}
	}

	[[nodiscard]] Taint ArgDecision(Value value) const {
		value = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || (!Inside(inst->Parent()) && inst->Parent() != m_merge)) return {};
		return m_decision[m_slot.at(inst)];
	}

	[[nodiscard]] Taint ArgTaint(Value value) const {
		value = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		const auto slot = m_slot.at(inst);
		if (!Inside(inst->Parent()) && inst->Parent() != m_merge) {
			Taint taint;
			const auto at = m_live_ids.find(slot);
			if (at != m_live_ids.end()) taint.set(TaintLive + std::min(at->second, TaintLives - 1u));
			return taint;
		}
		return m_taint[slot];
	}

	const char* Successor(const Block* block, const Block*& next) {
		const auto& term = block->terminator;
		switch (term.kind) {
			case CFG::TerminatorKind::Branch: next = term.true_block; return nullptr;
			case CFG::TerminatorKind::ConditionalBranch: {
				if (m_tracking && m_walk) m_control |= ArgTaint(block->condition);
				auto first = Read(block->condition, 0) & 1u;
				for (uint32_t lane = 1; lane < m_wave; ++lane) {
					if ((Read(block->condition, lane) & 1u) == first) continue;
					if (m_walk) return "a branch condition is not uniform";
					first = 1u;
				}
				next = first != 0 ? term.true_block : term.false_block;
				return nullptr;
			}
			case CFG::TerminatorKind::Return: next = nullptr; return nullptr;
			default: return "an indirect or unsupported branch";
		}
	}

	const char* Visit(const Block* block, const Block* from) {
		m_work += block->Instructions().size();
		Phis(block, from);
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == Op::Phi) continue;
			if (const char* failure = Execute(inst)) return failure;
		}
		return nullptr;
	}

	template <class Fn>
	void Map(uint32_t out, Fn&& fn) {
		for (uint32_t lane = 0; lane < m_wave; ++lane) At(out, lane) = fn(lane);
	}



	enum class Space { Global, Private, Shared };

	template <class Arg>
	std::pair<Space, uint64_t> Route(const Inst& inst, uint32_t lane, const Arg& a) const {
		const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
		const auto  offset = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(memory.offset)));
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if ((handle != nullptr && handle->GetOpcode() == Op::GetScratchResource) || memory.kind == ResourceKind::Scratch) {
			return {Space::Private, (a(1, lane) & 0xffffffffu) + offset};
		}
		if (memory.kind == ResourceKind::Lds) return {Space::Shared, (a(1, lane) & 0xffffffffu) + offset};
		const auto address = memory.address_is_full ? ((a(1, lane) & 0xffffffffu) | (a(2, lane) & 0xffffffffu) << 32u) + offset
		                                            : ((a(0, lane, 0) & 0xffffffffu) | (a(0, lane, 1) << 32u)) + (a(1, lane) & 0xffffffffu) + offset;
		const auto high = static_cast<uint32_t>(address >> 32u);
		if (memory.kind == ResourceKind::FlatLocal || memory.kind == ResourceKind::Flat) {
			if (high == 0x70000000u) return {Space::Private, address & 0xffffffffu};
			if (high == 0x80000000u) return {Space::Shared, address & 0xffffffffu};
		}
		return {Space::Global, address};
	}

	const char* Execute(const Inst& inst) {
		const char* failure = Compute(inst);
		if (failure == nullptr && !m_walk && inst.GetOpcode() != Op::ConditionRef && inst.GetType() == Type::U1 && m_varying[m_slot.at(&inst)] != 0) {
			const auto out = m_slot.at(&inst);
			for (uint32_t lane = 0; lane < m_wave; ++lane) At(out, lane) = m_active[lane];
		}
		return failure;
	}

	struct Ref {
		const uint64_t*         base = nullptr;
		std::array<uint64_t, 4> immediate {};
	};

	const char* Compute(const Inst& inst) {
		const auto op  = inst.GetOpcode();
		const auto out = m_slot.at(&inst);
		const auto count = inst.NumArgs();
		m_refs.resize(std::max<size_t>(m_refs.size(), count));
		for (size_t arg = 0; arg < count; ++arg) {
			const auto value = inst.Arg(arg).Resolve();
			auto&      ref   = m_refs[arg];
			if (const auto* used = value.TryInstruction()) {
				ref.base = &m_values[static_cast<size_t>(m_slot.at(used)) * m_wave * 4u];
			} else {
				ref.base = nullptr;
				for (uint32_t k = 0; k < 4; ++k) ref.immediate[k] = value.IsEmpty() ? 0u : Immediate(value, k);
			}
		}
		const auto a   = [&](size_t arg, uint32_t lane, uint32_t k = 0) {
			const auto& ref = m_refs[arg];
			return ref.base != nullptr ? ref.base[lane * 4u + k] : ref.immediate[k];
		};
		const auto f   = [&](size_t arg, uint32_t lane) { return F(a(arg, lane)); };
		if (m_tracking) {
			Taint taint, decision;
			for (size_t arg = 0; arg < inst.NumArgs(); ++arg) {
				taint |= ArgTaint(inst.Arg(arg));
				decision |= ArgDecision(inst.Arg(arg));
			}
			if (op == Op::SelectU1 || op == Op::SelectU32 || op == Op::SelectF32 || op == Op::SelectF64) decision |= ArgTaint(inst.Arg(0));
			m_taint[out]    = taint;
			m_decision[out] = decision;
		}
		const auto u32 = [](uint64_t value) { return value & 0xffffffffu; };
		const auto s32 = [](uint64_t value) { return static_cast<int32_t>(static_cast<uint32_t>(value)); };
		switch (op) {
			case Op::Reference:
			case Op::ReferenceU32:
			case Op::Void:
			case Op::ControlNop:
			case Op::InstPrefetch:
			case Op::StoreCompletion:
			case Op::DataCacheInvalidate:
			case Op::Barrier:
			case Op::Sendmsg:
			case Op::TtraceData:
			case Op::MakeImageAddress: return nullptr;
			case Op::Identity:
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = a(0, lane, k);
				}
				return nullptr;
			case Op::ConditionRef: {
				const auto kind = inst.Flags<CFG::BranchCondition>();
				if (kind == CFG::BranchCondition::ScalarInstruction) {
					Map(out, [&](uint32_t lane) { return a(0, lane) & 1u; });
					return nullptr;
				}
				const bool zero = kind == CFG::BranchCondition::ExecZero || kind == CFG::BranchCondition::VccZero ||
				                  kind == CFG::BranchCondition::SccZero;
				bool any = false, all = true;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					const bool value = (a(0, lane) & 1u) != 0;
					any              = any || value;
					all              = all && value;
				}
				const uint64_t result = zero ? (all ? 1u : 0u) : (any ? 1u : 0u);
				Map(out, [&](uint32_t) { return result; });
				return nullptr;
			}
			case Op::UndefU1:
			case Op::UndefU32: Map(out, [](uint32_t) { return uint64_t {0}; }); return nullptr;
			case Op::GetUserData:
				Map(out, [&](uint32_t) { return uint64_t {UserData(static_cast<uint32_t>(inst.Arg(0).ScalarRegister()))}; });
				return nullptr;
			case Op::GetBuiltin: {
				const auto kind  = static_cast<StageInputKind>(inst.Arg(0).U32());
				const auto index = inst.Arg(1).U32();
				Map(out, [&](uint32_t lane) -> uint64_t {
					switch (kind) {
						case StageInputKind::LocalInvocationId: return index == 0 ? lane % 8u : index == 1 ? lane / 8u : 0u;
						case StageInputKind::LocalInvocationIndex: return lane;
						case StageInputKind::GlobalInvocationId: return index == 0 ? 8u + lane % 8u : index == 1 ? 8u + lane / 8u : 0u;
						case StageInputKind::WorkgroupId: return index == 2 ? 0u : 1u;
						default: return Hash(0x4249000000000000ull + static_cast<uint64_t>(kind) * 16u + index) & 0xffu;
					}
				});
				return nullptr;
			}
			case Op::LaneId: Map(out, [](uint32_t lane) { return uint64_t {lane}; }); return nullptr;
			case Op::BitCastU32F32:
			case Op::BitCastF32U32:
			case Op::BitCastU64F64:
			case Op::BitCastF64U64:
			case Op::BitCastU16F16:
			case Op::BitCastF16U16: Map(out, [&](uint32_t lane) { return a(0, lane); }); return nullptr;
			case Op::ConvertU16U32: Map(out, [&](uint32_t lane) { return a(0, lane) & 0xffffu; }); return nullptr;
			case Op::ConvertU32U16: Map(out, [&](uint32_t lane) { return a(0, lane) & 0xffffu; }); return nullptr;
			case Op::ConvertU8U32: Map(out, [&](uint32_t lane) { return a(0, lane) & 0xffu; }); return nullptr;
			case Op::ConvertU32U8: Map(out, [&](uint32_t lane) { return a(0, lane) & 0xffu; }); return nullptr;
			case Op::ConvertS32F32:
				Map(out, [&](uint32_t lane) -> uint64_t {
					const float value = f(0, lane);
					if (std::isnan(value)) return 0;
					if (value >= 2147483647.0f) return 0x7fffffffu;
					if (value <= -2147483648.0f) return 0x80000000u;
					return static_cast<uint32_t>(static_cast<int32_t>(value));
				});
				return nullptr;
			case Op::ConvertU32F32:
				Map(out, [&](uint32_t lane) -> uint64_t {
					const float value = f(0, lane);
					if (std::isnan(value) || value <= 0.0f) return 0;
					if (value >= 4294967295.0f) return 0xffffffffu;
					return static_cast<uint32_t>(value);
				});
				return nullptr;
			case Op::ConvertF32S32: Map(out, [&](uint32_t lane) { return B(static_cast<float>(s32(a(0, lane)))); }); return nullptr;
			case Op::ConvertF32U32: Map(out, [&](uint32_t lane) { return B(static_cast<float>(u32(a(0, lane)))); }); return nullptr;
			case Op::CompositeConstructU64: Map(out, [&](uint32_t lane) { return u32(a(0, lane)) | u32(a(1, lane)) << 32u; }); return nullptr;
			case Op::CompositeConstructU32x2:
			case Op::CompositeConstructU32x3:
			case Op::CompositeConstructU32x4:
			case Op::CompositeConstructF32x2:
			case Op::GetBufferResource:
			case Op::GetAddressResource:
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = k < inst.NumArgs() ? a(k, lane) : 0u;
				}
				return nullptr;
			case Op::CompositeExtractU64: Map(out, [&](uint32_t lane) { return u32(a(0, lane) >> (a(1, lane) != 0 ? 32u : 0u)); }); return nullptr;
			case Op::CompositeExtractU32x2:
			case Op::CompositeExtractU32x3:
			case Op::CompositeExtractU32x4: Map(out, [&](uint32_t lane) { return a(0, lane, static_cast<uint32_t>(a(1, lane) & 3u)); }); return nullptr;
			case Op::FPAbs32: Map(out, [&](uint32_t lane) { return a(0, lane) & 0x7fffffffu; }); return nullptr;
			case Op::FPNeg32: Map(out, [&](uint32_t lane) { return (a(0, lane) ^ 0x80000000u) & 0xffffffffu; }); return nullptr;
			case Op::FPSaturate32: Map(out, [&](uint32_t lane) { return B(std::isnan(f(0, lane)) ? 0.0f : std::clamp(f(0, lane), 0.0f, 1.0f)); }); return nullptr;
			case Op::BitFieldInsert:
				Map(out, [&](uint32_t lane) -> uint64_t {
					const auto offset = a(2, lane) & 31u, count = a(3, lane) & 31u;
					const auto mask   = count == 0 ? 0u : static_cast<uint32_t>(((1ull << count) - 1u) << offset);
					return (u32(a(0, lane)) & ~mask) | ((u32(a(1, lane)) << offset) & mask);
				});
				return nullptr;
			case Op::BitFieldUExtract:
				Map(out, [&](uint32_t lane) -> uint64_t {
					const auto offset = a(1, lane) & 31u, count = a(2, lane) & 31u;
					return count == 0 ? 0u : (u32(a(0, lane)) >> offset) & ((1ull << count) - 1u);
				});
				return nullptr;
			case Op::BitFieldSExtract:
				Map(out, [&](uint32_t lane) -> uint64_t {
					const auto offset = a(1, lane) & 31u, count = a(2, lane) & 31u;
					if (count == 0) return 0;
					const auto bits = (u32(a(0, lane)) >> offset) & ((1ull << count) - 1u);
					const auto sign = 1ull << (count - 1u);
					return u32((bits ^ sign) - sign);
				});
				return nullptr;
			case Op::SelectU1:
			case Op::SelectU32:
			case Op::SelectF32:
			case Op::SelectF64:
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					const auto pick = (a(0, lane) & 1u) != 0 ? 1u : 2u;
					for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = a(pick, lane, k);
				}
				return nullptr;
			case Op::IAdd32: Map(out, [&](uint32_t lane) { return u32(a(0, lane) + a(1, lane)); }); return nullptr;
			case Op::IAdd64: Map(out, [&](uint32_t lane) { return a(0, lane) + a(1, lane); }); return nullptr;
			case Op::ISub32: Map(out, [&](uint32_t lane) { return u32(a(0, lane) - a(1, lane)); }); return nullptr;
			case Op::ISub64: Map(out, [&](uint32_t lane) { return a(0, lane) - a(1, lane); }); return nullptr;
			case Op::IMul32: Map(out, [&](uint32_t lane) { return u32(a(0, lane) * a(1, lane)); }); return nullptr;
			case Op::UDiv32: Map(out, [&](uint32_t lane) { return u32(a(1, lane)) == 0 ? 0xffffffffu : u32(a(0, lane)) / u32(a(1, lane)); }); return nullptr;
			case Op::SMulHi: Map(out, [&](uint32_t lane) { return u32(static_cast<uint64_t>((static_cast<int64_t>(s32(a(0, lane))) * s32(a(1, lane))) >> 32u)); }); return nullptr;
			case Op::UMulHi: Map(out, [&](uint32_t lane) { return (u32(a(0, lane)) * u32(a(1, lane))) >> 32u; }); return nullptr;
			case Op::IAbs32: Map(out, [&](uint32_t lane) { return u32(static_cast<uint64_t>(std::abs(static_cast<int64_t>(s32(a(0, lane)))))); }); return nullptr;
			case Op::ShiftLeftLogical32: Map(out, [&](uint32_t lane) { return u32(a(0, lane) << (a(1, lane) & 31u)); }); return nullptr;
			case Op::ShiftLeftLogical64: Map(out, [&](uint32_t lane) { return a(0, lane) << (a(1, lane) & 63u); }); return nullptr;
			case Op::ShiftRightLogical32: Map(out, [&](uint32_t lane) { return u32(a(0, lane)) >> (a(1, lane) & 31u); }); return nullptr;
			case Op::ShiftRightLogical64: Map(out, [&](uint32_t lane) { return a(0, lane) >> (a(1, lane) & 63u); }); return nullptr;
			case Op::ShiftRightArithmetic32: Map(out, [&](uint32_t lane) { return u32(static_cast<uint64_t>(s32(a(0, lane)) >> (a(1, lane) & 31u))); }); return nullptr;
			case Op::ShiftRightArithmetic64: Map(out, [&](uint32_t lane) { return static_cast<uint64_t>(static_cast<int64_t>(a(0, lane)) >> (a(1, lane) & 63u)); }); return nullptr;
			case Op::BitwiseAnd32: Map(out, [&](uint32_t lane) { return u32(a(0, lane) & a(1, lane)); }); return nullptr;
			case Op::BitwiseAnd64: Map(out, [&](uint32_t lane) { return a(0, lane) & a(1, lane); }); return nullptr;
			case Op::BitwiseOr32: Map(out, [&](uint32_t lane) { return u32(a(0, lane) | a(1, lane)); }); return nullptr;
			case Op::BitwiseXor32: Map(out, [&](uint32_t lane) { return u32(a(0, lane) ^ a(1, lane)); }); return nullptr;
			case Op::BitwiseNot32: Map(out, [&](uint32_t lane) { return u32(~a(0, lane)); }); return nullptr;
			case Op::BitReverse32: Map(out, [&](uint32_t lane) {
				uint32_t value = static_cast<uint32_t>(a(0, lane)), result = 0;
				for (uint32_t bit = 0; bit < 32; ++bit) result |= ((value >> bit) & 1u) << (31u - bit);
				return uint64_t {result};
			}); return nullptr;
			case Op::BitCount32: Map(out, [&](uint32_t lane) { return uint64_t(std::popcount(static_cast<uint32_t>(a(0, lane)))); }); return nullptr;
			case Op::FindUMsb32: Map(out, [&](uint32_t lane) -> uint64_t {
				const auto value = static_cast<uint32_t>(a(0, lane));
				return value == 0 ? 0xffffffffu : 31u - static_cast<uint32_t>(std::countl_zero(value));
			}); return nullptr;
			case Op::FindILsb32: Map(out, [&](uint32_t lane) -> uint64_t {
				const auto value = static_cast<uint32_t>(a(0, lane));
				return value == 0 ? 0xffffffffu : static_cast<uint32_t>(std::countr_zero(value));
			}); return nullptr;
			case Op::SMin32: Map(out, [&](uint32_t lane) { return u32(static_cast<uint64_t>(std::min(s32(a(0, lane)), s32(a(1, lane))))); }); return nullptr;
			case Op::UMin32: Map(out, [&](uint32_t lane) { return std::min(u32(a(0, lane)), u32(a(1, lane))); }); return nullptr;
			case Op::SMax32: Map(out, [&](uint32_t lane) { return u32(static_cast<uint64_t>(std::max(s32(a(0, lane)), s32(a(1, lane))))); }); return nullptr;
			case Op::UMax32: Map(out, [&](uint32_t lane) { return std::max(u32(a(0, lane)), u32(a(1, lane))); }); return nullptr;
			case Op::SMinTri32: Map(out, [&](uint32_t lane) { return u32(static_cast<uint64_t>(std::min({s32(a(0, lane)), s32(a(1, lane)), s32(a(2, lane))}))); }); return nullptr;
			case Op::UMinTri32: Map(out, [&](uint32_t lane) { return std::min({u32(a(0, lane)), u32(a(1, lane)), u32(a(2, lane))}); }); return nullptr;
			case Op::SMaxTri32: Map(out, [&](uint32_t lane) { return u32(static_cast<uint64_t>(std::max({s32(a(0, lane)), s32(a(1, lane)), s32(a(2, lane))}))); }); return nullptr;
			case Op::UMaxTri32: Map(out, [&](uint32_t lane) { return std::max({u32(a(0, lane)), u32(a(1, lane)), u32(a(2, lane))}); }); return nullptr;
			case Op::SMedTri32: Map(out, [&](uint32_t lane) {
				std::array<int32_t, 3> v {s32(a(0, lane)), s32(a(1, lane)), s32(a(2, lane))};
				std::ranges::sort(v);
				return u32(static_cast<uint64_t>(v[1]));
			}); return nullptr;
			case Op::UMedTri32: Map(out, [&](uint32_t lane) {
				std::array<uint64_t, 3> v {u32(a(0, lane)), u32(a(1, lane)), u32(a(2, lane))};
				std::ranges::sort(v);
				return v[1];
			}); return nullptr;
			case Op::SLessThan32: Map(out, [&](uint32_t lane) { return uint64_t(s32(a(0, lane)) < s32(a(1, lane))); }); return nullptr;
			case Op::SLessThan64: Map(out, [&](uint32_t lane) { return uint64_t(static_cast<int64_t>(a(0, lane)) < static_cast<int64_t>(a(1, lane))); }); return nullptr;
			case Op::SLessThanEqual64: Map(out, [&](uint32_t lane) { return uint64_t(static_cast<int64_t>(a(0, lane)) <= static_cast<int64_t>(a(1, lane))); }); return nullptr;
			case Op::ULessThanEqual64: Map(out, [&](uint32_t lane) { return uint64_t(a(0, lane) <= a(1, lane)); }); return nullptr;
			case Op::UGreaterThanEqual64: Map(out, [&](uint32_t lane) { return uint64_t(a(0, lane) >= a(1, lane)); }); return nullptr;
			case Op::ULessThan32: Map(out, [&](uint32_t lane) { return uint64_t(u32(a(0, lane)) < u32(a(1, lane))); }); return nullptr;
			case Op::ULessThan64: Map(out, [&](uint32_t lane) { return uint64_t(a(0, lane) < a(1, lane)); }); return nullptr;
			case Op::IEqual32: Map(out, [&](uint32_t lane) { return uint64_t(u32(a(0, lane)) == u32(a(1, lane))); }); return nullptr;
			case Op::IEqual64: Map(out, [&](uint32_t lane) { return uint64_t(a(0, lane) == a(1, lane)); }); return nullptr;
			case Op::SLessThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(s32(a(0, lane)) <= s32(a(1, lane))); }); return nullptr;
			case Op::ULessThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(u32(a(0, lane)) <= u32(a(1, lane))); }); return nullptr;
			case Op::SGreaterThan32: Map(out, [&](uint32_t lane) { return uint64_t(s32(a(0, lane)) > s32(a(1, lane))); }); return nullptr;
			case Op::UGreaterThan32: Map(out, [&](uint32_t lane) { return uint64_t(u32(a(0, lane)) > u32(a(1, lane))); }); return nullptr;
			case Op::UGreaterThan64: Map(out, [&](uint32_t lane) { return uint64_t(a(0, lane) > a(1, lane)); }); return nullptr;
			case Op::INotEqual32: Map(out, [&](uint32_t lane) { return uint64_t(u32(a(0, lane)) != u32(a(1, lane))); }); return nullptr;
			case Op::INotEqual64: Map(out, [&](uint32_t lane) { return uint64_t(a(0, lane) != a(1, lane)); }); return nullptr;
			case Op::SGreaterThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(s32(a(0, lane)) >= s32(a(1, lane))); }); return nullptr;
			case Op::UGreaterThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(u32(a(0, lane)) >= u32(a(1, lane))); }); return nullptr;
			case Op::LogicalOr: Map(out, [&](uint32_t lane) { return (a(0, lane) | a(1, lane)) & 1u; }); return nullptr;
			case Op::LogicalAnd: Map(out, [&](uint32_t lane) { return (a(0, lane) & a(1, lane)) & 1u; }); return nullptr;
			case Op::LogicalXor: Map(out, [&](uint32_t lane) { return (a(0, lane) ^ a(1, lane)) & 1u; }); return nullptr;
			case Op::LogicalNot: Map(out, [&](uint32_t lane) { return (a(0, lane) & 1u) ^ 1u; }); return nullptr;
			case Op::FPOrdEqual32: Map(out, [&](uint32_t lane) { return uint64_t(f(0, lane) == f(1, lane)); }); return nullptr;
			case Op::FPUnordEqual32: Map(out, [&](uint32_t lane) { return uint64_t(!(f(0, lane) != f(1, lane))); }); return nullptr;
			case Op::FPOrdNotEqual32: Map(out, [&](uint32_t lane) { return uint64_t(f(0, lane) < f(1, lane) || f(0, lane) > f(1, lane)); }); return nullptr;
			case Op::FPUnordNotEqual32: Map(out, [&](uint32_t lane) { return uint64_t(f(0, lane) != f(1, lane)); }); return nullptr;
			case Op::FPOrdLessThan32: Map(out, [&](uint32_t lane) { return uint64_t(f(0, lane) < f(1, lane)); }); return nullptr;
			case Op::FPUnordLessThan32: Map(out, [&](uint32_t lane) { return uint64_t(!(f(0, lane) >= f(1, lane))); }); return nullptr;
			case Op::FPOrdGreaterThan32: Map(out, [&](uint32_t lane) { return uint64_t(f(0, lane) > f(1, lane)); }); return nullptr;
			case Op::FPUnordGreaterThan32: Map(out, [&](uint32_t lane) { return uint64_t(!(f(0, lane) <= f(1, lane))); }); return nullptr;
			case Op::FPOrdLessThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(f(0, lane) <= f(1, lane)); }); return nullptr;
			case Op::FPUnordLessThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(!(f(0, lane) > f(1, lane))); }); return nullptr;
			case Op::FPOrdGreaterThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(f(0, lane) >= f(1, lane)); }); return nullptr;
			case Op::FPUnordGreaterThanEqual32: Map(out, [&](uint32_t lane) { return uint64_t(!(f(0, lane) < f(1, lane))); }); return nullptr;
			case Op::FPIsNan32: Map(out, [&](uint32_t lane) { return uint64_t(std::isnan(f(0, lane))); }); return nullptr;
			case Op::FPCmpClass32:
				Map(out, [&](uint32_t lane) -> uint64_t {
					const float    value = f(0, lane);
					const uint32_t mask  = static_cast<uint32_t>(a(1, lane));
					const bool     minus = std::signbit(value);
					uint32_t       cls   = 0;
					switch (std::fpclassify(value)) {
						case FP_NAN: cls = (std::bit_cast<uint32_t>(value) & 0x400000u) != 0 ? 1u : 0u; break;
						case FP_INFINITE: cls = minus ? 2u : 9u; break;
						case FP_NORMAL: cls = minus ? 3u : 8u; break;
						case FP_SUBNORMAL: cls = minus ? 4u : 7u; break;
						default: cls = minus ? 5u : 6u; break;
					}
					return (mask >> cls) & 1u;
				});
				return nullptr;
			case Op::FPAdd32: Map(out, [&](uint32_t lane) { return B(f(0, lane) + f(1, lane)); }); return nullptr;
			case Op::FPSub32: Map(out, [&](uint32_t lane) { return B(f(0, lane) - f(1, lane)); }); return nullptr;
			case Op::FPMul32: Map(out, [&](uint32_t lane) { return B(f(0, lane) * f(1, lane)); }); return nullptr;
			case Op::FPFma32: Map(out, [&](uint32_t lane) { return B(std::fma(f(0, lane), f(1, lane), f(2, lane))); }); return nullptr;
			case Op::FPMad32: Map(out, [&](uint32_t lane) { return B(f(0, lane) * f(1, lane) + f(2, lane)); }); return nullptr;
			case Op::FPMin32: Map(out, [&](uint32_t lane) { return B(std::fmin(f(0, lane), f(1, lane))); }); return nullptr;
			case Op::FPMax32: Map(out, [&](uint32_t lane) { return B(std::fmax(f(0, lane), f(1, lane))); }); return nullptr;
			case Op::FPMinTri32: Map(out, [&](uint32_t lane) { return B(std::fmin(std::fmin(f(0, lane), f(1, lane)), f(2, lane))); }); return nullptr;
			case Op::FPMaxTri32: Map(out, [&](uint32_t lane) { return B(std::fmax(std::fmax(f(0, lane), f(1, lane)), f(2, lane))); }); return nullptr;
			case Op::FPMedTri32: Map(out, [&](uint32_t lane) {
				std::array<float, 3> v {f(0, lane), f(1, lane), f(2, lane)};
				if (std::ranges::any_of(v, [](float x) { return std::isnan(x); })) return B(std::fmin(std::fmin(v[0], v[1]), v[2]));
				std::ranges::sort(v);
				return B(v[1]);
			}); return nullptr;
			case Op::FPRecip32:
			case Op::FPRecipIFlag32: Map(out, [&](uint32_t lane) { return B(1.0f / f(0, lane)); }); return nullptr;
			case Op::FPRecipSqrt32: Map(out, [&](uint32_t lane) { return B(1.0f / std::sqrt(f(0, lane))); }); return nullptr;
			case Op::FPSqrt: Map(out, [&](uint32_t lane) { return B(std::sqrt(f(0, lane))); }); return nullptr;
			case Op::FPSin: Map(out, [&](uint32_t lane) { return B(std::sin(f(0, lane))); }); return nullptr;
			case Op::FPCos: Map(out, [&](uint32_t lane) { return B(std::cos(f(0, lane))); }); return nullptr;
			case Op::FPExp2: Map(out, [&](uint32_t lane) { return B(std::exp2(f(0, lane))); }); return nullptr;
			case Op::FPLog2: Map(out, [&](uint32_t lane) { return B(std::log2(f(0, lane))); }); return nullptr;
			case Op::FPLdexp: Map(out, [&](uint32_t lane) { return B(std::ldexp(f(0, lane), s32(a(1, lane)))); }); return nullptr;
			case Op::FPRoundEven32: Map(out, [&](uint32_t lane) { return B(std::nearbyint(f(0, lane))); }); return nullptr;
			case Op::FPFloor32: Map(out, [&](uint32_t lane) { return B(std::floor(f(0, lane))); }); return nullptr;
			case Op::FPCeil32: Map(out, [&](uint32_t lane) { return B(std::ceil(f(0, lane))); }); return nullptr;
			case Op::FPTrunc32: Map(out, [&](uint32_t lane) { return B(std::trunc(f(0, lane))); }); return nullptr;
			case Op::FPFract32: Map(out, [&](uint32_t lane) { return B(f(0, lane) - std::floor(f(0, lane))); }); return nullptr;
			case Op::Ballot: {
				uint64_t mask = 0;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					if ((a(0, lane) & 1u) != 0) mask |= 1ull << lane;
				}
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					At(out, lane, 0) = mask & 0xffffffffu;
					At(out, lane, 1) = mask >> 32u;
					At(out, lane, 2) = 0;
					At(out, lane, 3) = 0;
				}
				return nullptr;
			}
			case Op::LiveMask: {
				const uint64_t mask = m_wave == 64 ? ~0ull : (1ull << m_wave) - 1u;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					At(out, lane, 0) = mask & 0xffffffffu;
					At(out, lane, 1) = mask >> 32u;
					At(out, lane, 2) = 0;
					At(out, lane, 3) = 0;
				}
				return nullptr;
			}
			case Op::ReadFirstLane: {
				uint32_t first = 0;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					if ((a(1, lane) & 1u) != 0) {
						first = lane;
						break;
					}
				}
				const auto value = a(0, first);
				Map(out, [&](uint32_t) { return value; });
				return nullptr;
			}
			case Op::ReadLane: {
				std::vector<uint64_t> result(m_wave);
				for (uint32_t lane = 0; lane < m_wave; ++lane) result[lane] = a(0, static_cast<uint32_t>(a(1, lane)) & (m_wave - 1u));
				Map(out, [&](uint32_t lane) { return result[lane]; });
				return nullptr;
			}
			case Op::WriteLane: {
				std::vector<uint64_t> result(m_wave);
				for (uint32_t lane = 0; lane < m_wave; ++lane) result[lane] = u32(a(2, lane)) == lane ? a(1, lane) : a(0, lane);
				Map(out, [&](uint32_t lane) { return result[lane]; });
				return nullptr;
			}
			case Op::SubgroupUMinU32:
			case Op::SubgroupUMaxU32:
			case Op::SubgroupOrU32: {
				uint64_t value = op == Op::SubgroupUMinU32 ? 0xffffffffu : 0u;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					const auto x = u32(a(0, lane));
					value        = op == Op::SubgroupUMinU32 ? std::min(value, x) : op == Op::SubgroupUMaxU32 ? std::max(value, x) : value | x;
				}
				Map(out, [&](uint32_t) { return value; });
				return nullptr;
			}
			case Op::ReadConstBuffer: {
				const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
				const auto* buffer = inst.Arg(0).Resolve().TryInstruction();
				if (buffer == nullptr) return "a constant buffer read without resource";
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					const auto base    = (a(0, lane, 0) & 0xffffffffu) | ((a(0, lane, 1) & 0xffffu) << 32u);
					const auto address = base + u32(a(1, lane)) + memory.offset;
					At(out, lane)      = Memory(address);
					if (m_tracking && m_walk) {
						if (lane == 0) m_taint[out].reset();
						m_taint[out] |= MemoryTaint(address);
					}
				}
				return nullptr;
			}
			case Op::StoreAddressU32: {
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					if ((a(4, lane) & 1u) == 0) continue;
					const auto [space, address] = Route(inst, lane, a);
					if (space == Space::Global && m_walk) return "the walk writes memory";
					const auto value = static_cast<uint32_t>(a(3, lane));
					if (space == Space::Private) m_scratch[(address & ~uint64_t {3}) << 6u | lane] = value;
					if (space == Space::Shared) m_shared[static_cast<uint32_t>(address & ~uint64_t {3})] = value;
					if (space == Space::Global) m_written[address & ~uint64_t {3}] = value;
				}
				return nullptr;
			}
			case Op::LoadBufferU32:
			case Op::LoadBufferU32x2:
			case Op::LoadBufferU32x3:
			case Op::LoadBufferU32x4: {
				const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
				const uint32_t count = op == Op::LoadBufferU32 ? 1u : op == Op::LoadBufferU32x2 ? 2u : op == Op::LoadBufferU32x3 ? 3u : 4u;
				if (m_walk && (memory.formatted || memory.typed)) return "a formatted buffer load in the walk";
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					if ((a(4, lane) & 1u) == 0) {
						for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = 0;
						continue;
					}
					const auto w0 = a(0, lane, 0), w1 = a(0, lane, 1), w3 = a(0, lane, 3);
					if (m_walk && ((w3 >> 23u) & 1u) != 0 && ((w1 >> 16u) & 0x3fffu) != 0) return "a swizzled buffer load in the walk";
					const auto base    = (w0 & 0xffffffffu) | ((w1 & 0xffffu) << 32u);
					const auto stride  = (w1 >> 16u) & 0x3fffu;
					const auto address = base + (a(1, lane) & 0xffffffffu) * stride + (a(2, lane) & 0xffffffffu) + (a(3, lane) & 0xffffffffu) + memory.offset;
					for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = k < count ? Memory(address + 4u * k) : 0u;
					if (m_tracking && m_walk) {
						if (lane == 0) m_taint[out].reset();
						m_taint[out] |= MemoryTaint(address);
					}
				}
				return nullptr;
			}
			case Op::LoadAddressU32:
			case Op::LoadAddressU16:
			case Op::LoadAddressU8: {
				const auto& kind = m_program.memory_info[inst.Flags<MemoryFlags>().index].kind;
				const auto* handle = inst.Arg(0).Resolve().TryInstruction();
				if ((handle != nullptr && handle->GetOpcode() == Op::GetScratchResource) || kind == ResourceKind::FlatLocal ||
				    kind == ResourceKind::Scratch || kind == ResourceKind::Flat) {
					bool local = true;
					for (uint32_t lane = 0; lane < m_wave; ++lane) {
						if ((a(3, lane) & 1u) == 0) {
							At(out, lane) = 0;
							continue;
						}
						const auto [space, address] = Route(inst, lane, a);
						if (space == Space::Global) {
							local = false;
							break;
						}
						uint64_t value = 0;
						if (space == Space::Private) {
							const auto at = m_scratch.find((address & ~uint64_t {3}) << 6u | lane);
							value         = at != m_scratch.end() ? at->second : Hash(0x5c00000000000000ull + (address << 6u) + lane);
						} else {
							const auto at = m_shared.find(static_cast<uint32_t>(address & ~uint64_t {3}));
							value         = at != m_shared.end() ? at->second : Hash(0x1d50000000000000ull + (address & 0xffffffffu));
						}
						value >>= (address & 3u) * 8u;
						if (op == Op::LoadAddressU16) value &= 0xffffu;
						if (op == Op::LoadAddressU8) value &= 0xffu;
						At(out, lane) = value;
					}
					if (local) return nullptr;
				}
				bool cleared = false;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					if ((a(3, lane) & 1u) == 0) {
						At(out, lane) = 0;
						continue;
					}
					const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
					const auto  offset = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(memory.offset)));
					const auto  address = memory.address_is_full ? ((a(1, lane) & 0xffffffffu) | (a(2, lane) & 0xffffffffu) << 32u) + offset
					                                             : ((a(0, lane, 0) & 0xffffffffu) | (a(0, lane, 1) << 32u)) + (a(1, lane) & 0xffffffffu) + offset;
					auto        value   = static_cast<uint64_t>(Memory(address));
					value >>= (address & 3u) * 8u;
					if (op == Op::LoadAddressU16) value &= 0xffffu;
					if (op == Op::LoadAddressU8) value &= 0xffu;
					At(out, lane) = value;
					if (m_tracking && m_walk) {
						if (!cleared) m_taint[out].reset();
						cleared = true;
						m_taint[out] |= MemoryTaint(address);
					}
				}
				return nullptr;
			}
			case Op::LoadDeviceAddressU32:
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					if ((a(3, lane) & 1u) == 0) {
						At(out, lane) = 0;
						continue;
					}
					const auto address = (u32(a(0, lane)) | u32(a(1, lane)) << 32u) + u32(a(2, lane));
					At(out, lane)      = Memory(address);
					if (m_tracking && m_walk) m_taint[out] |= MemoryTaint(address);
				}
				return nullptr;
			case Op::LoadSharedU32:
			case Op::LoadSharedU32x2:
			case Op::LoadSharedU32x3:
			case Op::LoadSharedU32x4: {
				const uint32_t count = op == Op::LoadSharedU32 ? 1u : op == Op::LoadSharedU32x2 ? 2u : op == Op::LoadSharedU32x3 ? 3u : 4u;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					const auto address = static_cast<uint32_t>(a(0, lane));
					for (uint32_t k = 0; k < 4; ++k) {
						if (k >= count || (a(1, lane) & 1u) == 0) {
							At(out, lane, k) = 0;
							continue;
						}
						const auto at    = m_shared.find(address + 4u * k);
						At(out, lane, k) = at != m_shared.end() ? at->second : Hash(0x1d50000000000000ull + address + 4u * k);
					}
				}
				return nullptr;
			}
			case Op::WriteSharedU32:
			case Op::WriteSharedU32x2:
			case Op::WriteSharedU32x3:
			case Op::WriteSharedU32x4: {
				const uint32_t count = op == Op::WriteSharedU32 ? 1u : op == Op::WriteSharedU32x2 ? 2u : op == Op::WriteSharedU32x3 ? 3u : 4u;
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					if ((a(count + 1u, lane) & 1u) == 0) continue;
					const auto address = static_cast<uint32_t>(a(0, lane));
					for (uint32_t k = 0; k < count; ++k) m_shared[address + 4u * k] = static_cast<uint32_t>(a(1u + k, lane));
				}
				return nullptr;
			}
			case Op::GetScratchResource:
				for (uint32_t lane = 0; lane < m_wave; ++lane) {
					for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = 0;
				}
				return nullptr;
			case Op::BvhIntersect: return Intersect(inst);
			default: break;
		}
		if (m_walk) {
			return "an instruction the interpreter does not run";
		}
		if (TypeOf(op) == Type::Void) return nullptr;
		for (uint32_t lane = 0; lane < m_wave; ++lane) {
			for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = Hash((static_cast<uint64_t>(out) << 20u) + k);
		}
		return nullptr;
	}

	const char* Intersect(const Inst& inst) {
		if (!m_walk) {
			const auto out = m_slot.at(&inst);
			const auto* ray = inst.Arg(1).Resolve().TryInstruction();
			for (uint32_t lane = 0; lane < m_wave; ++lane) {
				const bool triangle = ray != nullptr && ray->GetOpcode() == Op::MakeImageAddress && (Read(ray->Arg(0), lane) & 7u) < 4u;
				const std::array<uint64_t, 4> missed = triangle ? std::array<uint64_t, 4> {InfBits, OneBits, Miss, 0u} : std::array<uint64_t, 4> {Miss, Miss, Miss, Miss};
				for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = (Read(inst.Arg(2), lane) & 1u) != 0 ? missed[k] : 0u;
			}
			return nullptr;
		}
		if (m_oracle == nullptr) return "an intersection outside the walk";
		const auto* address = inst.Arg(1).Resolve().TryInstruction();
		if (address == nullptr || address->GetOpcode() != Op::MakeImageAddress) return "an intersection without its ray";
		const uint32_t words = inst.Flags<uint32_t>() & 0xffu;
		if (words != 1u && words != 2u) return "an intersection with an unknown node width";
		std::vector<Call> calls;
		for (uint32_t lane = 0; lane < m_wave; ++lane) {
			if ((Read(inst.Arg(2), lane) & 1u) == 0) continue;
			Call call;
			call.lane = lane;
			for (uint32_t k = 0; k < 4; ++k) call.descriptor[k] = static_cast<uint32_t>(Read(inst.Arg(0), lane, k));
			call.base   = ((static_cast<uint64_t>(call.descriptor[0]) | (static_cast<uint64_t>(call.descriptor[1] & 0xffu) << 32u)) << 8u);
			call.node   = Read(address->Arg(0), lane) & 0xffffffffu;
			if (words == 2u) call.node |= (Read(address->Arg(1), lane) & 0xffffffffu) << 32u;
			call.extent = static_cast<uint32_t>(Read(address->Arg(words), lane));
			calls.push_back(call);
		}
		std::vector<std::array<uint32_t, 4>> answers(calls.size());
		if (!(*m_oracle)(inst, calls, answers)) return "the oracle refuses a call";
		const auto out = m_slot.at(&inst);
		for (uint32_t lane = 0; lane < m_wave; ++lane) {
			for (uint32_t k = 0; k < 4; ++k) At(out, lane, k) = 0;
		}
		for (size_t index = 0; index < calls.size(); ++index) {
			for (uint32_t k = 0; k < 4; ++k) At(out, calls[index].lane, k) = answers[index][k];
		}
		if (m_tracking) {
			m_taint[out].reset();
			m_taint[out].set(TaintOracle);
			m_decision[out].reset();
		}
		return nullptr;
	}

	const Program&                             m_program;
	uint32_t                                   m_wave;
	std::unordered_map<const Block*, size_t>   m_block_index;
	std::unordered_map<const Inst*, uint32_t>  m_slot;
	std::vector<Inst*>                         m_insts;
	std::vector<uint64_t>                      m_values;
	std::vector<Taint>                         m_taint;
	std::vector<Taint>                         m_decision;
	Taint                                      m_control;
	size_t                                     m_first  = 0;
	size_t                                     m_last   = 0;
	const Block*                               m_header = nullptr;
	const Block*                               m_merge  = nullptr;
	uint64_t                                   m_seed   = 0;
	bool                                       m_walk   = false;
	bool                                       m_tracking = false;
	const Oracle*                              m_oracle = nullptr;
	const std::unordered_map<uint64_t, uint32_t>* m_planted = nullptr;
	const std::unordered_map<const Block*, const Block*>* m_forced = nullptr;
	std::unordered_map<uint64_t, uint32_t>     m_written;
	std::unordered_map<uint32_t, uint32_t>     m_shared;
	std::unordered_map<uint64_t, uint32_t>     m_scratch;
	std::unordered_map<uint32_t, uint32_t>     m_user;
	std::unordered_set<uint64_t>               m_reads;
	std::unordered_set<uint64_t>               m_planted_reads;
	std::vector<const Block*>                  m_visited;
	std::vector<Ref>                           m_refs;
	std::vector<uint64_t>                      m_staged;
	std::unordered_map<uint32_t, uint32_t>     m_live_ids;
	std::unordered_set<const Block*>           m_reaches;
	std::vector<uint8_t>                       m_active;
	std::vector<uint8_t>                       m_varying;
	uint64_t                                   m_work = 0;
	std::vector<std::pair<uint64_t, uint32_t>> m_records;
	std::vector<uint64_t>                      m_lists;
};

enum Kind : uint32_t { KindT, KindU, KindV, KindWord, KindNode, KindInstanceLo, KindInstanceHi, KindHas, KindCount };

constexpr std::array<RayQueryResultKind, KindCount> HostKinds {RayQueryResultKind::T,          RayQueryResultKind::U,
                                                               RayQueryResultKind::V,          RayQueryResultKind::Word,
                                                               RayQueryResultKind::Node,       RayQueryResultKind::InstanceLo,
                                                               RayQueryResultKind::InstanceHi, RayQueryResultKind::Has};

constexpr std::array<std::pair<uint32_t, uint32_t>, 8> Transforms {{{0u, 0xffffffffu}, {0u, 7u}, {0u, 3u}, {3u, 0xffffffffu}, {0u, 0x3fffffffu}, {30u, 1u}, {31u, 1u}, {30u, 3u}}};

uint64_t Transformed(uint64_t value, uint32_t transform) {
	return ((value & 0xffffffffu) >> Transforms[transform].first) & Transforms[transform].second;
}

enum class Opacity : uint32_t { AllOpaque, OnlyOpaque, Live };

struct Params {
	bool     both        = false;
	uint32_t cull        = NoSlot;
	uint32_t flip        = NoSlot;
	Opacity  opacity     = Opacity::AllOpaque;
	uint32_t alpha_off   = NoSlot;
	uint32_t ray_mask    = NoSlot;
	bool     flags       = false;
	uint32_t flags_dword = 0;
	uint32_t flags_shift = 0;
	bool     flags_flip  = false;
	bool     mask        = false;
	uint32_t mask_dword  = 0;
	uint32_t mask_shift  = 0;
};

struct Instance {
	uint32_t              leaf   = 0;
	uint64_t              record = 0;
	uint64_t              blas   = 0;
	uint32_t              root   = 0;
	std::vector<uint32_t> words;
};

struct Triangle {
	uint32_t instance = 0;
	uint32_t entry    = 0;
	uint32_t tri      = 0;
	uint32_t word     = 0;
};

struct Candidate {
	uint32_t triangle = 0;
	float    t        = 0.0f;
	float    u        = 0.0f;
	float    v        = 0.0f;
	float    den      = 1.0f;
	bool     front    = true;
	bool     inside   = true;
};

struct LaneScript {
	std::vector<Candidate> candidates;
	size_t                 position = 0;
	bool                   started  = false;
	bool                   offered  = false;
	uint32_t               proposed = 0;
	uint32_t               leaf     = Miss;
	uint32_t               entry    = Miss;
};

struct World {
	uint64_t                               header        = 0;
	uint64_t                               instance_base = 0;
	uint32_t                               root          = 0;
	std::vector<Instance>                  instances;
	std::vector<Triangle>                  triangles;
	std::unordered_map<uint64_t, uint32_t> memory;
	std::vector<LaneScript>                lanes;
	bool                                   proposal      = false;
	uint32_t                               proposed      = 0;
	std::vector<uint32_t>                  history;
	uint32_t                               starts        = 0;
	const Inst*                            start_inst    = nullptr;
	std::vector<float>                     extents;

	[[nodiscard]] uint32_t Leaf(uint32_t triangle) const { return instances[triangles[triangle].instance].leaf; }
};

struct Row {
	bool                  front  = true;
	uint32_t              inside = 1;
	bool                  opaque = true;
	uint32_t              flags  = 0;
	uint32_t              mask   = 0xff;
	uint32_t              type   = 0;
	std::vector<uint64_t> lives;
};

struct Observation {
	uint32_t                            seed = 0;
	uint32_t                            lane = 0;
	std::vector<uint64_t>               outs;
	std::vector<uint64_t>               base;
	std::vector<uint64_t>               lives;
	std::vector<Candidate>              candidates;
	std::vector<Instance>               instances;
	std::vector<Triangle>               triangles;
	float                               extent = 0.0f;
};

struct LiveOut {
	Inst* inst  = nullptr;
	bool  phi   = false;
	bool  flag  = false;
};

struct Live {
	uint32_t slot = 0;
	Inst*    inst = nullptr;
	bool     flag = false;
};

struct Start {
	std::vector<const Block*>              path;
	const Inst*                            inst = nullptr;
	std::vector<uint32_t>                  lanes;
	std::vector<float>                     extents;
	uint64_t                               header = 0;
	std::array<Taint, 16>                  roles {};
};

class Random {
public:
	explicit Random(uint64_t seed): m_state(seed) {}
	uint64_t Next() { return Mix(m_state++); }
	uint32_t Below(uint32_t bound) { return bound == 0 ? 0u : static_cast<uint32_t>(Next() % bound); }

private:
	uint64_t m_state;
};

bool Usable(float extent) {
	return extent >= 1.0e-3f && extent <= 1.0e30f;
}

uint32_t Fraction(Random& random, uint32_t steps) {
	return 1u + random.Below(steps - 1u);
}

class Analysis {
public:
	Analysis(Program& program, const NativeTraversalOptions& options, Block* header, Block* latch,
	         std::span<Inst* const> served, NativeTraversals& result)
	    : m_program(program), m_options(options), m_header(header), m_served(served), m_result(result),
	      m_layout(result.layout), m_machine(program, options.wave_size, header, latch) {}

	const char* Derive();
	void        Emit(uint32_t site);

	uint32_t detail = 0;

private:
	void        Begin(uint64_t seed);
	const char* Prepare();
	const char* Seed(uint64_t seed, Start& start);
	const char* Discover(uint64_t seed, const Start& start);
	const char* Trial(uint64_t seed, const Start& start);
	const char* Play(const Start& start, World world, std::vector<std::vector<uint64_t>>& outs,
	                 const std::vector<std::pair<uint32_t, std::vector<uint64_t>>>& overrides, bool calls = true);
	const char* Rows();
	const char* Classify();
	const char* Search();
	const char* Validate();
	const char* Hoist(const Start& start);
	Value       Peel(Value value, std::unordered_map<const Inst*, Value>& peeled, bool& failed);
	const char* ForcedPath();

	World    MakeWorld(const Start& start, Random& random) const;
	uint32_t AddInstance(World& world, Random& random, uint32_t type, uint32_t mask, uint32_t flags, bool opaque_words) const;
	uint32_t AddTriangle(World& world, Random& random, uint32_t instance, bool opaque) const;
	void     PlantWorld(World& world) const;
	bool     Answer(World& world, std::span<const Machine::Call> calls, std::span<std::array<uint32_t, 4>> answers) const;
	[[nodiscard]] std::array<uint64_t, KindCount> KindValues(const World& world, const Candidate& candidate) const;
	[[nodiscard]] bool HostAccepts(const Params& params, const Instance& instance, const Triangle& triangle,
	                               const Candidate& candidate, std::span<const uint64_t> lives) const;
	[[nodiscard]] std::optional<size_t> HostClosest(const Params& params, const Observation& observation) const;
	[[nodiscard]] bool Inline(uint32_t node) const { return m_layout.Inline(node); }

	Program&                       m_program;
	const NativeTraversalOptions&  m_options;
	Block*                         m_header;
	std::span<Inst* const>         m_served;
	NativeTraversals&              m_result;
	Layout                         m_layout;
	Machine                        m_machine;
	std::vector<LiveOut>           m_outs;
	std::vector<Live>              m_lives;
	std::vector<uint64_t>          m_seeds;
	std::vector<Start>             m_starts;
	std::vector<uint32_t>          m_types;
	std::vector<uint32_t>          m_flag_dwords;
	std::vector<Observation>       m_rows;
	std::vector<std::optional<Kind>> m_kinds;
	std::vector<uint32_t>          m_transforms;
	std::vector<std::optional<uint64_t>> m_fixed;
	Params                         m_params;
	std::array<Value, 6>           m_ray {};
	Value                          m_extent;
	Value                          m_lanes;
	std::array<Value, 2>           m_descriptor {};
	std::array<Value, 2>           m_node {};
	std::vector<std::pair<Block*, Block*>> m_path;
	Block*                         m_preheader = nullptr;
};

const char* Analysis::Prepare() {
	if (m_program.stage != ShaderType::Compute) return "only compute walks are rewritten";
	if (m_options.wave_size != 32 && m_options.wave_size != 64) return "an unknown wave size";
	if (m_header->terminator.merge_block == nullptr) return "the walk has no merge block";
	for (auto* pred: m_header->ImmPredecessors()) {
		if (m_machine.Inside(pred)) continue;
		if (m_preheader != nullptr) return "the walk has several entries";
		m_preheader = pred;
	}
	if (m_preheader == nullptr) return "the walk has no entry";
	const auto* merge = m_machine.Merge();
	for (const auto* block: m_program.blocks) {
		if (!m_machine.Inside(block)) continue;
		for (auto* next: block->ImmSuccessors()) {
			if (!m_machine.Inside(next) && next != merge) return "the walk leaves through another block than its merge";
		}
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == Op::BvhIntersect && std::ranges::find(m_served, &inst) == m_served.end()) {
				return "an intersection of the walk is not served";
			}
			const bool used_after = std::ranges::any_of(inst.Uses(), [&](const Use& use) {
				return !m_machine.Inside(use.user->Parent()) && !(use.user->Parent() == merge && use.user->GetOpcode() == Op::Phi);
			});
			if (used_after) {
				const auto type = inst.GetType();
				if (type != Type::U32 && type != Type::U1 && type != Type::F32) return "a value of the walk used after it is not a word";
				m_outs.push_back({const_cast<Inst*>(&inst), false, type == Type::U1});
			}
		}
	}
	for (auto& inst: *const_cast<Block*>(merge)) {
		if (inst.GetOpcode() != Op::Phi) continue;
		const auto type = inst.GetType();
		if (type != Type::U32 && type != Type::U1 && type != Type::F32) return "a merge value of the walk is not a word";
		m_outs.push_back({&inst, true, type == Type::U1});
	}
	if (m_outs.empty()) return "the walk has no result";
	m_types = {0u, 1u, 2u, 3u};
	if (m_layout.inline_record) m_types.push_back(m_layout.inline_type);
	return nullptr;
}

std::vector<uint8_t> ActiveLanes(uint64_t seed, uint32_t wave) {
	std::vector<uint8_t> active(wave, 0u);
	Random random(seed * 0xa24baed4963ee407ull);
	const uint32_t count = std::min<uint32_t>(wave, 16u);
	for (uint32_t k = 0; k < count; ++k) active[(seed % 2 == 0 ? k * (wave / count) + random.Below(wave / count) : k)] = 1u;
	return active;
}

void Analysis::Begin(uint64_t seed) {
	m_machine.Reset(seed);
	m_machine.Activate(ActiveLanes(seed, m_machine.Wave()));
	const auto header  = 0x0000100000000000ull + ((Mix(seed) & 0xffffffu) << 16u);
	const auto table   = 0x0000080000000000ull + ((Mix(seed + 77u) & 0xffffffu) << 12u);
	const auto sgpr    = m_result.user_sgpr;
	auto       pointer = static_cast<uint64_t>(m_machine.UserData(sgpr)) | static_cast<uint64_t>(m_machine.UserData(sgpr + 1u)) << 32u;
	if (m_result.header_vsharp == 0xfffffffeu) {
		pointer = m_result.header_offset == 0xffffffffu ? header : table;
		m_machine.SetUserData(sgpr, static_cast<uint32_t>(pointer));
		m_machine.SetUserData(sgpr + 1u, static_cast<uint32_t>(pointer >> 32u));
	} else if (m_result.header_vsharp != 0xffffffffu) {
		const auto loaded = m_result.header_offset == 0xffffffffu ? header : table;
		m_machine.Preset(pointer + m_result.header_vsharp, static_cast<uint32_t>(loaded));
		m_machine.Preset(pointer + m_result.header_vsharp + 4u, static_cast<uint32_t>(loaded >> 32u) & 0xffffu);
		pointer = loaded;
	} else if (m_result.header_offset == 0xffffffffu) {
		m_machine.SetUserData(sgpr, static_cast<uint32_t>(header));
		m_machine.SetUserData(sgpr + 1u, static_cast<uint32_t>(header >> 32u));
	}
	if (m_result.header_offset != 0xffffffffu) {
		m_machine.Preset(pointer + m_result.header_offset, static_cast<uint32_t>(header));
		m_machine.Preset(pointer + m_result.header_offset + 4u, static_cast<uint32_t>(header >> 32u));
	}
}

const char* Analysis::Seed(uint64_t seed, Start& start) {
	Begin(seed);
	const Block* from = nullptr;
	if (const char* failure = m_machine.RunPrologue(from)) return failure;
	if (from != m_preheader) return "the walk is entered from another block";
	bool             recorded = false;
	Machine::Oracle  oracle   = [&](const Inst& inst, std::span<const Machine::Call> calls, std::span<std::array<uint32_t, 4>>) {
		if (calls.empty()) return false;
		const auto root  = m_result.root;
		const auto first = calls.front();
		const auto address = first.base + ((first.node >> 3u) << 6u);
		const auto header  = address - ((static_cast<uint64_t>(root) >> 3u) << 6u);
		for (const auto& call: calls) {
			const auto kind = call.node & 7u;
			if ((kind != 4u && kind != 5u) || call.base + ((call.node >> 3u) << 6u) != address) {
				return false;
			}
		}
		start.inst   = &inst;
		start.header = header;
		start.path   = m_machine.Visited();
		for (const auto& call: calls) {
			start.lanes.push_back(call.lane);
			start.extents.push_back(F(call.extent));
		}
		const auto* ray = inst.Arg(1).Resolve().TryInstruction();
		start.roles[0]  = m_machine.TaintOf(inst.Arg(0));
		start.roles[1]  = m_machine.TaintOf(inst.Arg(2));
		for (size_t arg = 0; arg < ray->NumArgs() && arg + 2 < start.roles.size(); ++arg) start.roles[arg + 2] = m_machine.TaintOf(ray->Arg(arg));
		recorded = true;
		return false;
	};
	const Block* exit_from = nullptr;
	m_machine.TrackTaint(true);
	const char* failure = m_machine.RunWalk(from, oracle, exit_from);
	m_machine.TrackTaint(false);
	if (!recorded) return failure != nullptr ? failure : "the walk exits without an intersection";
	if ((start.header & 0xffu) != 0 || start.header == 0) return "the top-level header is not aligned";
	return nullptr;
}

World Analysis::MakeWorld(const Start& start, Random& random) const {
	World world;
	world.header        = start.header;
	world.root          = m_result.root;
	const auto io       = start.header + m_layout.instance_offset;
	world.instance_base = start.header + (static_cast<uint64_t>(m_machine.Peek(io)) | static_cast<uint64_t>(m_machine.Peek(io + 4u)) << 32u);
	(void)random;
	world.lanes.assign(m_machine.Wave(), {});
	return world;
}

uint32_t Analysis::AddInstance(World& world, Random& random, uint32_t type, uint32_t mask, uint32_t flags, bool) const {
	Instance instance;
	uint32_t index = 0;
	for (;;) {
		index = 1u + random.Below((1u << 16u) - 1u);
		const auto leaf = index << 3u | type;
		if (std::ranges::none_of(world.instances, [&](const Instance& other) { return (other.leaf >> 3u) == index || (Inline(leaf) && Inline(other.leaf)); }) ||
		    (Inline(leaf) && std::ranges::any_of(world.instances, [&](const Instance& other) { return other.leaf == leaf; }))) {
			break;
		}
	}
	instance.leaf   = index << 3u | type;
	instance.record = Inline(instance.leaf) ? world.header + m_layout.inline_offset : world.instance_base + static_cast<uint64_t>(index) * m_layout.stride;
	if (Inline(instance.leaf)) {
		for (const auto& other: world.instances) {
			if (Inline(other.leaf)) return static_cast<uint32_t>(&other - world.instances.data());
		}
	}
	instance.blas = (0x0000200000000000ull + (static_cast<uint64_t>(random.Below(1u << 24u)) << 12u)) & ~uint64_t {0xff};
	instance.root = (1u + random.Below(1u << 12u)) << 3u | 5u;
	instance.words.resize(std::max<uint32_t>(m_layout.Words(), 40u));
	for (auto& word: instance.words) word = static_cast<uint32_t>(random.Next());
	for (uint32_t i = 0; i < 12; ++i) {
		const auto row = i / 4u, column = i % 4u;
		instance.words[m_layout.transform[i] % instance.words.size()] = std::bit_cast<uint32_t>(row == column ? 1.0f : 0.0f);
	}
	if (m_layout.packed) {
		instance.words[m_layout.blas]      = static_cast<uint32_t>(instance.blas >> 3u);
		instance.words[m_layout.blas + 1u] = static_cast<uint32_t>(instance.blas >> 35u) & 0x3fffffu;
	} else {
		instance.words[m_layout.blas]      = static_cast<uint32_t>(instance.blas);
		instance.words[m_layout.blas + 1u] = static_cast<uint32_t>(instance.blas >> 32u);
	}
	if (Inline(instance.leaf) && m_layout.inline_root != 0xffu) {
		instance.root = m_layout.inline_root;
	} else {
		instance.words[m_layout.root] = instance.root;
	}
	if (m_layout.mask) {
		const auto width = m_layout.mask_width == 0 ? 8u : m_layout.mask_width;
		const auto field = static_cast<uint32_t>(((1ull << width) - 1u) << m_layout.mask_shift);
		auto&      word  = instance.words[m_layout.mask_dword];
		word             = (word & ~field) | ((mask << m_layout.mask_shift) & field);
	}
	const bool     known_flags = m_params.flags || m_layout.flags;
	const uint32_t dword       = m_params.flags ? m_params.flags_dword : m_layout.flags_dword;
	const uint32_t shift       = m_params.flags ? m_params.flags_shift : m_layout.flags_shift;
	if (known_flags) {
		auto& word = instance.words[dword];
		word       = (word & ~(15u << shift)) | ((flags & 15u) << shift);
	}
	world.instances.push_back(std::move(instance));
	return static_cast<uint32_t>(world.instances.size() - 1u);
}

uint32_t Analysis::AddTriangle(World& world, Random& random, uint32_t instance, bool opaque) const {
	Triangle triangle;
	triangle.instance = instance;
	for (;;) {
		triangle.entry = (1u + random.Below((1u << 16u) - 1u)) << 3u;
		if (std::ranges::none_of(world.triangles, [&](const Triangle& other) { return other.instance == instance && other.entry == triangle.entry; })) break;
	}
	triangle.tri  = (1u + random.Below(1u << 16u)) << 3u | random.Below(4u);
	triangle.word = (static_cast<uint32_t>(random.Next()) & 0x3fffffffu) | (opaque ? 1u << 30u : 0u) | 1u << 31u;
	world.triangles.push_back(triangle);
	return static_cast<uint32_t>(world.triangles.size() - 1u);
}

void Analysis::PlantWorld(World& world) const {
	for (const auto& instance: world.instances) {
		for (uint32_t k = 0; k < m_layout.Words(); ++k) world.memory[instance.record + 4u * k] = instance.words[k];
	}
	for (const auto& triangle: world.triangles) {
		const auto& instance = world.instances[triangle.instance];
		const auto  at       = instance.blas + (static_cast<uint64_t>(triangle.entry >> 3u) << 3u);
		world.memory[at]      = triangle.tri;
		world.memory[at + 4u] = triangle.word;
	}
}

bool Analysis::Answer(World& world, std::span<const Machine::Call> calls, std::span<std::array<uint32_t, 4>> answers) const {
	const auto first = world.header + ((static_cast<uint64_t>(world.root) >> 3u) << 6u);
	const auto again = first + 64u;
	std::vector<size_t> top;
	for (size_t index = 0; index < calls.size(); ++index) {
		const auto& call    = calls[index];
		const auto  address = call.base + ((call.node >> 3u) << 6u);
		const auto  kind    = call.node & 7u;
		if ((kind == 4u || kind == 5u) && (address == first || address == again)) top.push_back(index);
	}
	const auto key_less = [&](uint32_t a, uint32_t b) {
		const auto la = world.Leaf(a), lb = world.Leaf(b);
		return la != lb ? la < lb : world.triangles[a].entry < world.triangles[b].entry;
	};
	const auto beyond = [](const Candidate& candidate, float extent) { return !(candidate.t <= extent * 1.0001f); };
	if (!top.empty()) {
		bool     any  = false;
		uint32_t best = 0;
		for (const auto index: top) {
			const auto& call    = calls[index];
			auto&       lane    = world.lanes[call.lane];
			const auto  extent  = F(call.extent);
			const auto  address = call.base + ((call.node >> 3u) << 6u);
			if (address == first) {
				lane.started  = true;
				lane.position = 0;
				lane.offered  = false;
				++world.starts;
			} else if (!lane.started) {
				continue;
			} else if (lane.position < lane.candidates.size()) {
				const auto& current = lane.candidates[lane.position];
				const bool  taken   = lane.offered && world.Leaf(current.triangle) == lane.leaf && world.triangles[current.triangle].entry == lane.entry;
				if (taken || beyond(current, extent)) ++lane.position;
			}
			while (lane.position < lane.candidates.size() && beyond(lane.candidates[lane.position], extent)) ++lane.position;
			if (lane.position < lane.candidates.size()) {
				const auto triangle = lane.candidates[lane.position].triangle;
				if (!any || key_less(triangle, best)) best = triangle;
				any = true;
			}
		}
		world.proposal = any;
		world.proposed = best;
		if (any) world.history.push_back(world.triangles[best].instance);
		for (const auto index: top) {
			const auto& call = calls[index];
			auto&       lane = world.lanes[call.lane];
			if (!lane.started || !any) {
				answers[index] = {Miss, Miss, Miss, Miss};
				lane.offered   = false;
				if (lane.started && !any) lane.started = false;
				continue;
			}
			lane.offered   = true;
			lane.proposed  = best;
			lane.leaf      = world.Leaf(best);
			lane.entry     = world.triangles[best].entry;
			answers[index] = {lane.leaf, world.root + 8u, Miss, Miss};
		}
	}
	for (size_t index = 0; index < calls.size(); ++index) {
		if (std::ranges::find(top, index) != top.end()) continue;
		const auto& call = calls[index];
		auto&       lane = world.lanes[call.lane];
		if ((call.node & 7u) > 5u) {
			answers[index] = {Miss, Miss, Miss, Miss};
			continue;
		}
		if (!lane.offered) {
			return false;
		}
		const auto& triangle = world.triangles[lane.proposed];
		const auto& instance = world.instances[triangle.instance];
		const auto  base     = instance.blas & ~uint64_t {0xff};
		const auto  address  = call.base + ((call.node >> 3u) << 6u);
		const auto  kind     = call.node & 7u;
		if (kind == 4u || kind == 5u) {
			if (address != base + ((static_cast<uint64_t>(instance.root) >> 3u) << 6u)) {
				return false;
			}
			answers[index] = {triangle.entry, Miss, Miss, Miss};
			continue;
		}
		if (kind >= 4u || address != base + ((static_cast<uint64_t>(triangle.tri) >> 3u) << 6u) || kind != (triangle.tri & 7u)) {
			return false;
		}
		const bool bary  = (call.descriptor[3] & 0x01000000u) != 0;
		answers[index] = {InfBits, OneBits, Miss, 0u};
		for (const auto& candidate: lane.candidates) {
			if (candidate.triangle != lane.proposed) continue;
			answers[index] = {static_cast<uint32_t>(B(candidate.t * candidate.den)), static_cast<uint32_t>(B(candidate.den)),
			                  bary ? static_cast<uint32_t>(B(candidate.u * candidate.den)) : triangle.tri ^ 0x5a5a0000u,
			                  bary ? static_cast<uint32_t>(B(candidate.v * candidate.den)) : 1u};
		}
	}
	return true;
}

std::array<uint64_t, KindCount> Analysis::KindValues(const World& world, const Candidate& candidate) const {
	const auto& triangle = world.triangles[candidate.triangle];
	const auto& instance = world.instances[triangle.instance];
	return {B(candidate.t), B(candidate.u), B(candidate.v), triangle.word, triangle.tri,
	        instance.record & 0xffffffffu, instance.record >> 32u, 1u};
}

const char* Analysis::Play(const Start& start, World world, std::vector<std::vector<uint64_t>>& outs,
                           const std::vector<std::pair<uint32_t, std::vector<uint64_t>>>& overrides, bool calls) {
	PlantWorld(world);
	std::vector<std::pair<uint64_t, uint32_t>> records;
	std::vector<uint64_t>                      lists;
	for (const auto& instance: world.instances) records.emplace_back(instance.record, m_layout.Words());
	for (const auto& triangle: world.triangles) lists.push_back(world.instances[triangle.instance].blas + (static_cast<uint64_t>(triangle.entry >> 3u) << 3u));
	m_machine.SetRecords(std::move(records));
	m_machine.SetLists(std::move(lists));
	for (const auto& [address, value]: world.memory) {
		if (m_machine.PreReads().contains(address)) return "the prologue reads the planted world";
	}
	std::vector<std::pair<std::pair<uint32_t, uint32_t>, uint64_t>> saved;
	for (const auto& [live, values]: overrides) {
		for (uint32_t lane = 0; lane < values.size(); ++lane) {
			saved.push_back({{m_lives[live].slot, lane}, m_machine.At(m_lives[live].slot, lane)});
			m_machine.At(m_lives[live].slot, lane) = values[lane];
		}
	}
	m_machine.Plant(&world.memory);
	Machine::Oracle oracle = [&](const Inst& inst, std::span<const Machine::Call> list, std::span<std::array<uint32_t, 4>> answers) {
		if (!calls) return false;
		const auto first = world.header + ((static_cast<uint64_t>(world.root) >> 3u) << 6u);
		for (const auto& call: list) {
			if (call.base + ((call.node >> 3u) << 6u) == first && &inst != start.inst) {
				return false;
			}
		}
		return Answer(world, list, answers);
	};
	const Block* exit_from = nullptr;
	const char*  failure   = m_machine.RunWalk(m_preheader, oracle, exit_from);
	if (failure == nullptr) {
		for (const auto instance: world.history) {
			const auto& record = world.instances[instance];
			bool        read   = false;
			for (uint32_t k = 0; k < m_layout.Words() && !read; ++k) read = m_machine.PlantedRead(record.record + 4u * k);
			if (!read) {
				failure = "the walk does not read the planted record of a proposed instance";
				break;
			}
		}
	}
	m_machine.Plant(nullptr);
	if (failure == nullptr) {
		outs.assign(m_outs.size(), std::vector<uint64_t>(m_machine.Wave()));
		for (size_t index = 0; index < m_outs.size(); ++index) {
			const auto slot = m_machine.SlotOf(m_outs[index].inst);
			for (uint32_t lane = 0; lane < m_machine.Wave(); ++lane) outs[index][lane] = m_machine.At(slot, lane) & 0xffffffffu;
		}
	}
	for (const auto& [where, value]: saved) m_machine.At(where.first, where.second) = value;
	return failure;
}

const char* Analysis::Trial(uint64_t seed, const Start& start) {
	Random random(seed * 0x3bu + 11u);
	Begin(seed);
	const Block* from = nullptr;
	if (const char* failure = m_machine.RunPrologue(from)) return failure;
	World world = MakeWorld(start, random);
	for (size_t at = 0; at < start.lanes.size(); ++at) {
		const auto extent = start.extents[at];
		if (!Usable(extent)) continue;
		const auto type     = m_types[at % m_types.size()];
		const auto instance = AddInstance(world, random, type, 0xffu, 0u, true);
		Candidate  candidate;
		candidate.triangle = AddTriangle(world, random, instance, true);
		candidate.t        = extent * 0.5f;
		world.lanes[start.lanes[at]].candidates.push_back(candidate);
	}
	std::vector<std::vector<uint64_t>> outs;
	return Play(start, world, outs, {});
}

const char* Analysis::Discover(uint64_t seed, const Start& start) {
	Random random(seed * 0x51u + 7u);
	Begin(seed);
	{
		const Block* from = nullptr;
		if (const char* failure = m_machine.RunPrologue(from)) return failure;
	}
	World world = MakeWorld(start, random);
	std::vector<uint32_t> lanes;
	for (size_t at = 0; at < start.lanes.size(); ++at) {
		const auto lane   = start.lanes[at];
		const auto extent = start.extents[at];
		if (!Usable(extent)) continue;
		const auto type     = m_types[random.Below(static_cast<uint32_t>(m_types.size()))];
		const auto instance = AddInstance(world, random, type, random.Below(2) != 0 ? 0xffu : random.Below(256), random.Below(16), true);
		Candidate  candidate;
		candidate.triangle = AddTriangle(world, random, instance, random.Below(2) != 0);
		candidate.t        = extent * static_cast<float>(Fraction(random, 64)) / 64.0f;
		candidate.front    = random.Below(2) != 0;
		candidate.den      = (candidate.front ? 1.0f : -1.0f) * ((random.Below(2) != 0) ? 2.0f : 1.0f);
		candidate.u        = 0.25f;
		candidate.v        = 0.5f;
		world.lanes[lane].candidates.push_back(candidate);
		lanes.push_back(lane);
	}
	World base = world;
	for (auto& lane: base.lanes) lane.candidates.clear();
	m_machine.TrackTaint(true);
	std::vector<std::vector<uint64_t>> outs, base_outs;
	const char* failure = Play(start, world, outs, {});
	m_machine.TrackTaint(false);
	if (failure != nullptr) return failure;
	Taint decided = m_machine.Control();
	for (const auto& out: m_outs) decided |= m_machine.DecisionTaint(out.inst);
	Taint roles;
	for (const auto& role: start.roles) roles |= role;
	if (const char* failure = Play(start, base, base_outs, {})) return failure;
	const auto accepted = [&](const std::vector<std::vector<uint64_t>>& with, const std::vector<std::vector<uint64_t>>& without) {
		std::vector<bool> result;
		for (const auto lane: lanes) {
			bool any = false;
			for (size_t index = 0; index < m_outs.size(); ++index) any = any || with[index][lane] != without[index][lane];
			result.push_back(any);
		}
		return result;
	};
	const auto natural = accepted(outs, base_outs);
	for (const auto& [slot, id]: m_machine.LiveIds()) {
		if (id >= TaintLives - 1u || !decided.test(TaintLive + id) || roles.test(TaintLive + id)) continue;
		Inst*      inst = m_machine.InstOf(slot);
		const auto type = inst->GetType();
		if (type != Type::U1 && type != Type::U32 && type != Type::F32) continue;
		const bool flag = type == Type::U1;
		const std::vector<uint64_t> domain = flag ? std::vector<uint64_t> {0u, 1u} : std::vector<uint64_t> {0u, 1u, 0x80u, 0xffu, 0x7fu, 0x40u};
		m_lives.push_back({slot, inst, flag});
		const auto live = static_cast<uint32_t>(m_lives.size() - 1u);
		bool sensitive = false, structural = false;
		for (const auto value: domain) {
			std::vector<uint64_t> values(m_machine.Wave());
			for (uint32_t lane = 0; lane < m_machine.Wave(); ++lane) values[lane] = m_machine.At(slot, lane);
			for (const auto lane: lanes) values[lane] = value;
			std::vector<std::pair<uint32_t, std::vector<uint64_t>>> overrides {{live, values}};
			std::vector<std::vector<uint64_t>> moved, moved_base;
			if (Play(start, base, moved_base, overrides) != nullptr || Play(start, world, moved, overrides) != nullptr) {
				structural = true;
				break;
			}
			if (accepted(moved, moved_base) != natural) sensitive = true;
		}
		if (!sensitive || structural) m_lives.pop_back();
	}
	if (m_lives.size() > 3) return "more than three walk inputs decide a hit";
	for (uint32_t k = 0; k < 40; ++k) {
		if (!decided.test(TaintRecord + k)) continue;
		if (k == m_layout.blas || k == m_layout.blas + 1u || k == m_layout.root ||
		    std::ranges::find(m_layout.transform, k) != m_layout.transform.end()) {
			continue;
		}
		m_flag_dwords.push_back(k);
	}
	return nullptr;
}

const char* Analysis::Rows() {
	const bool flags = m_layout.flags || !m_flag_dwords.empty();
	std::vector<std::vector<uint64_t>> domains;
	for (const auto& live: m_lives) {
		if (live.flag) {
			domains.push_back({0u, 1u});
		} else {
			domains.push_back({0u, 1u, 0x80u, 0xffu, 0x7fu, 0x40u});
		}
	}
	std::vector<Row> rows;
	const std::vector<uint32_t> masks = m_layout.mask ? std::vector<uint32_t> {0u, 1u, 0x80u, 0xffu, 0x5au} : std::vector<uint32_t> {0xffu};
	const uint32_t flag_count = flags ? 16u : 1u;
	size_t lives_count = 1;
	for (const auto& domain: domains) lives_count *= domain.size();
	for (uint32_t type_index = 0; type_index < m_types.size(); ++type_index) {
		for (uint32_t front = 0; front < 2; ++front) {
			for (uint32_t inside = 0; inside < 4; ++inside) {
				for (uint32_t opaque = 0; opaque < 2; ++opaque) {
					for (uint32_t flag = 0; flag < flag_count; ++flag) {
						for (const auto mask: masks) {
							for (size_t combo = 0; combo < lives_count; ++combo) {
								Row row;
								row.front  = front != 0;
								row.inside = inside;
								row.opaque = opaque != 0;
								row.flags  = flag;
								row.mask   = mask;
								row.type   = m_types[type_index];
								size_t rest = combo;
								for (const auto& domain: domains) {
									row.lives.push_back(domain[rest % domain.size()]);
									rest /= domain.size();
								}
								rows.push_back(std::move(row));
							}
						}
					}
				}
			}
		}
	}
	Random shuffle(0x9e37u);
	for (size_t index = rows.size(); index > 1; --index) std::swap(rows[index - 1], rows[shuffle.Below(static_cast<uint32_t>(index))]);
	if (rows.size() > MaxRows) rows.resize(MaxRows);
	size_t next = 0;
	for (uint32_t round = 0; next < rows.size(); ++round) {
		const auto  which = round % m_seeds.size();
		const auto  seed  = m_seeds[which];
		const auto& start = m_starts[which];
		Random random(seed * 0x1000003u + round);
		Begin(seed);
		const Block* from = nullptr;
		if (const char* failure = m_machine.RunPrologue(from)) return failure;
		World world = MakeWorld(start, random);
		std::vector<std::pair<uint32_t, std::vector<uint64_t>>> overrides;
		for (size_t live = 0; live < m_lives.size(); ++live) {
			std::vector<uint64_t> values(m_machine.Wave());
			for (uint32_t lane = 0; lane < m_machine.Wave(); ++lane) values[lane] = m_machine.At(m_lives[live].slot, lane);
			overrides.emplace_back(static_cast<uint32_t>(live), std::move(values));
		}
		std::vector<std::pair<uint32_t, size_t>> placed;
		for (size_t at = 0; at < start.lanes.size() && next < rows.size(); ++at) {
			const auto lane   = start.lanes[at];
			const auto extent = start.extents[at];
			if (!Usable(extent)) continue;
			const auto& row      = rows[next];
			const auto  instance = AddInstance(world, random, row.type, row.mask, row.flags, true);
			Candidate   candidate;
			candidate.triangle = AddTriangle(world, random, instance, row.opaque);
			const float scale  = row.inside == 0 ? 1.0f / 1024.0f : row.inside == 1 ? 0.5f : row.inside == 2 ? 0.9921875f : 2.0f;
			candidate.t        = extent * scale;
			candidate.inside   = row.inside != 3;
			candidate.front    = row.front;
			candidate.den      = (row.front ? 1.0f : -1.0f) * ((random.Below(2) != 0) ? 2.0f : 1.0f);
			candidate.u        = static_cast<float>(random.Below(128)) / 256.0f;
			candidate.v        = static_cast<float>(random.Below(128)) / 256.0f;
			world.lanes[lane].candidates.push_back(candidate);
			for (size_t live = 0; live < m_lives.size(); ++live) overrides[live].second[lane] = row.lives[live];
			placed.emplace_back(lane, next);
			++next;
		}
		if (placed.empty()) return "no lane of the walk has a usable extent";
		World base = world;
		for (auto& lane: base.lanes) lane.candidates.clear();
		std::vector<std::vector<uint64_t>> base_outs, outs;
		if (const char* failure = Play(start, base, base_outs, overrides)) return failure;
		if (const char* failure = Play(start, world, outs, overrides)) {
			detail = round;
			return failure;
		}
		for (const auto& [lane, row]: placed) {
			Observation observation;
			observation.seed = static_cast<uint32_t>(seed);
			observation.lane = lane;
			for (size_t index = 0; index < m_outs.size(); ++index) {
				observation.outs.push_back(outs[index][lane]);
				observation.base.push_back(base_outs[index][lane]);
			}
			for (size_t live = 0; live < m_lives.size(); ++live) observation.lives.push_back(overrides[live].second[lane] & 0xffffffffu);
			auto candidate         = world.lanes[lane].candidates.front();
			auto triangle          = world.triangles[candidate.triangle];
			observation.instances  = {world.instances[triangle.instance]};
			triangle.instance      = 0;
			candidate.triangle     = 0;
			observation.triangles  = {triangle};
			observation.candidates = {candidate};
			m_rows.push_back(std::move(observation));
		}
	}
	return nullptr;
}


const char* Analysis::Classify() {
	m_kinds.assign(m_outs.size(), std::nullopt);
	m_transforms.assign(m_outs.size(), 0u);
	std::vector<uint64_t> possible(m_outs.size(), ~uint64_t {0});
	std::vector<bool>     changed(m_outs.size(), false);
	uint32_t              accepted = 0;
	for (const auto& row: m_rows) {
		World world;
		world.instances = row.instances;
		world.triangles = row.triangles;
		const auto values = KindValues(world, row.candidates.front());
		bool any = false;
		for (size_t index = 0; index < m_outs.size(); ++index) any = any || row.outs[index] != row.base[index];
		if (!any) continue;
		++accepted;
		for (size_t index = 0; index < m_outs.size(); ++index) {
			if (row.outs[index] == row.base[index]) continue;
			changed[index] = true;
			uint64_t match = 0;
			for (uint32_t transform = 0; transform < Transforms.size(); ++transform) {
				for (uint32_t kind = 0; kind < KindCount; ++kind) {
					const bool flag = m_outs[index].flag;
					const bool hit  = kind == KindHas ? transform == 0 && row.outs[index] == 1u
					                                  : !flag && row.outs[index] == Transformed(values[kind], transform);
					if (hit) match |= uint64_t {1} << (transform * KindCount + kind);
				}
			}
			possible[index] &= match;
		}
	}
	if (accepted == 0) return "the walk never accepts a planted hit";
	bool distance = false;
	for (size_t index = 0; index < m_outs.size(); ++index) {
		if (!changed[index]) continue;
		if (possible[index] == 0) {
			detail = static_cast<uint32_t>(index);
			return "a value of the walk changes with a hit and is no host result";
		}
		const auto code     = static_cast<uint32_t>(std::countr_zero(possible[index]));
		m_kinds[index]      = static_cast<Kind>(code % KindCount);
		m_transforms[index] = code / KindCount;
		distance       = distance || *m_kinds[index] == KindT;
	}
	for (const auto& row: m_rows) {
		World world;
		world.instances = row.instances;
		world.triangles = row.triangles;
		const auto values = KindValues(world, row.candidates.front());
		bool any = false;
		for (size_t index = 0; index < m_outs.size(); ++index) any = any || row.outs[index] != row.base[index];
		if (!any) continue;
		for (size_t index = 0; index < m_outs.size(); ++index) {
			if (!m_kinds[index]) continue;
			const auto expected = m_outs[index].flag ? 1u : Transformed(values[*m_kinds[index]], m_transforms[index]);
			if (row.outs[index] != expected) return "a hit updates the results only in part";
		}
	}
	if (!distance) return "no value of the walk carries the hit distance";
	return nullptr;
}

bool Analysis::HostAccepts(const Params& params, const Instance& instance, const Triangle& triangle, const Candidate& candidate,
                           std::span<const uint64_t> lives) const {
	if (!candidate.inside) return false;
	const auto live = [&](uint32_t index) { return lives[index] & 0xffffffffu; };
	uint32_t   ray_mask = params.ray_mask == NoSlot ? 0xffu : static_cast<uint32_t>(live(params.ray_mask)) & 0xffu;
	uint32_t   mask     = 0xffu;
	if (params.mask) {
		mask = (instance.words[params.mask_dword] >> params.mask_shift) & 0xffu;
	} else if (m_layout.mask) {
		const auto bits = instance.words[m_layout.mask_dword] >> m_layout.mask_shift;
		mask = m_layout.mask_width == 0 ? bits & 0xffu : (bits & ((1u << m_layout.mask_width) - 1u)) != 0 ? 0xffu : 0u;
	}
	if ((mask & ray_mask) == 0) return false;
	uint32_t guest = 0;
	bool     flip  = m_layout.flip;
	if (params.flags) {
		guest = instance.words[params.flags_dword] >> params.flags_shift;
		flip  = params.flags_flip;
	} else if (m_layout.flags) {
		guest = instance.words[m_layout.flags_dword] >> m_layout.flags_shift;
	}
	const bool cull_off = (guest & 4u) != 0;
	const bool flipped  = !cull_off && (((guest >> 3u) ^ (flip ? 1u : 0u)) & 1u) != 0;
	const bool no_opaque = (guest & 2u) != 0;
	const bool opaque    = !no_opaque && ((guest & 1u) != 0 || ((triangle.word >> 30u) & 1u) != 0);
	bool       alpha_off = params.opacity == Opacity::AllOpaque;
	if (params.opacity == Opacity::Live) alpha_off = live(params.alpha_off) != 0;
	if (!alpha_off && !opaque) return false;
	if (!params.both) {
		const bool enabled = params.cull == NoSlot || live(params.cull) != 0;
		if (enabled && !cull_off) {
			const bool front      = candidate.front != flipped;
			const bool flip_ray   = params.flip != NoSlot && live(params.flip) != 0;
			const bool culled     = flip_ray ? front : !front;
			if (culled) return false;
		}
	}
	return true;
}

std::optional<size_t> Analysis::HostClosest(const Params& params, const Observation& observation) const {
	std::optional<size_t> best;
	for (size_t index = 0; index < observation.candidates.size(); ++index) {
		const auto& candidate = observation.candidates[index];
		const auto& triangle  = observation.triangles[candidate.triangle];
		if (!HostAccepts(params, observation.instances[triangle.instance], triangle, candidate, observation.lives)) continue;
		if (!best || candidate.t < observation.candidates[*best].t) best = index;
	}
	return best;
}

const char* Analysis::Search() {
	std::vector<bool> accepted;
	for (const auto& row: m_rows) {
		bool any = false;
		for (size_t index = 0; index < m_outs.size(); ++index) any = any || row.outs[index] != row.base[index];
		accepted.push_back(any);
	}
	std::vector<uint32_t> lives {NoSlot};
	for (uint32_t index = 0; index < m_lives.size(); ++index) lives.push_back(index);
	struct Flags {
		bool     on    = false;
		uint32_t dword = 0;
		uint32_t shift = 0;
		bool     flip  = false;
	};
	std::vector<Flags> flags;
	if (m_layout.flags) {
		flags.push_back({});
	} else {
		flags.push_back({});
		for (const auto dword: m_flag_dwords) {
			for (uint32_t shift = 0; shift <= 28; ++shift) {
				for (uint32_t flip = 0; flip < 2; ++flip) flags.push_back({true, dword, shift, flip != 0});
			}
		}
	}
	const auto matches = [&](const Params& params) {
		for (size_t index = 0; index < m_rows.size(); ++index) {
			const auto& row       = m_rows[index];
			const auto& candidate = row.candidates.front();
			const auto& triangle  = row.triangles[candidate.triangle];
			if (HostAccepts(params, row.instances[triangle.instance], triangle, candidate, row.lives) != accepted[index]) return false;
		}
		return true;
	};
	struct Mask {
		bool     on    = false;
		uint32_t dword = 0;
		uint32_t shift = 0;
	};
	std::vector<Mask> masks {{}};
	if (!m_layout.mask) {
		for (const auto dword: m_flag_dwords) {
			for (uint32_t shift = 0; shift <= 24; ++shift) masks.push_back({true, dword, shift});
		}
	}
	for (const auto& mask_option: masks) {
		for (const auto ray_mask: lives) {
			bool plausible = true;
			for (size_t index = 0; index < m_rows.size() && plausible; ++index) {
				if (!accepted[index]) continue;
				const auto& row      = m_rows[index];
				const auto& instance = row.instances[row.triangles[row.candidates.front().triangle].instance];
				uint32_t    bits     = 0xffu;
				if (mask_option.on) {
					bits = (instance.words[mask_option.dword] >> mask_option.shift) & 0xffu;
				} else if (m_layout.mask) {
					const auto field = instance.words[m_layout.mask_dword] >> m_layout.mask_shift;
					bits = m_layout.mask_width == 0 ? field & 0xffu : (field & ((1u << m_layout.mask_width) - 1u)) != 0 ? 0xffu : 0u;
				}
				const uint32_t ray = ray_mask == NoSlot ? 0xffu : static_cast<uint32_t>(row.lives[ray_mask]) & 0xffu;
				plausible = (bits & ray) != 0;
			}
			if (!plausible) continue;
			for (const auto& flag: flags) {
				for (uint32_t opacity = 0; opacity < 2 + m_lives.size(); ++opacity) {
					for (uint32_t both = 0; both < 2; ++both) {
						for (const auto cull: lives) {
							for (const auto flip: lives) {
								Params params;
								params.flags       = flag.on;
								params.flags_dword = flag.dword;
								params.flags_shift = flag.shift;
								params.flags_flip  = flag.flip;
								params.mask        = mask_option.on;
								params.mask_dword  = mask_option.dword;
								params.mask_shift  = mask_option.shift;
								params.ray_mask    = ray_mask;
								params.both        = both != 0;
								params.cull        = cull;
								params.flip        = flip;
								params.opacity     = opacity == 0 ? Opacity::AllOpaque : opacity == 1 ? Opacity::OnlyOpaque : Opacity::Live;
								params.alpha_off   = opacity >= 2 ? opacity - 2u : NoSlot;
								if (params.both && (cull != NoSlot || flip != NoSlot)) continue;
								if (matches(params)) {
									m_params = params;
									return nullptr;
								}
							}
						}
					}
				}
			}
		}
	}
	return "no host ray query applies the hit test of the walk";
}

const char* Analysis::Validate() {
	for (uint32_t round = 0; round < 2 * m_seeds.size(); ++round) {
		const auto  which = round % m_seeds.size();
		const auto  seed  = m_seeds[which];
		const auto& start = m_starts[which];
		Random      random(seed * 0x2000005u + round + 0x77u);
		std::vector<std::pair<uint32_t, std::vector<uint64_t>>> overrides;
		Begin(seed);
		const Block* from = nullptr;
		if (const char* failure = m_machine.RunPrologue(from)) return failure;
		World world = MakeWorld(start, random);
		for (size_t live = 0; live < m_lives.size(); ++live) {
			std::vector<uint64_t> values(m_machine.Wave());
			for (uint32_t lane = 0; lane < m_machine.Wave(); ++lane) {
				values[lane] = m_machine.At(m_lives[live].slot, lane);
				if (random.Below(2) != 0) values[lane] = m_lives[live].flag ? random.Below(2) : (random.Below(3) == 0 ? 0u : random.Next() & 0xffu);
			}
			overrides.emplace_back(static_cast<uint32_t>(live), std::move(values));
		}
		std::vector<uint32_t> shared;
		for (size_t at = 0; at < start.lanes.size(); ++at) {
			const auto lane   = start.lanes[at];
			const auto extent = start.extents[at];
			if (!Usable(extent)) continue;
			const auto count = random.Below(4);
			for (uint32_t k = 0; k < count; ++k) {
				Candidate candidate;
				if (!shared.empty() && random.Below(3) == 0) {
					candidate.triangle = shared[random.Below(static_cast<uint32_t>(shared.size()))];
				} else {
					const auto type     = m_types[random.Below(static_cast<uint32_t>(m_types.size()))];
					const auto instance = AddInstance(world, random, type, m_layout.mask ? random.Below(256) : 0xffu, random.Below(16), true);
					candidate.triangle  = AddTriangle(world, random, instance, random.Below(2) != 0);
					shared.push_back(candidate.triangle);
				}
				const auto step  = Fraction(random, 64);
				candidate.t      = extent * static_cast<float>(step) / 64.0f;
				if (random.Below(6) == 0) candidate.t = extent * 1.5f;
				candidate.inside = candidate.t <= extent;
				candidate.front  = random.Below(2) != 0;
				candidate.den    = (candidate.front ? 1.0f : -1.0f) * ((random.Below(2) != 0) ? 2.0f : 1.0f);
				candidate.u      = static_cast<float>(random.Below(128)) / 256.0f;
				candidate.v      = static_cast<float>(random.Below(128)) / 256.0f;
				bool same_t = std::ranges::any_of(world.lanes[lane].candidates, [&](const Candidate& other) { return other.t == candidate.t || other.triangle == candidate.triangle; });
				if (!same_t) world.lanes[lane].candidates.push_back(candidate);
			}
		}
		World base = world;
		for (auto& lane: base.lanes) lane.candidates.clear();
		std::vector<std::vector<uint64_t>> base_outs, outs;
		if (const char* failure = Play(start, base, base_outs, overrides)) return failure;
		if (const char* failure = Play(start, world, outs, overrides)) return failure;
		for (uint32_t lane = 0; lane < m_machine.Wave(); ++lane) {
			Observation observation;
			observation.candidates = world.lanes[lane].candidates;
			observation.instances  = world.instances;
			observation.triangles  = world.triangles;
			for (size_t live = 0; live < m_lives.size(); ++live) observation.lives.push_back(overrides[live].second[lane] & 0xffffffffu);
			const auto best = observation.candidates.empty() ? std::nullopt : HostClosest(m_params, observation);
			const auto values = best ? KindValues(world, observation.candidates[*best]) : std::array<uint64_t, KindCount> {};
			for (size_t index = 0; index < m_outs.size(); ++index) {
				const auto expected = best && m_kinds[index] ? (m_outs[index].flag ? 1u : Transformed(values[*m_kinds[index]], m_transforms[index])) : base_outs[index][lane];
				if (outs[index][lane] != expected) {
					detail = static_cast<uint32_t>(index) << 8u | lane;
					return "a wave with several candidates per lane ends on another hit than the host";
				}
			}
		}
	}
	return nullptr;
}

const char* Analysis::Hoist(const Start& start) {
	const uint32_t words = start.inst->Flags<uint32_t>() & 0xffu;
	const auto* descriptor = start.inst->Arg(0).Resolve().TryInstruction();
	if (descriptor == nullptr || descriptor->GetOpcode() != Op::CompositeConstructU32x4) return "the top-level descriptor is not built from words";
	for (const auto& other: m_starts) {
		if (other.path != start.path) return "the first round of the walk takes another path in another world";
	}
	std::unordered_set<const Block*> seen;
	for (const auto* block: start.path) {
		if (!seen.insert(block).second) return "the first round of the walk loops before its first intersection";
	}
	std::unordered_set<const Inst*> visited;
	std::function<bool(Value)>      peelable = [&](Value value) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || !m_machine.Inside(inst->Parent()) || !visited.insert(inst).second) return true;
		const auto op = inst->GetOpcode();
		if (op == Op::Phi) {
			const auto where = std::ranges::find(start.path, inst->Parent());
			if (where == start.path.end()) return false;
			const Block* from = where == start.path.begin() ? m_preheader : *(where - 1);
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				if (inst->PhiBlock(index) == from) return peelable(inst->Arg(index));
			}
			return false;
		}
		if (op == Op::BvhIntersect || HasSideEffects(op) || op == Op::MakeImageAddress || TypeOf(op) == Type::Void) return false;
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			if (!peelable(inst->Arg(index))) return false;
		}
		return true;
	};
	const auto* ray = start.inst->Arg(1).Resolve().TryInstruction();
	for (uint32_t index = 0; index < 7; ++index) {
		if (!peelable(ray->Arg(words + index))) return "a ray input of the first round cannot be computed before the walk";
	}
	if (!peelable(start.inst->Arg(2))) return "the active lanes of the first round cannot be computed before the walk";
	for (uint32_t index = 0; index < words; ++index) {
		if (!peelable(ray->Arg(index))) return "the first node of the walk cannot be computed before it";
	}
	if (!peelable(descriptor->Arg(0)) || !peelable(descriptor->Arg(1))) return "the top-level descriptor cannot be computed before the walk";
	return nullptr;
}

Value Analysis::Peel(Value value, std::unordered_map<const Inst*, Value>& peeled, bool& failed) {
	value = value.Resolve();
	auto* inst = value.TryInstruction();
	if (inst == nullptr || !m_machine.Inside(inst->Parent())) return value;
	if (const auto at = peeled.find(inst); at != peeled.end()) return at->second;
	const auto& path = m_starts.front().path;
	const auto  op   = inst->GetOpcode();
	Value       result;
	if (op == Op::Phi) {
		const auto where = std::ranges::find(path, inst->Parent());
		if (where == path.end()) {
			failed = true;
			return {};
		}
		const Block* from = where == path.begin() ? m_preheader : *(where - 1);
		Value        chosen;
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			if (inst->PhiBlock(index) == from) chosen = inst->Arg(index);
		}
		if (chosen.IsEmpty()) {
			failed = true;
			return {};
		}
		result = Peel(chosen, peeled, failed);
	} else if (op == Op::BvhIntersect || HasSideEffects(op) || op == Op::MakeImageAddress || TypeOf(op) == Type::Void) {
		failed = true;
		return {};
	} else {
		std::vector<Value> args;
		for (size_t index = 0; index < inst->NumArgs(); ++index) args.push_back(Peel(inst->Arg(index), peeled, failed));
		if (failed) return {};
		auto& copy = *m_preheader->Instructions().emplace(m_preheader->end(), op, inst->Flags<uint64_t>());
		copy.SetParent(m_preheader);
		for (size_t index = 0; index < args.size(); ++index) copy.SetArg(index, args[index]);
		const bool memory = op == Op::ReadConstBuffer || op == Op::LoadAddressU8 || op == Op::LoadAddressU16 || op == Op::LoadAddressU32 ||
		                    op == Op::LoadBufferU8 || op == Op::LoadBufferU16 || op == Op::LoadBufferU32 || op == Op::LoadBufferU32x2 ||
		                    op == Op::LoadBufferU32x3 || op == Op::LoadBufferU32x4;
		if (memory) {
			auto flags = inst->Flags<MemoryFlags>();
			m_program.memory_info.push_back(m_program.memory_info[flags.index]);
			flags.index = static_cast<uint32_t>(m_program.memory_info.size() - 1u);
			copy.SetFlags(flags);
		}
		result = Value(&copy);
	}
	peeled.emplace(inst, result);
	return result;
}

const char* Analysis::ForcedPath() {
	const auto* merge = m_machine.Merge();
	std::unordered_map<const Block*, const Block*> parent;
	std::vector<const Block*> queue {m_header};
	parent[m_header] = nullptr;
	const auto clean = [&](const Block* block) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == Op::BvhIntersect || (HasSideEffects(op) && op != Op::Reference && op != Op::ReferenceU32)) return false;
		}
		return true;
	};
	const Block* last = nullptr;
	for (size_t at = 0; at < queue.size() && last == nullptr; ++at) {
		const auto* block = queue[at];
		if (!clean(block)) continue;
		for (auto* next: block->ImmSuccessors()) {
			if (next == merge) {
				last = block;
				break;
			}
			if (!m_machine.Inside(next) || parent.contains(next) || next == m_header) continue;
			parent[next] = block;
			queue.push_back(next);
		}
	}
	if (last == nullptr) return "no path leaves the walk without an intersection";
	std::vector<const Block*> path;
	for (const auto* block = last; block != nullptr; block = parent[block]) path.push_back(block);
	std::ranges::reverse(path);
	std::unordered_map<const Block*, const Block*> forced;
	for (size_t at = 0; at < path.size(); ++at) {
		const auto* block = path[at];
		const auto* next  = at + 1 < path.size() ? path[at + 1] : merge;
		forced[block]     = next;
		if (block->terminator.kind != CFG::TerminatorKind::ConditionalBranch) continue;
		if (block->condition.IsEmpty() || block->terminator.true_block == block->terminator.false_block) return "a branch on the exit path cannot be forced";
		m_path.emplace_back(const_cast<Block*>(block), const_cast<Block*>(next));
	}
	std::vector<std::vector<std::vector<uint64_t>>> naturals, forceds;
	for (size_t which = 0; which < m_seeds.size(); ++which) {
		const auto seed = m_seeds[which];
		Begin(seed);
		const Block* from = nullptr;
		if (const char* failure = m_machine.RunPrologue(from)) return failure;
		World world;
		world.header = m_starts[which].header;
		world.root   = m_result.root;
		world.lanes.assign(m_machine.Wave(), {});
		std::vector<std::vector<uint64_t>> base_outs, outs;
		if (const char* failure = Play(m_starts[which], world, base_outs, {})) return failure;
		m_machine.Force(&forced);
		const char* failure = Play(m_starts[which], world, outs, {}, false);
		m_machine.Force(nullptr);
		if (failure != nullptr) return "the exit path runs an intersection";
		naturals.push_back(std::move(base_outs));
		forceds.push_back(std::move(outs));
	}
	m_fixed.assign(m_outs.size(), std::nullopt);
	for (size_t index = 0; index < m_outs.size(); ++index) {
		bool differs = false;
		for (size_t which = 0; which < m_seeds.size(); ++which) {
			for (uint32_t lane = 0; lane < m_machine.Wave(); ++lane) {
				if (forceds[which][index][lane] == naturals[which][index][lane]) continue;
				const bool active = std::ranges::find(m_starts[which].lanes, lane) != m_starts[which].lanes.end();
				if (!active || m_kinds[index]) {
					detail = static_cast<uint32_t>(index);
					return "the exit path changes a value of the walk";
				}
				differs = true;
			}
		}
		if (!differs) continue;
		const auto constant = naturals.front()[index][m_starts.front().lanes.front()];
		for (size_t which = 0; which < m_seeds.size(); ++which) {
			for (const auto lane: m_starts[which].lanes) {
				if (naturals[which][index][lane] != constant) {
					detail = static_cast<uint32_t>(index);
					return "a value the walk leaves without a hit is not a constant";
				}
			}
		}
		m_fixed[index] = constant;
	}
	return nullptr;
}

const char* Analysis::Derive() {
	if (const char* failure = Prepare()) return failure;
	for (uint64_t seed = 1; seed <= MaxSeeds && m_seeds.size() < 3; ++seed) {
		if (seed > 3 && m_seeds.empty() && m_machine.Work() > TotalWork / 16u) break;
		Start start;
		const char* seed_failure = Seed(seed, start);
		if (seed_failure != nullptr) {
			continue;
		}
		uint32_t usable = 0;
		for (const auto extent: start.extents) usable += Usable(extent) ? 1u : 0u;
		if (usable < 2) continue;
		if (!m_starts.empty() && start.inst != m_starts.front().inst) return "the walk starts at another place in another world";
		const char* trial_failure = Trial(seed, start);
		if (trial_failure != nullptr) {
			continue;
		}
		m_seeds.push_back(seed);
		m_starts.push_back(std::move(start));
	}
	if (m_seeds.size() < 2) return "fewer than two worlds start the walk with active lanes";
	if (const char* failure = Discover(m_seeds.front(), m_starts.front())) return failure;
	if (const char* failure = Rows()) return failure;
	if (const char* failure = Classify()) return failure;
	if (const char* failure = Search()) return failure;
	if (const char* failure = Validate()) return failure;
	if (const char* failure = Hoist(m_starts.front())) return failure;
	return ForcedPath();
}

struct At {
	Block*          block;
	Block::iterator where;

	Value operator()(Op opcode, std::initializer_list<Value> args, uint64_t flags = 0) const {
		auto& inst = *block->PrependNewInst(where, opcode, args, flags);
		return TypeOf(opcode) == Type::Void ? Value {} : Value(&inst);
	}
};

Block::iterator AfterPhis(Block& block) {
	return std::ranges::find_if(block, [](const Inst& inst) { return inst.GetOpcode() != Op::Phi; });
}

void Analysis::Emit(uint32_t site) {
	const auto flags = [site](uint32_t variant) { return RayQueryFlags(site, variant); };
	const At   entry {m_preheader, m_preheader->end()};
	const auto word  = [&](uint32_t live) {
		const auto value = Value(m_lives[live].inst);
		return m_lives[live].flag ? entry(Op::SelectU32, {value, Value(1u), Value(0u)}) : value;
	};
	const auto truth = [&](uint32_t live, bool otherwise) {
		if (live == NoSlot) return Value(otherwise);
		const auto value = Value(m_lives[live].inst);
		return m_lives[live].flag ? value : entry(Op::INotEqual32, {value, Value(0u)});
	};
	{
		const auto&                            start      = m_starts.front();
		const uint32_t                         words      = start.inst->Flags<uint32_t>() & 0xffu;
		const auto*                            ray        = start.inst->Arg(1).Resolve().TryInstruction();
		const auto*                            descriptor = start.inst->Arg(0).Resolve().TryInstruction();
		std::unordered_map<const Inst*, Value> peeled;
		bool                                   failed = false;
		for (uint32_t index = 0; index < 6; ++index) m_ray[index] = Peel(ray->Arg(words + 1u + index), peeled, failed);
		m_extent        = Peel(ray->Arg(words), peeled, failed);
		m_lanes         = Peel(start.inst->Arg(2), peeled, failed);
		m_descriptor[0] = Peel(descriptor->Arg(0), peeled, failed);
		m_descriptor[1] = Peel(descriptor->Arg(1), peeled, failed);
		m_node[0]       = Peel(ray->Arg(0), peeled, failed);
		m_node[1]       = words == 2u ? Peel(ray->Arg(1), peeled, failed) : Value(0u);
	}
	const auto bits = [&](Value value) {
		if (value.IsImmediate()) return value.GetType() == Type::F32 ? Value(std::bit_cast<uint32_t>(value.F32Value())) : value;
		return value.GetType() == Type::F32 ? entry(Op::BitCastU32F32, {value}) : value;
	};
	for (auto& part: m_ray) part = bits(part);
	m_extent      = bits(m_extent);
	m_descriptor  = {bits(m_descriptor[0]), bits(m_descriptor[1])};
	const auto high   = entry(Op::BitwiseAnd32, {m_descriptor[1], Value(0xffu)});
	auto       header = entry(Op::ShiftLeftLogical64, {entry(Op::CompositeConstructU64, {m_descriptor[0], high}), Value(8u)});
	if (!(m_node[0].IsImmediate() && m_node[0].U32() == m_result.root && m_node[1].IsImmediate() && m_node[1].U32() == 0u)) {
		const auto node    = entry(Op::CompositeConstructU64, {m_node[0], m_node[1]});
		const auto address = entry(Op::IAdd64, {header, entry(Op::ShiftLeftLogical64, {entry(Op::ShiftRightLogical64, {node, Value(3u)}), Value(6u)})});
		header             = entry(Op::ISub64, {address, Value(static_cast<uint64_t>(m_result.root >> 3u) << 6u)});
	}
	const auto low_word  = entry(Op::CompositeExtractU64, {header, Value(0u)});
	const auto high_word = entry(Op::CompositeExtractU64, {header, Value(1u)});
	const auto mask      = m_params.ray_mask == NoSlot ? Value(0xffu) : word(m_params.ray_mask);
	const auto alpha_off = m_params.opacity == Opacity::AllOpaque ? Value(1u) : m_params.opacity == Opacity::OnlyOpaque ? Value(0u) : word(m_params.alpha_off);
	const uint32_t variant = (m_params.both ? RayQueryInitBothFacings : 0u) | (m_params.opacity != Opacity::AllOpaque ? RayQueryInitOpaqueOnly : 0u);
	const auto native = entry(Op::RayQueryInit,
	                          {m_ray[0], m_ray[1], m_ray[2], m_ray[3], m_ray[4], m_ray[5], m_extent, mask, low_word, high_word, alpha_off,
	                           m_lanes, truth(m_params.cull, true), truth(m_params.flip, false), Value(true)},
	                          flags(variant));
	const auto ballot = entry(Op::Ballot, {native});
	const auto any    = entry(Op::INotEqual32, {entry(Op::BitwiseOr32, {entry(Op::CompositeExtractU32x4, {ballot, Value(0u)}),
	                                                                   entry(Op::CompositeExtractU32x4, {ballot, Value(1u)})}),
	                                              Value(0u)});
	entry(Op::RayQueryNext, {}, flags(0u));
	for (const auto& [block, next]: m_path) {
		const At   in {block, block->end()};
		const auto forced = in(Op::SelectU1, {any, Value(next == block->terminator.true_block), block->condition});
		const auto ref    = in(Op::ConditionRef, {forced}, static_cast<uint64_t>(CFG::BranchCondition::ScalarInstruction));
		in(Op::Reference, {ref});
		block->condition = ref;
	}
	auto*    merge = const_cast<Block*>(m_machine.Merge());
	const At after {merge, AfterPhis(*merge)};
	const auto result = [&](RayQueryResultKind kind) { return after(Op::RayQueryResult, {}, flags(static_cast<uint32_t>(kind))); };
	const auto has    = after(Op::INotEqual32, {result(RayQueryResultKind::Has), Value(0u)});
	const auto commit = after(Op::LogicalAnd, {has, after(Op::FPOrdLessThanEqual32, {after(Op::BitCastF32U32, {result(RayQueryResultKind::T)}),
	                                                                                  after(Op::BitCastF32U32, {m_extent})})});
	for (size_t index = 0; index < m_outs.size(); ++index) {
		if (!m_kinds[index]) continue;
		auto&      out    = m_outs[index];
		auto       value  = out.flag ? Value(true) : result(HostKinds[*m_kinds[index]]);
		const auto [shift, mask] = Transforms[m_transforms[index]];
		if (!out.flag && shift != 0) value = after(Op::ShiftRightLogical32, {value, Value(shift)});
		if (!out.flag && mask != 0xffffffffu) value = after(Op::BitwiseAnd32, {value, Value(mask)});
		const auto chosen = out.flag ? after(Op::SelectU1, {commit, Value(true), Value(out.inst)}) : after(Op::SelectU32, {commit, value, Value(out.inst)});
		const auto uses   = out.inst->Uses();
		for (const auto& use: uses) {
			if (use.user == chosen.Instruction()) continue;
			if (!out.phi && (m_machine.Inside(use.user->Parent()) || (use.user->Parent() == merge && use.user->GetOpcode() == Op::Phi))) continue;
			use.user->SetArg(use.operand, chosen);
		}
	}
	const auto served = after(Op::LogicalAnd, {any, m_lanes});
	for (size_t index = 0; index < m_outs.size(); ++index) {
		if (!m_fixed[index]) continue;
		auto&      out    = m_outs[index];
		const auto chosen = out.flag ? after(Op::SelectU1, {served, Value(*m_fixed[index] != 0), Value(out.inst)})
		                             : after(Op::SelectU32, {served, Value(static_cast<uint32_t>(*m_fixed[index])), Value(out.inst)});
		const auto uses   = out.inst->Uses();
		for (const auto& use: uses) {
			if (use.user == chosen.Instruction()) continue;
			if (!out.phi && (m_machine.Inside(use.user->Parent()) || (use.user->Parent() == merge && use.user->GetOpcode() == Op::Phi))) continue;
			use.user->SetArg(use.operand, chosen);
		}
	}
	after(Op::RayQueryMark, {}, flags(static_cast<uint32_t>(RayQueryMarkKind::Exit)));
	for (auto* bvh: m_served) bvh->SetFlags(bvh->Flags<uint32_t>() | BvhIntersectNative);
	if (m_params.mask) {
		auto& layout = m_result.layout;
		layout[2] |= (m_params.mask_dword & 15u) | (m_params.mask_shift & 31u) << 4u | 1u << 19u;
		layout[3] |= (m_params.mask_dword >> 4u & 3u) << 24u;
	}
	if (m_params.flags) {
		auto& layout = m_result.layout;
		layout[2] |= (m_params.flags_dword & 15u) << 9u | 1u << 13u | (m_params.flags_shift & 31u) << 14u;
		layout[3] |= (m_params.flags_dword >> 4u & 3u) << 26u;
		layout[1] = (layout[1] & ~(1u << 29u)) | (m_params.flags_flip ? 1u << 29u : 0u);
	}
	auto source           = m_result;
	source.oracle_sources = {};
	source.oracle_improvements = {};
	m_result.oracle_sources      = {source};
	m_result.oracle_improvements = {};
}

}

bool RewriteResultsWalk(Program& program, const NativeTraversalOptions& options, Block* header, Block* latch,
                        std::span<Inst* const> served, NativeTraversals& result, uint32_t site) {
	const bool compute = program.stage == ShaderType::Compute;
	if ((options.oracle & OracleLoop) != 0 || (options.oracle & (OracleForms | OracleOnly | OracleResults)) == 0 || site >= MaxRayQuerySites ||
	    served.empty() || header == nullptr || latch == nullptr) {
		return false;
	}
	Analysis    analysis(program, options, header, latch, served, result);
	const char* failure = analysis.Derive();
	if (failure != nullptr) {
		LOGF("%s universal ray tracing: hash=0x%016llx keeps the oracle loop: %s (%u)\n", compute ? "CS" : "PS",
		     static_cast<unsigned long long>(program.shader_hash), failure, analysis.detail);
		return false;
	}
	analysis.Emit(site);
	LOGF("%s universal ray tracing: hash=0x%016llx answers %zu intersections with host results\n", compute ? "CS" : "PS",
	     static_cast<unsigned long long>(program.shader_hash), served.size());
	return true;
}

}
