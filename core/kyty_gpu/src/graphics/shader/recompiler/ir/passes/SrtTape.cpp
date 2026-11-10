#include "graphics/shader/recompiler/ir/passes/SrtTape.h"

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "common/shaderEnv.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {

bool SrtTapeEnabled() {
	static const bool enabled = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_FIX_SRT_TAPE");
		// Default on: 4.0M verified materializations, 0 mismatches; flyover 2.84 -> 3.22
		// fps. "0" turns it off.
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

bool SrtTapeVerifyEnabled() {
	static const bool enabled = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_FIX_SRT_TAPE_VERIFY");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

bool SrtClassTrace() {
	static const bool enabled = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_TRACE_SRT_CLASSES");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) std::fprintf(stderr, "TRACE_SRT_CLASSES enabled\n");
		return on;
	}();
	return enabled;
}

namespace {
// The node operands SrtTapeWalker::Run evaluates (RawRead: handle, offset and the vector guard; args[3] is the
// fusion group, not a node).
uint32_t SrtOperandCount(const SrtNode& node) {
	switch (node.op) {
		case SrtOp::Imm:
		case SrtOp::Fail:
		case SrtOp::UserData:
		case SrtOp::ShaderBase: return 0;
		case SrtOp::Alias:
		case SrtOp::ExtractU64:
		case SrtOp::ReadConst:
		case SrtOp::ConvertF32U32:
		case SrtOp::ConvertU32F32:
		case SrtOp::FPTrunc32:
		case SrtOp::FPRecipIFlag32:
		case SrtOp::FPIsNan32:
		case SrtOp::Not32:
		case SrtOp::LogicalNot: return 1;
		case SrtOp::BitFieldUExtract:
		case SrtOp::BitFieldSExtract:
		case SrtOp::Select: return 3;
		case SrtOp::BitFieldInsert: return 4;
		case SrtOp::RawRead: return (node.flags & SrtNode::FlagVector) != 0u ? 3 : 2;
		case SrtOp::Handle: return (node.flags & SrtNode::FlagHandle4) != 0u ? 4 : 2;
		default: return 2;
	}
}

// KYTY_TRACE_SRT_CLASSES: marks every node whose value the CPU needs (reachable from a descriptor root, an indirect
// key/selector value, a block condition or a fill root, or from the address operands of any read).
void ClassifySrtReads(SrtTape& tape) {
	const auto n = static_cast<uint32_t>(tape.nodes.size());
	tape.cpu_needed.assign(n, 0u);
	std::vector<uint32_t> pending;
	const auto push = [&](uint32_t id) {
		if (id < n && tape.cpu_needed[id] == 0u) {
			tape.cpu_needed[id] = 1u;
			pending.push_back(id);
		}
	};
	for (const auto root: tape.descriptor_roots) push(root);
	for (const auto& values: tape.indirect_values) {
		for (const auto root: values) push(root);
	}
	for (const auto root: tape.block_conditions) push(root);
	for (const auto root: tape.fill_roots) push(root);
	for (uint32_t id = 0; id < n; ++id) {
		const auto& node = tape.nodes[id];
		if (node.op != SrtOp::RawRead) continue;
		for (uint32_t arg = 0; arg < SrtOperandCount(node); ++arg) push(node.args[arg]);
	}
	while (!pending.empty()) {
		const auto& node = tape.nodes[pending.back()];
		pending.pop_back();
		for (uint32_t arg = 0; arg < SrtOperandCount(node); ++arg) push(node.args[arg]);
	}
}
} // namespace

void MergeSrtReads(SrtReadRanges& reads) {
	size_t kept = 0;
	for (size_t index = 0; index < reads.size(); ++index) {
		if (kept != 0u && reads[index].second != 0u &&
		    reads[kept - 1u].first + reads[kept - 1u].second == reads[index].first) {
			reads[kept - 1u].second += reads[index].second;
		} else {
			reads[kept++] = reads[index];
		}
	}
	reads.resize(kept);
}

namespace {
// One counter per scratch slot, each written by the thread that owns the slot (plain load and store, no
// locked add) and read by SrtTapeFusionStats from any thread.
std::atomic<uint64_t> g_fused_runs[ResourceScratchSlots];
std::atomic<uint64_t> g_fused_dwords[ResourceScratchSlots];

void CountFusedRun(uint64_t dwords) {
	const auto slot = t_resource_scratch_slot;
	g_fused_runs[slot].store(g_fused_runs[slot].load(std::memory_order_relaxed) + 1u, std::memory_order_relaxed);
	g_fused_dwords[slot].store(g_fused_dwords[slot].load(std::memory_order_relaxed) + dwords,
	                           std::memory_order_relaxed);
}
} // namespace

void SrtTapeFusionStats(uint64_t& runs, uint64_t& dwords) {
	runs   = 0;
	dwords = 0;
	for (uint32_t slot = 0; slot < ResourceScratchSlots; ++slot) {
		runs += g_fused_runs[slot].load(std::memory_order_relaxed);
		dwords += g_fused_dwords[slot].load(std::memory_order_relaxed);
	}
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

float Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

// Same predicate as the walker's: a scalar/uniform vector DWORD read the host can evaluate.
bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

enum class Reason : uint32_t { None, Indirect, ReadFirstLane, Phi, Opcode, Cycle, Shape, Count };

constexpr const char* ReasonNames[] = {"none", "indirect", "read_first_lane", "phi", "opcode", "cycle", "shape"};

struct OpInfo {
	SrtOp   op;
	uint8_t count;
};

// Plain operand-only opcodes; the evaluator mirrors SrtWalker::EvaluateInst for each.
OpInfo OpInfoOf(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConditionRef: return {SrtOp::Alias, 1};
		case ValueOpcode::CompositeConstructU64: return {SrtOp::ConstructU64, 2};
		case ValueOpcode::IAdd32: return {SrtOp::IAdd32, 2};
		case ValueOpcode::IAdd64: return {SrtOp::IAdd64, 2};
		case ValueOpcode::ISub32: return {SrtOp::ISub32, 2};
		case ValueOpcode::ISub64: return {SrtOp::ISub64, 2};
		case ValueOpcode::IMul32: return {SrtOp::IMul32, 2};
		case ValueOpcode::UMulHi: return {SrtOp::UMulHi, 2};
		case ValueOpcode::UMin32: return {SrtOp::UMin32, 2};
		case ValueOpcode::ConvertF32U32: return {SrtOp::ConvertF32U32, 1};
		case ValueOpcode::ConvertU32F32: return {SrtOp::ConvertU32F32, 1};
		case ValueOpcode::FPMul32: return {SrtOp::FPMul32, 2};
		case ValueOpcode::FPTrunc32: return {SrtOp::FPTrunc32, 1};
		case ValueOpcode::FPRecipIFlag32: return {SrtOp::FPRecipIFlag32, 1};
		case ValueOpcode::FPIsNan32: return {SrtOp::FPIsNan32, 1};
		case ValueOpcode::FPOrdLessThanEqual32: return {SrtOp::FPLessEqual, 2};
		case ValueOpcode::FPOrdGreaterThanEqual32: return {SrtOp::FPGreaterEqual, 2};
		case ValueOpcode::BitwiseAnd32: return {SrtOp::And32, 2};
		case ValueOpcode::BitwiseAnd64: return {SrtOp::And64, 2};
		case ValueOpcode::BitwiseOr32: return {SrtOp::Or32, 2};
		case ValueOpcode::BitwiseXor32: return {SrtOp::Xor32, 2};
		case ValueOpcode::BitwiseNot32: return {SrtOp::Not32, 1};
		case ValueOpcode::ShiftLeftLogical32: return {SrtOp::Shl32, 2};
		case ValueOpcode::ShiftLeftLogical64: return {SrtOp::Shl64, 2};
		case ValueOpcode::ShiftRightLogical32: return {SrtOp::Shr32, 2};
		case ValueOpcode::ShiftRightLogical64: return {SrtOp::Shr64, 2};
		case ValueOpcode::ShiftRightArithmetic32: return {SrtOp::Sar32, 2};
		case ValueOpcode::ShiftRightArithmetic64: return {SrtOp::Sar64, 2};
		case ValueOpcode::BitFieldUExtract: return {SrtOp::BitFieldUExtract, 3};
		case ValueOpcode::BitFieldSExtract: return {SrtOp::BitFieldSExtract, 3};
		case ValueOpcode::BitFieldInsert: return {SrtOp::BitFieldInsert, 4};
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: return {SrtOp::Select, 3};
		case ValueOpcode::IEqual32: return {SrtOp::IEqual, 2};
		case ValueOpcode::INotEqual32: return {SrtOp::INotEqual, 2};
		case ValueOpcode::ULessThan32: return {SrtOp::ULess, 2};
		case ValueOpcode::ULessThanEqual32: return {SrtOp::ULessEqual, 2};
		case ValueOpcode::UGreaterThan32: return {SrtOp::UGreater, 2};
		case ValueOpcode::SGreaterThanEqual32: return {SrtOp::SGreaterEqual, 2};
		case ValueOpcode::LogicalAnd: return {SrtOp::LogicalAnd, 2};
		case ValueOpcode::LogicalOr: return {SrtOp::LogicalOr, 2};
		case ValueOpcode::LogicalXor: return {SrtOp::LogicalXor, 2};
		case ValueOpcode::LogicalNot: return {SrtOp::LogicalNot, 1};
		default: return {SrtOp::Fail, 0};
	}
}

class Builder {
public:
	Builder(const ResourcePlan& plan, SrtTape& tape): m_plan(plan), m_tape(tape) {}

	Reason      Run();
	ValueOpcode Detail() const { return m_detail; }

private:
	static constexpr uint32_t InProgress = UINT32_MAX;

	void     Reject(Reason reason, ValueOpcode detail = ValueOpcode::Void);
	uint32_t Push(const SrtNode& node);
	uint32_t Build(Value value);
	uint32_t BuildImmediate(Value value);
	uint32_t BuildInst(const Inst& inst);
	uint32_t BuildExtract(const Inst& inst);
	uint32_t BuildReadConst(const Inst& inst);
	uint32_t BuildRawRead(const Inst& inst);
	uint32_t SlotRoot(uint32_t slot);

	const ResourcePlan&                       m_plan;
	SrtTape&                                  m_tape;
	Reason                                    m_reason = Reason::None;
	ValueOpcode                               m_detail = ValueOpcode::Void;
	std::unordered_map<const Inst*, uint32_t> m_nodes;
	std::unordered_map<uint64_t, uint32_t>    m_immediates;
	std::vector<uint8_t>                      m_slot_state;
	// Scalar RawReads by {Handle flags and operands, offset operand, RawRead flags}: the fusion groups.
	std::map<std::array<uint32_t, 7>, uint32_t> m_groups;
};

void Builder::Reject(Reason reason, ValueOpcode detail) {
	if (m_reason == Reason::None) {
		m_reason = reason;
		m_detail = detail;
	}
}

uint32_t Builder::Push(const SrtNode& node) {
	m_tape.nodes.push_back(node);
	return static_cast<uint32_t>(m_tape.nodes.size() - 1u);
}

Reason Builder::Run() {
	const auto direct_source = [&](uint32_t index) {
		// An out-of-range index fails at run time in both evaluators.
		return index >= m_plan.descriptor_sources.size() ||
		       !m_plan.descriptor_sources[index].indirect_descriptor.has_value();
	};
	for (const auto& source: m_plan.descriptor_sources) {
		if (source.dword_count > source.dwords.size()) {
			Reject(Reason::Shape);
			return m_reason;
		}
		if (!source.indirect_descriptor.has_value()) continue;
		// The indirect path evaluates its table, selector and candidate sources; one that is itself
		// indirect has no dwords on the tape.
		const auto& indirect = *source.indirect_descriptor;
		bool        direct   = direct_source(indirect.table_source) &&
		                (!indirect.selector.has_value() || direct_source(indirect.selector->source));
		for (const auto candidate: indirect.sources) direct = direct && direct_source(candidate);
		if (!direct) {
			Reject(Reason::Indirect);
			return m_reason;
		}
		m_tape.has_indirect = true;
	}
	if (m_plan.uniform_fill.fill.words > m_tape.fill_roots.size()) {
		Reject(Reason::Shape);
		return m_reason;
	}
	Push({}); // Node 0 always fails; unsupported immediates and untaken-arm failures share it.
	m_tape.slots.resize(m_plan.srt_reads.size());
	m_slot_state.assign(m_plan.srt_reads.size(), 0u);
	for (uint32_t slot = 0; slot < m_plan.srt_reads.size(); ++slot) {
		const auto& read   = m_plan.srt_reads[slot];
		auto&       target = m_tape.slots[slot];
		SlotRoot(slot);
		const auto* inst    = read.value.Resolve().TryInstruction();
		target.flat_offset  = read.flat_offset;
		target.flat_clean   = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
		target.has_inst     = inst != nullptr;
		target.opcode       = inst != nullptr ? inst->GetOpcode() : ValueOpcode::Void;
		// Retained reads carry SrtReadFlags, no guest PC.
		target.pc           = 0u;
	}
	for (const auto& source: m_plan.descriptor_sources) {
		auto& values = m_tape.indirect_values.emplace_back();
		values.fill(SrtTape::None);
		if (source.indirect_descriptor.has_value()) {
			const auto& indirect = *source.indirect_descriptor;
			const auto  root     = [&](Value value) { return value.IsEmpty() ? SrtTape::None : Build(value); };
			values[static_cast<size_t>(SrtIndirectValue::KeyCount)]      = root(indirect.key_count);
			values[static_cast<size_t>(SrtIndirectValue::SelectorFirst)] = root(indirect.selector_first);
			values[static_cast<size_t>(SrtIndirectValue::SelectorMask)]  = root(indirect.selector_mask);
			m_tape.descriptor_ranges.push_back(
			    {static_cast<uint32_t>(m_tape.descriptor_roots.size()), SrtTape::Unavailable});
			continue;
		}
		m_tape.descriptor_ranges.push_back(
		    {static_cast<uint32_t>(m_tape.descriptor_roots.size()), source.dword_count});
		for (uint32_t dword = 0; dword < source.dword_count; ++dword) {
			m_tape.descriptor_roots.push_back(Build(source.dwords[dword]));
		}
	}
	for (const auto& block: m_plan.control_flow) {
		m_tape.block_conditions.push_back(block.condition.IsEmpty() ? SrtTape::None
		                                                            : Build(block.condition));
	}
	m_tape.fill_words = m_plan.uniform_fill.fill.words;
	for (uint32_t i = 0; i < m_tape.fill_words; ++i) {
		m_tape.fill_roots[i] = Build(m_plan.uniform_fill.values[i]);
	}
	return m_reason;
}

uint32_t Builder::SlotRoot(uint32_t slot) {
	if (m_slot_state[slot] == 2u) return m_tape.slots[slot].root;
	if (m_slot_state[slot] == 1u) {
		Reject(Reason::Cycle);
		return 0;
	}
	m_slot_state[slot] = 1u;
	const auto root    = Build(m_plan.srt_reads[slot].value);
	m_slot_state[slot] = 2u;
	m_tape.slots[slot].root = root;
	return root;
}

uint32_t Builder::BuildImmediate(Value value) {
	SrtNode node {.op = SrtOp::Imm};
	switch (value.GetType()) {
		case Type::U1: node.imm = value.U1(); break;
		case Type::U8: node.imm = value.U8(); break;
		case Type::U16: node.imm = value.U16(); break;
		case Type::U32: node.imm = value.U32(); break;
		case Type::U64: node.imm = value.U64(); break;
		case Type::F32: node.imm = std::bit_cast<uint32_t>(value.F32Value()); break;
		default: return 0;
	}
	const auto [found, inserted] = m_immediates.try_emplace(node.imm, 0u);
	if (inserted) found->second = Push(node);
	return found->second;
}

uint32_t Builder::Build(Value value) {
	value = value.Resolve();
	if (m_reason != Reason::None) return 0;
	if (value.IsImmediate()) return BuildImmediate(value);
	const auto* inst = value.TryInstruction();
	const auto [found, inserted] = m_nodes.try_emplace(inst, InProgress);
	if (!inserted) {
		if (found->second == InProgress) {
			Reject(Reason::Cycle);
			return 0;
		}
		return found->second;
	}
	const auto index = BuildInst(*inst);
	m_nodes[inst]    = index;
	return index;
}

uint32_t Builder::BuildExtract(const Inst& inst) {
	if (inst.NumArgs() < 2u) {
		Reject(Reason::Shape, inst.GetOpcode());
		return 0;
	}
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) return 0;
	const auto component = index.U32();
	SrtNode    node;
	node.flags = static_cast<uint8_t>(component);
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		node.op      = SrtOp::ExtractU64;
		node.args[0] = Build(inst.Arg(0));
		return Push(node);
	}
	const auto* source = inst.Arg(0).Resolve().TryInstruction();
	if (source == nullptr) {
		Reject(Reason::Shape, inst.GetOpcode());
		return 0;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2 && source->NumArgs() >= 2u) {
		node.op      = SrtOp::Alias;
		node.args[0] = Build(source->Arg(component));
		return Push(node);
	}
	return 0;
}

uint32_t Builder::BuildReadConst(const Inst& inst) {
	if (inst.NumArgs() < 2u) {
		Reject(Reason::Shape, inst.GetOpcode());
		return 0;
	}
	const auto slot = inst.Arg(1).Resolve();
	if (!slot.IsImmediate() || slot.GetType() != Type::U32 ||
	    slot.U32() >= m_plan.srt_reads.size()) {
		return 0;
	}
	SrtNode node {.op = SrtOp::ReadConst};
	const auto* read = m_plan.srt_reads[slot.U32()].value.ResolveInstruction();
	node.flags   = read != nullptr && read->Flags<SrtReadFlags>().clean != 0u ? SrtNode::FlagClean : 0u;
	node.args[0] = SlotRoot(slot.U32());
	return Push(node);
}

uint32_t Builder::BuildRawRead(const Inst& inst) {
	if (!IsRawRead(m_plan, inst)) return 0;
	const auto* handle = inst.Arg(0).Resolve().TryInstruction();
	if (handle == nullptr || handle->NumArgs() < 2u) {
		Reject(Reason::Shape, inst.GetOpcode());
		return 0;
	}
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const bool buffer = vector || inst.GetOpcode() == ValueOpcode::ReadConstBuffer;
	SrtNode    head {.op = SrtOp::Handle};
	head.args[0] = Build(handle->Arg(0));
	head.args[1] = Build(handle->Arg(1));
	if (buffer && handle->NumArgs() == 4u) {
		head.flags   = SrtNode::FlagHandle4;
		head.args[2] = Build(handle->Arg(2));
		head.args[3] = Build(handle->Arg(3));
	}
	SrtNode node {.op = SrtOp::RawRead};
	node.flags = static_cast<uint8_t>((vector ? SrtNode::FlagVector : 0u) |
	                                  (buffer ? SrtNode::FlagBuffer : 0u));
	node.imm   = static_cast<uint64_t>(static_cast<int64_t>(
	    static_cast<int32_t>(m_plan.memory_info[inst.Flags<MemoryFlags>().index].offset)));
	node.args[0] = Push(head);
	node.args[1] = Build(inst.Arg(1));
	if (vector) node.args[2] = Build(inst.Arg(4));
	node.args[3] = SrtTape::None;
	if (!vector) {
		const std::array<uint32_t, 7> key {head.flags, head.args[0], head.args[1], head.args[2],
		                                   head.args[3], node.args[1], node.flags};
		node.args[3] = m_groups.try_emplace(key, static_cast<uint32_t>(m_groups.size())).first->second;
	}
	return Push(node);
}

uint32_t Builder::BuildInst(const Inst& inst) {
	const auto opcode = inst.GetOpcode();
	switch (opcode) {
		case ValueOpcode::GetUserData: {
			if (inst.NumArgs() < 1u || inst.Arg(0).Resolve().GetType() != Type::ScalarReg) {
				Reject(Reason::Shape, opcode);
				return 0;
			}
			const auto reg = RegIndex(inst.Arg(0).Resolve().ScalarRegister());
			SrtNode node {.op = SrtOp::UserData};
			node.imm = reg < m_plan.user_data_base ? UINT64_MAX : reg - m_plan.user_data_base;
			return Push(node);
		}
		case ValueOpcode::GetShaderBase: return Push({.op = SrtOp::ShaderBase});
		case ValueOpcode::Phi: Reject(Reason::Phi, opcode); return 0;
		case ValueOpcode::ReadFirstLane: Reject(Reason::ReadFirstLane, opcode); return 0;
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return BuildExtract(inst);
		case ValueOpcode::ReadConst: return BuildReadConst(inst);
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32: return BuildRawRead(inst);
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU32: return 0;
		default: break;
	}
	const auto info = OpInfoOf(opcode);
	if (info.op == SrtOp::Fail || inst.NumArgs() < info.count) {
		Reject(info.op == SrtOp::Fail ? Reason::Opcode : Reason::Shape, opcode);
		return 0;
	}
	SrtNode node {.op = info.op};
	if (info.op == SrtOp::FPLessEqual || info.op == SrtOp::FPGreaterEqual) {
		node.flags = inst.Flags<FPCompareFlags>().flush_input_denorms ? SrtNode::FlagFlush : 0u;
	}
	for (uint32_t index = 0; index < info.count; ++index) {
		node.args[index] = Build(inst.Arg(index));
	}
	return Push(node);
}

} // namespace

bool SrtTape::Matches(const ResourcePlan& plan) const {
	if (slots.size() != plan.srt_reads.size() || descriptor_ranges.size() != plan.descriptor_sources.size() ||
	    indirect_values.size() != plan.descriptor_sources.size() ||
	    block_conditions.size() != plan.control_flow.size() || fill_words != plan.uniform_fill.fill.words) {
		return false;
	}
	// Verify mode always re-scans, so a plan edited in place cannot hide behind the cache there.
	const bool cached = !SrtTapeVerifyEnabled();
	auto&      cache  = checked[t_resource_scratch_slot];
	if (cached && cache.plan == &plan && cache.sources == plan.descriptor_sources.data() &&
	    cache.count == plan.descriptor_sources.size()) {
		return true;
	}
	for (size_t source = 0; source < descriptor_ranges.size(); ++source) {
		if ((descriptor_ranges[source][1] == Unavailable) !=
		    plan.descriptor_sources[source].indirect_descriptor.has_value()) {
			return false;
		}
	}
	cache.plan    = &plan;
	cache.sources = plan.descriptor_sources.data();
	cache.count   = plan.descriptor_sources.size();
	return true;
}

std::shared_ptr<SrtTape> BuildSrtTape(const ResourcePlan& plan) {
	auto    tape = std::make_shared<SrtTape>();
	Builder builder(plan, *tape);
	const auto reason = builder.Run();
	static std::atomic_uint built {0};
	static std::atomic_uint built_indirect {0};
	static std::atomic_uint fallback[static_cast<size_t>(Reason::Count)] {};
	if (reason == Reason::None) {
		// The GPU thread's (slot 0) memos; other slots size theirs on first use (AcquireMemo).
		tape->memo[0][0].assign(tape->nodes.size(), {});
		tape->memo[0][1].assign(tape->nodes.size(), {});
		if (SrtClassTrace()) ClassifySrtReads(*tape);
		built.fetch_add(1);
		if (tape->has_indirect) built_indirect.fetch_add(1);
	} else if (fallback[static_cast<size_t>(reason)].fetch_add(1) < 8u) {
		const bool has_indirect = std::ranges::any_of(plan.descriptor_sources, [](const DescriptorSource& source) {
			return source.indirect_descriptor.has_value();
		});
		std::fprintf(stderr, "SRT_TAPE_FALLBACK hash=0x%016llx reason=%s op=%s has_indirect=%u\n",
		             static_cast<unsigned long long>(plan.shader_hash),
		             ReasonNames[static_cast<size_t>(reason)],
		             ValueOpcodeName(builder.Detail()).data(), has_indirect ? 1u : 0u);
	}
	uint32_t failed = 0;
	for (const auto& count: fallback) failed += count.load();
	const auto total = built.load() + failed;
	// Plans are cached per permutation, so report on powers of two as well as every 4096 builds.
	if ((total & (total - 1u)) == 0u || total % 4096u == 0u) {
		std::fprintf(stderr,
		             "SRT_TAPE_COVERAGE built=%u fallback=%u (indirect=%u read_first_lane=%u phi=%u "
		             "opcode=%u cycle=%u shape=%u) built_with_indirect=%u\n",
		             built.load(), failed, fallback[1].load(), fallback[2].load(), fallback[3].load(),
		             fallback[4].load(), fallback[5].load(), fallback[6].load(), built_indirect.load());
	}
	return reason == Reason::None ? tape : nullptr;
}

SrtTapeWalker::SrtTapeWalker(const SrtTape& tape, const SrtRuntime& runtime, uint32_t context,
                             SrtTapeWalker* clean_evaluator, bool quiet, bool fuse_reads,
                             SrtReadRanges* captured_reads)
    : m_tape(tape), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
      m_memo(AcquireMemo(tape, context)), m_generation(++tape.generation[t_resource_scratch_slot][context]),
      m_quiet(quiet), m_fuse(fuse_reads), m_capture(captured_reads) {}

std::vector<SrtTape::MemoEntry>& SrtTapeWalker::AcquireMemo(const SrtTape& tape, uint32_t context) {
	auto& memo = tape.memo[t_resource_scratch_slot][context];
	if (memo.size() != tape.nodes.size()) memo.assign(tape.nodes.size(), {});
	return memo;
}

uint32_t SrtTapeWalker::FusableRead(uint32_t root) const {
	if (root >= m_tape.nodes.size()) return SrtTape::None;
	const auto* node = &m_tape.nodes[root];
	if (node->op == SrtOp::ReadConst) {
		// A clean-flagged slot is evaluated by the clean evaluator (strict reader, its own memo).
		if ((node->flags & SrtNode::FlagClean) != 0u && m_clean_evaluator != nullptr) return SrtTape::None;
		root = node->args[0];
		if (root >= m_tape.nodes.size()) return SrtTape::None;
		node = &m_tape.nodes[root];
	}
	return node->op == SrtOp::RawRead && node->args[3] != SrtTape::None ? root : SrtTape::None;
}

// A run is read with one memory read only when the original evaluation would read exactly those DWORDs,
// in this order, with nothing in between:
//  * the entries are evaluated in sequence by the caller, unconditionally (a failure aborts the whole
//    materialization, which is then discarded);
//  * all reads of a group have the same operands, so once the operands of the first one evaluate, the
//    operands of the others are memo hits: nothing is read between their DWORDs;
//  * only the immediate differs (+4 per entry); each entry's own checks (negative offset, address overflow,
//    buffer range check that makes it read zero without touching memory) are repeated here, and the run
//    stops before the first entry that would not read memory at the next consecutive address;
//  * the run never crosses a 4 KiB page, so a page the original would not touch is not touched, and the
//    GPU-dirty decision of the ordinary-read hook covers the same page it covered per DWORD;
//  * entries that are already memoized (or become so while the operands evaluate) end the run.
template <typename At>
void SrtTapeWalker::PrefetchRun(size_t count, At&& at) {
	constexpr size_t MaxRun = 16;
	const uint32_t   lead_id = at(size_t {0});
	if (lead_id == SrtTape::None || m_memo[lead_id].generation == m_generation) return;
	const auto&    lead     = m_tape.nodes[lead_id];
	const uint32_t group    = lead.args[3];
	const auto     lead_imm = static_cast<int64_t>(lead.imm);
	std::array<uint32_t, MaxRun> ids;
	ids[0]     = lead_id;
	size_t run = 1;
	for (; run < count && run < MaxRun; ++run) {
		const uint32_t id = at(run);
		if (id == SrtTape::None || m_memo[id].generation == m_generation) break;
		const auto& node = m_tape.nodes[id];
		if (node.args[3] != group || static_cast<int64_t>(node.imm) != lead_imm + static_cast<int64_t>(run) * 4) {
			break;
		}
		ids[run] = id;
	}
	if (run < 2u) return;

	// The operands, in the order ReadRaw evaluates them for the first entry.
	const auto& handle = m_tape.nodes[lead.args[0]];
	const bool  buffer = (lead.flags & SrtNode::FlagBuffer) != 0u;
	uint64_t    low = 0, high = 0, offset = 0, records = 0, word3 = 0;
	if (!Eval(handle.args[0], low) || !Eval(handle.args[1], high) || !Eval(lead.args[1], offset)) return;
	if (buffer && ((handle.flags & SrtNode::FlagHandle4) == 0u || !Eval(handle.args[2], records) ||
	               !Eval(handle.args[3], word3))) {
		return;
	}
	const auto base     = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	uint64_t   address0 = 0;
	size_t     fused    = 0;
	for (; fused < run; ++fused) {
		if (m_memo[ids[fused]].generation == m_generation) break;
		const auto immediate = lead_imm + static_cast<int64_t>(fused) * 4;
		uint64_t   address   = 0;
		if (buffer) {
			if (immediate < 0) break;
			const auto byte_offset =
			    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
			const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
			const auto size   = stride == 0u
			                        ? static_cast<uint64_t>(static_cast<uint32_t>(records))
			                        : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
			if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) break;
			address = (base & ~uint64_t {3}) + byte_offset;
		} else {
			const auto relative = (immediate & ~int64_t {3}) +
			                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
			if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) break;
		}
		if (fused == 0u) {
			// The null page is never read (ReadRaw's own rule); such entries stay on the single path.
			if (address < 0x10000u) return;
			address0 = address;
		} else if (address != address0 + fused * sizeof(uint32_t) || (address >> 12u) != (address0 >> 12u)) {
			break;
		}
	}
	if (fused < 2u) return;

	std::array<uint32_t, MaxRun> words;
	// KYTY_TRACE_SRT_CLASSES: the run's class (3 when it mixes CPU-needed and shader-only words).
	if (!m_tape.cpu_needed.empty()) {
		uint8_t cls = 0;
		for (size_t index = 0; index < fused; ++index) cls |= m_tape.cpu_needed[ids[index]] != 0u ? 1u : 2u;
		t_srt_read_class = cls;
	}
	ReadOrdinaryMemory(address0, words.data(), fused * sizeof(uint32_t));
	t_srt_read_class = 0u;
	if (m_capture != nullptr) AppendSrtRead(*m_capture, address0, fused * sizeof(uint32_t), true);
	for (size_t index = 0; index < fused; ++index) {
		m_memo[ids[index]] = {words[index], m_generation};
	}
	CountFusedRun(fused);
}

bool SrtTapeWalker::Eval(uint32_t index, uint64_t& result) {
	const auto& node = m_tape.nodes[index];
	if (node.op == SrtOp::Imm) {
		result = node.imm;
		return true;
	}
	if (node.op == SrtOp::Fail) {
		return false;
	}
	if (m_memo[index].generation == m_generation) {
		result = m_memo[index].value;
		return true;
	}
	uint64_t out = 0;
	// Failures are not memoized: the walker re-evaluates (and re-reads) them on the next request.
	if (!Run(index, node, out)) {
		return false;
	}
	m_memo[index] = {out, m_generation};
	result        = out;
	return true;
}

bool SrtTapeWalker::ReadRaw(const SrtNode& node, uint64_t& result) {
	const bool vector = (node.flags & SrtNode::FlagVector) != 0u;
	const auto& handle = m_tape.nodes[node.args[0]];
	if (vector) {
		uint64_t enabled = 0;
		if (!Eval(node.args[2], enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Eval(handle.args[0], low) || !Eval(handle.args[1], high) || !Eval(node.args[1], offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(node.imm);
	uint64_t   address   = 0;
	if ((node.flags & SrtNode::FlagBuffer) != 0u) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if ((handle.flags & SrtNode::FlagHandle4) == 0u || !Eval(handle.args[2], records) ||
		    !Eval(handle.args[3], word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			// An out-of-range buffer load reads zero, as the hardware range check returns.
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t   word   = 0;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	// KYTY_TRACE_SRT_CLASSES: the class of this read while it runs.
	struct ClassScope {
		explicit ClassScope(uint8_t cls) { t_srt_read_class = cls; }
		~ClassScope() { t_srt_read_class = 0u; }
	} class_scope(m_tape.cpu_needed.empty()                                         ? uint8_t {0}
	              : m_tape.cpu_needed[static_cast<size_t>(&node - m_tape.nodes.data())] != 0u ? uint8_t {1}
	                                                                                            : uint8_t {2});
	if (reader != nullptr) {
		if (!reader(m_runtime.userdata, address, {&word, 1})) {
			return false;
		}
	} else {
		if (vector) return false;
		// The null page is never guest memory; an evaluation that reaches it reads zero.
		if (address >= 0x10000u) ReadOrdinaryMemory(address, &word, sizeof(word));
	}
	result = word;
	return true;
}

bool SrtTapeWalker::Run(uint32_t index, const SrtNode& node, uint64_t& result) {
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Eval(node.args[0], a) && Eval(node.args[1], b); };
	const auto ternary = [&]() { return binary() && Eval(node.args[2], c); };
	switch (node.op) {
		case SrtOp::UserData:
			if (node.imm >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[node.imm];
			return true;
		case SrtOp::ShaderBase: result = m_runtime.shader_base; return true;
		case SrtOp::Alias: return Eval(node.args[0], result);
		case SrtOp::ExtractU64:
			if (!Eval(node.args[0], a)) {
				return false;
			}
			result = static_cast<uint32_t>(a >> (node.flags * 32u));
			return true;
		case SrtOp::CarryExtract: {
			if (!binary()) {
				return false;
			}
			const auto sum =
			    static_cast<uint64_t>(static_cast<uint32_t>(a)) + static_cast<uint32_t>(b);
			result = node.flags == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
			return true;
		}
		case SrtOp::ConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) | (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case SrtOp::ReadConst:
			if ((node.flags & SrtNode::FlagClean) != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->Eval(node.args[0], result);
			}
			return Eval(node.args[0], result);
		case SrtOp::RawRead:
			if ((node.flags & SrtNode::FlagVector) != 0u && m_clean_evaluator != nullptr) {
				return m_clean_evaluator->Eval(index, result);
			}
			return ReadRaw(node, result);
		case SrtOp::IAdd32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a + b);
			return true;
		case SrtOp::IAdd64:
			if (!binary()) return false;
			result = a + b;
			return true;
		case SrtOp::ISub32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a - b);
			return true;
		case SrtOp::ISub64:
			if (!binary()) return false;
			result = a - b;
			return true;
		case SrtOp::IMul32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a * b);
			return true;
		case SrtOp::IMul64:
			if (!binary()) return false;
			result = a * b;
			return true;
		case SrtOp::UMulHi:
			if (!binary()) return false;
			result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
			return true;
		case SrtOp::UMin32:
			if (!binary()) return false;
			result = std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
			return true;
		case SrtOp::ConvertF32U32:
			if (!Eval(node.args[0], a)) return false;
			result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
			return true;
		case SrtOp::ConvertU32F32: {
			if (!Eval(node.args[0], a)) return false;
			const auto value = Float32(a);
			if (!std::isfinite(value) || value < 0.0f || static_cast<double>(value) > UINT32_MAX) {
				return false;
			}
			result = static_cast<uint32_t>(value);
			return true;
		}
		case SrtOp::FPMul32:
			if (!binary()) return false;
			result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
			return true;
		case SrtOp::FPTrunc32:
			if (!Eval(node.args[0], a)) return false;
			result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
			return true;
		case SrtOp::FPRecipIFlag32: {
			if (!Eval(node.args[0], a)) return false;
			const auto exponent = (a >> 23u) & 0xffu;
			// Normal positive powers of two have exact normal reciprocals in every FP mode.
			if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
			result = (254u - exponent) << 23u;
			return true;
		}
		case SrtOp::FPIsNan32:
			if (!Eval(node.args[0], a)) return false;
			result = std::isnan(Float32(a));
			return true;
		case SrtOp::FPLessEqual:
		case SrtOp::FPGreaterEqual: {
			if (!binary()) return false;
			const bool flush   = (node.flags & SrtNode::FlagFlush) != 0u;
			const auto operand = [&](uint64_t bits) {
				if (flush && (bits & 0x7fffffffu) < 0x00800000u) {
					bits &= 0x80000000u;
				}
				return Float32(bits);
			};
			result = node.op == SrtOp::FPLessEqual ? operand(a) <= operand(b) : operand(a) >= operand(b);
			return true;
		}
		case SrtOp::And32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a & b);
			return true;
		case SrtOp::And64:
			if (!binary()) return false;
			result = a & b;
			return true;
		case SrtOp::Or32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a | b);
			return true;
		case SrtOp::Xor32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a ^ b);
			return true;
		case SrtOp::Not32:
			if (!Eval(node.args[0], a)) return false;
			result = ~static_cast<uint32_t>(a);
			return true;
		case SrtOp::Shl32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) << (b & 31u);
			return true;
		case SrtOp::Shl64:
			if (!binary()) return false;
			result = a << (b & 63u);
			return true;
		case SrtOp::Shr32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) >> (b & 31u);
			return true;
		case SrtOp::Shr64:
			if (!binary()) return false;
			result = a >> (b & 63u);
			return true;
		case SrtOp::Sar32:
			if (!binary()) return false;
			result = static_cast<uint32_t>(std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
			return true;
		case SrtOp::Sar64:
			if (!binary()) return false;
			result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			return true;
		case SrtOp::BitFieldUExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			const auto mask = width == 32u  ? UINT32_MAX
			                  : width == 0u ? 0u
			                                : (uint32_t {1} << width) - 1u;
			result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
			return true;
		}
		case SrtOp::BitFieldSExtract: {
			if (!ternary()) return false;
			const auto offset = static_cast<uint32_t>(b);
			const auto width  = static_cast<uint32_t>(c);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = 0;
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
			auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
			if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
				bits |= ~mask;
			}
			result = bits;
			return true;
		}
		case SrtOp::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Eval(node.args[3], d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask = width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) | ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case SrtOp::Select: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (!predicate.Eval(node.args[0], a)) return false;
			return Eval(node.args[a != 0u ? 1u : 2u], result);
		}
		case SrtOp::IEqual:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
			return true;
		case SrtOp::INotEqual:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
			return true;
		case SrtOp::ULess:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
			return true;
		case SrtOp::ULessEqual:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
			return true;
		case SrtOp::UGreater:
			if (!binary()) return false;
			result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
			return true;
		case SrtOp::SGreaterEqual:
			if (!binary()) return false;
			result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
			         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
			return true;
		case SrtOp::LogicalAnd: {
			const bool left = Eval(node.args[0], a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (!Eval(node.args[1], b) || (b != 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case SrtOp::LogicalOr: {
			const bool left = Eval(node.args[0], a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Eval(node.args[1], b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case SrtOp::LogicalXor:
			if (!binary()) return false;
			result = (a != 0u) != (b != 0u);
			return true;
		case SrtOp::LogicalNot:
			if (!Eval(node.args[0], a)) return false;
			result = a == 0u;
			return true;
		default: return false;
	}
}

bool SrtTapeWalker::EvaluateFill(uint32_t index, uint32_t& result) {
	uint64_t wide = 0;
	if (index >= m_tape.fill_words || !Eval(m_tape.fill_roots[index], wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

bool SrtTapeWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_tape.descriptor_ranges.size()) {
		return false;
	}
	const auto [first, count] = m_tape.descriptor_ranges[source];
	if (count == SrtTape::Unavailable) {
		return false;
	}
	result             = {};
	result.dword_count = count;
	const uint32_t* roots = m_tape.descriptor_roots.data() + first;
	for (uint32_t index = 0; index < count; ++index) {
		if (m_fuse) {
			// The DWORDs are evaluated in this order whatever the earlier ones return (a failure aborts).
			PrefetchRun(count - index, [&](size_t offset) { return FusableRead(roots[index + offset]); });
		}
		uint64_t wide = 0;
		if (!Eval(m_tape.descriptor_roots[first + index], wide)) {
			return false;
		}
		result.dwords[index] = static_cast<uint32_t>(wide);
	}
	return true;
}

bool SrtTapeWalker::EvaluateIndirectValue(uint32_t source, SrtIndirectValue which, uint32_t& result) {
	if (source >= m_tape.indirect_values.size()) {
		return false;
	}
	const auto root = m_tape.indirect_values[source][static_cast<size_t>(which)];
	uint64_t   wide = 0;
	// An empty Value does not evaluate in SrtWalker either.
	if (root == SrtTape::None || !Eval(root, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

bool SrtTapeWalker::RefreshFlatBuffer(const ResourcePlan& program, std::vector<uint32_t>& flat) {
	if (!program.srt_plan_complete) return false;
	const auto refresh = [&](uint32_t slot) {
		if (slot >= m_tape.slots.size()) return false;
		const auto& read  = m_tape.slots[slot];
		const bool  clean = read.flat_clean;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto&    evaluator = clean ? *m_clean_evaluator : *this;
		uint64_t wide      = 0;
		const bool ok = read.flat_offset < flat.size() && evaluator.Eval(read.root, wide);
		if (ok) {
			flat[read.flat_offset] = static_cast<uint32_t>(wide);
		} else if (!m_quiet) {
			std::fprintf(stderr, "SRT refresh failed slot=%u clean=%u pc=0x%x op=%s\n", slot, clean ? 1u : 0u,
			             read.pc, read.has_inst ? ValueOpcodeName(read.opcode).data() : "-");
		}
		return ok;
	};
	// The scalar read refresh(slot) evaluates through this walker (clean slots go to the clean evaluator).
	const auto fusable = [&](uint32_t slot) {
		if (slot >= m_tape.slots.size()) return SrtTape::None;
		const auto& read = m_tape.slots[slot];
		return read.flat_clean || read.flat_offset >= flat.size() ? SrtTape::None : FusableRead(read.root);
	};
	auto& scratch = program.ThreadScratch();
	auto& active  = scratch.active_sources;
	if (program.control_flow.empty()) {
		active.clear();
		flat.resize(program.srt_reads.size());
		const auto slot_count = static_cast<uint32_t>(program.srt_reads.size());
		for (uint32_t slot = 0; slot < slot_count; ++slot) {
			if (m_fuse) {
				PrefetchRun(slot_count - slot, [&](size_t offset) { return fusable(slot + static_cast<uint32_t>(offset)); });
			}
			if (!refresh(slot)) return false;
		}
		return true;
	}
	flat.assign(program.srt_reads.size(), 0u);
	active.assign(program.descriptor_sources.size(), 1u);
	for (const auto& block: program.control_flow) {
		for (const auto source: block.sources) active.at(source) = 0u;
	}
	auto& visited = scratch.visited_blocks;
	auto& pending = scratch.pending_blocks;
	visited.assign(program.control_flow.size(), 0u);
	pending.clear();
	pending.push_back(0u);
	while (!pending.empty()) {
		const auto index = pending.back();
		pending.pop_back();
		if (visited.at(index)) continue;
		visited[index] = 1u;
		const auto& block = program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		const auto& list = block.srt_reads;
		for (size_t position = 0; position < list.size(); ++position) {
			if (m_fuse) {
				PrefetchRun(list.size() - position, [&](size_t offset) { return fusable(list[position + offset]); });
			}
			if (!refresh(list[position])) return false;
		}
		uint64_t condition = 0;
		auto&    predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
		const auto root    = m_tape.block_conditions[index];
		if (root != SrtTape::None && m_runtime.read_specialization_memory != nullptr &&
		    predicate.Eval(root, condition)) {
			pending.push_back(block.successors[static_cast<uint32_t>(condition) != 0u ? 0u : 1u]);
		} else {
			pending.insert(pending.end(), block.successors.begin(), block.successors.end());
		}
	}
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
