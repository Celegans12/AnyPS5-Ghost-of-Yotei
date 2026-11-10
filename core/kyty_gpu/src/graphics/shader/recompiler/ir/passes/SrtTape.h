#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTTAPE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTTAPE_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// KYTY_FIX_SRT_TAPE=0 disables the tape; KYTY_FIX_SRT_TAPE_VERIFY=1 cross-checks it against SrtWalker.
bool SrtTapeEnabled();
bool SrtTapeVerifyEnabled();

// Ranges of guest memory one materialization read (ResourceSnapshot::specialization_reads).
using SrtReadRanges = std::vector<std::pair<uint64_t, uint64_t>>;
// Appends a read. With `merge`, a read that starts where the previous one ended extends it: the consumers
// only test overlap with written ranges, so adjacent reads are one range.
inline void AppendSrtRead(SrtReadRanges& reads, uint64_t address, uint64_t bytes, bool merge) {
	if (merge && bytes != 0u && !reads.empty() && reads.back().first + reads.back().second == address) {
		reads.back().second += bytes;
		return;
	}
	reads.emplace_back(address, bytes);
}
// Merges adjacent entries in order: the list AppendSrtRead(.., true) builds incrementally.
void MergeSrtReads(SrtReadRanges& reads);
// Runs and DWORDs the tape has fused into single memory reads so far, all threads (diagnostics).
void SrtTapeFusionStats(uint64_t& runs, uint64_t& dwords);

enum class SrtOp : uint8_t {
	Imm, Fail, UserData, ShaderBase, Alias, ExtractU64, CarryExtract, ConstructU64, ReadConst,
	// args: handle, offset, guard. A Handle node only holds its operands: low, high, records, word3.
	RawRead, Handle,
	IAdd32, IAdd64, ISub32, ISub64, IMul32, IMul64, UMulHi, UMin32, ConvertF32U32, ConvertU32F32,
	FPMul32, FPTrunc32, FPRecipIFlag32, FPIsNan32, FPLessEqual, FPGreaterEqual,
	And32, And64, Or32, Xor32, Not32, Shl32, Shl64, Shr32, Shr64, Sar32, Sar64,
	BitFieldUExtract, BitFieldSExtract, BitFieldInsert, Select,
	IEqual, INotEqual, ULess, ULessEqual, UGreater, SGreaterEqual,
	LogicalAnd, LogicalOr, LogicalXor, LogicalNot,
};

// The Value members of DescriptorSource::IndirectDescriptor that MaterializeIndirectDescriptor evaluates.
enum class SrtIndirectValue : uint32_t { KeyCount, SelectorFirst, SelectorMask, Count };

struct SrtNode {
	// flags: component for extracts, Flag* bits otherwise.
	static constexpr uint8_t FlagClean   = 1u; // ReadConst: slot is evaluated by the clean evaluator
	static constexpr uint8_t FlagVector  = 1u; // RawRead: LoadBufferU32 (strict reader, guard operand)
	static constexpr uint8_t FlagBuffer  = 2u; // RawRead: records/word3 range check
	static constexpr uint8_t FlagHandle4 = 1u; // Handle: has the records/word3 operands
	static constexpr uint8_t FlagFlush   = 1u; // FPLessEqual/FPGreaterEqual: flush_input_denorms

	SrtOp                    op    = SrtOp::Fail;
	uint8_t                  flags = 0;
	std::array<uint32_t, 4> args {};
	// Imm: value. UserData: register index relative to the user-data base. RawRead: signed offset.
	uint64_t                 imm = 0;
	// RawRead args[3]: the fusion group of a scalar read (SrtTape::None for LoadBufferU32). Reads of one
	// group have the same handle operands, dynamic offset operand and flags, so they differ only in imm.
};

// Dense, immutable image of a ResourcePlan's SRT/descriptor/condition/fill value graph. It is built
// once per plan and evaluated with the same semantics as SrtWalker, but without tree walking.
class SrtTape {
public:
	static constexpr uint32_t None = UINT32_MAX;
	// descriptor_ranges[source][1] of an indirect source: its own dwords are never evaluated (they are
	// materialized by MaterializeIndirectDescriptor), so the tape holds none for it.
	static constexpr uint32_t Unavailable = UINT32_MAX;

	struct Slot {
		uint32_t       root        = None;
		uint32_t       flat_offset = 0;
		// SrtWalker::RefreshFlatBuffer indexes the clean mask by flat offset, ReadConst by slot.
		bool           flat_clean  = false;
		bool           has_inst    = false;
		ValueOpcode    opcode      = ValueOpcode::Void;
		uint32_t       pc          = 0;
	};

	struct MemoEntry {
		uint64_t value      = 0;
		uint64_t generation = 0;
	};

	// The plan may be edited after extraction by tests; refuse a tape of a different shape. With
	// The per-source scan is skipped for a plan it already accepted (same
	// object, same descriptor_sources storage and size: a plan is not edited in place after its first
	// materialization). The accepted-plan cache is per thread slot.
	[[nodiscard]] bool Matches(const ResourcePlan& plan) const;

	std::vector<SrtNode>                     nodes;
	std::vector<Slot>                        slots;
	// descriptor_ranges[source] = {first, count} into descriptor_roots.
	std::vector<std::array<uint32_t, 2>>     descriptor_ranges;
	std::vector<uint32_t>                    descriptor_roots;
	// Parallel to ResourcePlan::descriptor_sources: roots of an indirect descriptor's key_count,
	// selector_first and selector_mask (None when the Value is empty or the source is direct).
	std::vector<std::array<uint32_t, static_cast<size_t>(SrtIndirectValue::Count)>> indirect_values;
	bool                                     has_indirect = false;
	// Parallel to ResourcePlan::control_flow.
	std::vector<uint32_t>                    block_conditions;
	std::array<uint32_t, 4>                  fill_roots {None, None, None, None};
	uint32_t                                 fill_words = 0;
	// KYTY_TRACE_SRT_CLASSES: per node, 1 when the CPU needs its value (ClassifySrtReads). Empty when the trace is off.
	std::vector<uint8_t>                     cpu_needed;

	// Evaluation scratch, one per thread slot (t_resource_scratch_slot, see ResourcePlan::Scratch):
	// [slot][0] main evaluator, [slot][1] clean evaluator. Slot 0 is sized by BuildSrtTape, the others by
	// the first walker that uses them.
	mutable std::array<std::array<std::vector<MemoEntry>, 2>, ResourceScratchSlots> memo;
	mutable std::array<std::array<uint64_t, 2>, ResourceScratchSlots>               generation {};
	// Matches() cache, per slot: the plan whose descriptor_sources were last scanned and accepted.
	struct MatchCache {
		const ResourcePlan* plan    = nullptr;
		const void*         sources = nullptr;
		size_t              count   = 0;
	};
	mutable std::array<MatchCache, ResourceScratchSlots> checked;
};

// Returns null when the plan uses anything the tape does not model (the legacy walker is then used).
std::shared_ptr<SrtTape> BuildSrtTape(const ResourcePlan& plan);

// Mirrors SrtWalker for one evaluation session. `context` picks the memo (0 main, 1 clean).
class SrtTapeWalker {
public:
	// `fuse_reads` (main evaluator only): scalar DWORD reads that the refresh and descriptor loops evaluate
	// back to back, and that share handle and offset operands at consecutive offsets, are read with one
	// ReadOrdinaryMemory. Valid only when the runtime has no ordinary reader of its own
	// (SrtRuntime::read_memory == nullptr). `captured_reads` is the vector the capturing ordinary reader
	// appends to (null: reads are not captured); fused reads append to it with AppendSrtRead(merge).
	SrtTapeWalker(const SrtTape& tape, const SrtRuntime& runtime, uint32_t context,
	              SrtTapeWalker* clean_evaluator = nullptr, bool quiet = false, bool fuse_reads = false,
	              SrtReadRanges* captured_reads = nullptr);
	SrtTapeWalker(const SrtTapeWalker&)            = delete;
	SrtTapeWalker& operator=(const SrtTapeWalker&) = delete;

	bool EvaluateFill(uint32_t index, uint32_t& result);
	bool EvaluateDescriptor(uint32_t source, DescriptorValue& result);
	// Same as SrtWalker::Evaluate(<IndirectDescriptor member>, result) for the indirect descriptor of `source`.
	bool EvaluateIndirectValue(uint32_t source, SrtIndirectValue which, uint32_t& result);
	bool RefreshFlatBuffer(const ResourcePlan& program, std::vector<uint32_t>& flat);

private:
	// The calling thread's memo of `context` (0 main, 1 clean); a slot's memo is sized on its first use.
	static std::vector<SrtTape::MemoEntry>& AcquireMemo(const SrtTape& tape, uint32_t context);
	bool Eval(uint32_t index, uint64_t& result);
	bool Run(uint32_t index, const SrtNode& node, uint64_t& result);
	bool ReadRaw(const SrtNode& node, uint64_t& result);
	// The scalar RawRead node that evaluating `root` through this walker reads (the node itself, or the
	// slot of an unflagged ReadConst), or SrtTape::None.
	uint32_t FusableRead(uint32_t root) const;
	// `at(i)` is the fusable RawRead node of the i-th entry of a sequence that is evaluated in order and
	// without conditions (SrtTape::None: none). When entries 0..k-1 are unevaluated reads of one group at
	// consecutive offsets, reads their DWORDs with one memory read and memoizes them.
	template <typename At>
	void PrefetchRun(size_t count, At&& at);

	const SrtTape&                       m_tape;
	SrtRuntime                           m_runtime;
	SrtTapeWalker*                       m_clean_evaluator = nullptr;
	std::vector<SrtTape::MemoEntry>&     m_memo;
	uint64_t                             m_generation = 0;
	bool                                 m_quiet      = false;
	bool                                 m_fuse       = false;
	SrtReadRanges*                       m_capture    = nullptr;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTTAPE_H_ */
