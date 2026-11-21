#include "graphics/shader/recompiler/backend/spirv/SpirvWaveSplit.h"

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/GLSL.std.450.h>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <cstring>
#include <string>
#include <initializer_list>
#include <optional>
#include <span>
#include <utility>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

constexpr uint32_t None                    = UINT32_MAX;
constexpr uint32_t WaveLanes               = 64;
constexpr uint32_t MaxBatch                = 4;
constexpr uint32_t MaxThreads              = 1024;
constexpr uint32_t MaxSharedBytes          = 49152;
constexpr uint32_t MaxRelievedThreads      = 512;
constexpr uint32_t RegisterBoundLoopValues = 48;

struct Inst {
	uint32_t offset = 0;
	uint32_t op     = 0;
	uint32_t count  = 0;
	uint32_t type   = 0;
	uint32_t result = 0;
	uint32_t first  = 0;
};

struct Block {
	uint32_t              label = 0;
	uint32_t              begin = 0;
	uint32_t              end   = 0;
	uint32_t              merge = 0;
	std::vector<uint32_t> successors;
};

struct Function {
	uint32_t id    = 0;
	uint32_t begin = 0;
	uint32_t end   = 0;
};

bool IsSubgroupOp(uint32_t op) {
	switch (op) {
		case spv::OpGroupNonUniformRotateKHR:
		case spv::OpGroupNonUniformQuadAllKHR:
		case spv::OpGroupNonUniformQuadAnyKHR:
		case spv::OpGroupNonUniformPartitionNV:
		case spv::OpSubgroupBallotKHR:
		case spv::OpSubgroupFirstInvocationKHR:
		case spv::OpSubgroupAllKHR:
		case spv::OpSubgroupAnyKHR:
		case spv::OpSubgroupAllEqualKHR:
		case spv::OpSubgroupReadInvocationKHR: return true;
		default: return op >= spv::OpGroupNonUniformElect && op <= spv::OpGroupNonUniformQuadSwap;
	}
}

bool IsReduction(uint32_t op) {
	switch (op) {
		case spv::OpGroupNonUniformIAdd:
		case spv::OpGroupNonUniformSMin:
		case spv::OpGroupNonUniformUMin:
		case spv::OpGroupNonUniformSMax:
		case spv::OpGroupNonUniformUMax:
		case spv::OpGroupNonUniformBitwiseAnd:
		case spv::OpGroupNonUniformBitwiseOr:
		case spv::OpGroupNonUniformBitwiseXor:
		case spv::OpGroupNonUniformLogicalAnd:
		case spv::OpGroupNonUniformLogicalOr:
		case spv::OpGroupNonUniformLogicalXor: return true;
		default: return false;
	}
}

bool UniformResult(uint32_t op) {
	switch (op) {
		case spv::OpGroupNonUniformAll:
		case spv::OpGroupNonUniformAny:
		case spv::OpGroupNonUniformBroadcast:
		case spv::OpGroupNonUniformBroadcastFirst:
		case spv::OpGroupNonUniformBallot: return true;
		default: return IsReduction(op);
	}
}

bool IsImageOp(uint32_t op) {
	return (op >= spv::OpImageSampleImplicitLod && op <= spv::OpImageWrite) || op == spv::OpImageTexelPointer ||
	       (op >= spv::OpImageQuerySizeLod && op <= spv::OpImageQuerySamples) ||
	       (op >= spv::OpImageSparseSampleImplicitLod && op <= spv::OpImageSparseRead);
}

bool IsImageRead(uint32_t op) {
	return (op >= spv::OpImageSampleImplicitLod && op <= spv::OpImageRead) ||
	       (op >= spv::OpImageSparseSampleImplicitLod && op <= spv::OpImageSparseRead);
}

bool IsAtomic(uint32_t op) {
	return (op >= spv::OpAtomicLoad && op <= spv::OpAtomicXor) || op == spv::OpAtomicFlagTestAndSet || op == spv::OpAtomicFlagClear ||
	       op == spv::OpAtomicFMinEXT || op == spv::OpAtomicFMaxEXT || op == spv::OpAtomicFAddEXT;
}

bool IsRayQuery(uint32_t op) {
	return (op >= spv::OpRayQueryInitializeKHR && op <= spv::OpRayQueryGetIntersectionTypeKHR) ||
	       (op >= spv::OpRayQueryGetRayTMinKHR && op <= spv::OpRayQueryGetIntersectionWorldToObjectKHR);
}

class WaveSplit {
public:
	WaveSplit(const std::vector<uint32_t>& words, const SplitHints* hints): m_in(words) {
		if (hints != nullptr) {
			for (const auto& [pointer, update]: hints->unordered_updates) m_unordered[pointer] = update;
		}
	}

	bool Run(std::vector<uint32_t>& out, std::string& refusal) {
		if (!Parse(refusal) || !Check(refusal)) {
			return false;
		}
		Analyze();
		if (!CheckUniform(refusal) || !PlaceSharedBarriers(refusal) || !RelievesRegisters(refusal)) {
			return false;
		}
		Rewrite(out);
		return true;
	}

private:
	const std::vector<uint32_t>& m_in;
	std::unordered_map<uint32_t, SplitHints::Update> m_unordered;
	std::vector<Inst>            m_insts;
	std::vector<uint32_t>        m_def;
	std::vector<Function>        m_functions;
	std::vector<Block>           m_blocks;
	std::vector<uint32_t>        m_block_of;
	std::unordered_map<uint32_t, uint32_t> m_label_block;
	std::unordered_map<uint32_t, uint32_t> m_builtin;
	std::unordered_map<uint32_t, uint32_t> m_var_class;
	std::unordered_map<uint32_t, uint32_t> m_ptr_class;
	std::unordered_map<uint32_t, uint32_t> m_ptr_pointee;
	std::unordered_map<uint32_t, uint32_t> m_const;
	std::unordered_set<uint32_t>           m_true;
	std::unordered_set<uint32_t>           m_false;
	std::unordered_set<uint32_t>           m_stored_private;
	std::unordered_set<uint32_t>           m_bit32_types;
	std::vector<uint8_t>                   m_impure;
	std::vector<uint32_t>                  m_shared_writes;
	std::unordered_set<uint32_t>           m_barrier_before;
	std::unordered_set<uint32_t>           m_barrier_after;
	std::unordered_set<uint32_t>           m_last_lane_stores;

	uint32_t m_bound          = 0;
	uint32_t m_u32            = 0;
	uint32_t m_bool           = 0;
	uint32_t m_glsl           = 0;
	uint32_t m_entry          = 0;
	uint32_t m_entry_point    = None;
	uint32_t m_entry_function = None;
	uint32_t m_first_function = None;
	uint32_t m_last_decoration = None;
	uint32_t m_memory_model   = None;
	uint32_t m_lii            = 0;
	uint32_t m_subgroup_lane  = 0;
	uint32_t m_lid            = 0;
	uint32_t m_gid            = 0;
	uint32_t m_wgid           = 0;
	uint32_t m_uvec4          = 0;
	uint32_t m_local_size     = None;
	uint32_t m_size[3]        = {1, 1, 1};
	uint32_t m_threads        = WaveLanes;
	uint32_t m_waves          = 1;
	uint32_t m_batch          = MaxBatch;
	uint32_t m_void           = 0;
	uint32_t m_sync_fn_type   = 0;
	std::unordered_set<uint32_t> m_caps;
	bool                         m_caps_done = false;
	bool                         m_writes_buffers = false;
	bool                         m_writes_images  = false;

	std::vector<uint8_t> m_div;
	std::vector<uint8_t> m_var_div;
	std::vector<uint8_t> m_block_div;
	std::vector<uint8_t> m_phi_div;

	uint32_t Word(const Inst& inst, uint32_t index) const { return m_in[inst.first + index]; }
	uint32_t Operands(const Inst& inst) const { return inst.offset + inst.count - inst.first; }

	bool Parse(std::string& refusal) {
		if (m_in.size() < 5 || m_in[0] != spv::MagicNumber) {
			refusal = "not a SPIR-V module";
			return false;
		}
		m_bound = m_in[3];
		m_def.assign(m_bound, None);
		Function* function = nullptr;
		for (size_t i = 5; i < m_in.size();) {
			Inst inst;
			inst.offset = static_cast<uint32_t>(i);
			inst.op     = m_in[i] & spv::OpCodeMask;
			inst.count  = m_in[i] >> spv::WordCountShift;
			if (inst.count == 0 || i + inst.count > m_in.size()) {
				refusal = "malformed instruction";
				return false;
			}
			bool has_result = false;
			bool has_type   = false;
			spv::HasResultAndType(static_cast<spv::Op>(inst.op), &has_result, &has_type);
			uint32_t at = inst.offset + 1;
			if (has_type) inst.type = m_in[at++];
			if (has_result) inst.result = m_in[at++];
			inst.first       = at;
			const auto index = static_cast<uint32_t>(m_insts.size());
			if (inst.result != 0 && inst.result < m_bound) m_def[inst.result] = index;
			switch (inst.op) {
				case spv::OpTypeInt:
					if (Word(inst, 0) == 32u) {
						m_bit32_types.insert(inst.result);
						if (Word(inst, 1) == 0u) m_u32 = inst.result;
					}
					break;
				case spv::OpTypeFloat:
					if (Word(inst, 0) == 32u) m_bit32_types.insert(inst.result);
					break;
				case spv::OpTypeBool: m_bool = inst.result; break;
				case spv::OpTypeVoid: m_void = inst.result; break;
				case spv::OpTypeVector:
					if (Word(inst, 0) == m_u32 && Word(inst, 1) == 4u) m_uvec4 = inst.result;
					break;
				case spv::OpTypePointer:
					m_ptr_class[inst.result]   = Word(inst, 0);
					m_ptr_pointee[inst.result] = Word(inst, 1);
					break;
				case spv::OpVariable: m_var_class[inst.result] = Word(inst, 0); break;
				case spv::OpConstant:
					if (Operands(inst) == 1u) m_const[inst.result] = Word(inst, 0);
					break;
				case spv::OpConstantTrue: m_true.insert(inst.result); break;
				case spv::OpConstantFalse: m_false.insert(inst.result); break;
				case spv::OpCapability: m_caps.insert(Word(inst, 0)); break;
				case spv::OpExtInstImport:
					if (std::strncmp(reinterpret_cast<const char*>(&m_in[inst.first]), "GLSL.std.450",
					                 (inst.count - 2u) * 4u) == 0) {
						m_glsl = inst.result;
					}
					break;
				case spv::OpMemoryModel: m_memory_model = index; break;
				case spv::OpEntryPoint:
					if (m_entry_point != None) {
						refusal = "more than one entry point";
						return false;
					}
					if (m_in[inst.offset + 1] != spv::ExecutionModelGLCompute) {
						refusal = "not a compute entry point";
						return false;
					}
					m_entry_point = index;
					m_entry       = m_in[inst.offset + 2];
					break;
				case spv::OpExecutionMode:
					if (m_in[inst.offset + 2] == spv::ExecutionModeDerivativeGroupQuadsKHR ||
					    m_in[inst.offset + 2] == spv::ExecutionModeDerivativeGroupLinearKHR) {
						refusal = "compute derivatives group invocations by the workgroup layout";
						return false;
					}
					if (m_in[inst.offset + 2] == spv::ExecutionModeLocalSize) {
						const uint64_t threads =
						    uint64_t {m_in[inst.offset + 3]} * m_in[inst.offset + 4] * m_in[inst.offset + 5];
						if (threads == 0 || threads % WaveLanes != 0 || threads > MaxThreads) {
							refusal = "workgroup is not whole waves";
							return false;
						}
						m_local_size = index;
						m_threads    = static_cast<uint32_t>(threads);
						m_waves      = m_threads / WaveLanes;
						for (uint32_t k = 0; k < 3; k++) m_size[k] = m_in[inst.offset + 3 + k];
					}
					break;
				case spv::OpExecutionModeId:
					refusal = "workgroup size given by id";
					return false;
				case spv::OpDecorate:
				case spv::OpMemberDecorate:
				case spv::OpDecorateId:
				case spv::OpDecorateString:
				case spv::OpMemberDecorateString:
					m_last_decoration = index;
					if (inst.op == spv::OpDecorate && Word(inst, 1) == spv::DecorationBuiltIn) {
						m_builtin[Word(inst, 0)] = Word(inst, 2);
					}
					break;
				case spv::OpFunction:
					if (m_first_function == None) m_first_function = index;
					m_functions.push_back({inst.result, index, index});
					function = &m_functions.back();
					break;
				case spv::OpFunctionEnd:
					if (function != nullptr) function->end = index;
					function = nullptr;
					break;
				case spv::OpStore:
				case spv::OpCopyMemory:
				case spv::OpCopyMemorySized:
					if (const auto root = Root(m_in[inst.first]);
					    m_var_class.contains(root) && m_var_class[root] == spv::StorageClassPrivate) {
						m_stored_private.insert(root);
					}
					break;
				default: break;
			}
			m_insts.push_back(inst);
			i += inst.count;
		}
		if (m_entry_point == None || m_u32 == 0 || m_bool == 0 || m_void == 0 || m_first_function == None ||
		    m_memory_model == None || m_local_size == None) {
			refusal = "module lacks an entry point or basic types";
			return false;
		}
		for (const auto& [variable, builtin]: m_builtin) {
			switch (builtin) {
				case spv::BuiltInLocalInvocationIndex: m_lii = variable; break;
				case spv::BuiltInSubgroupLocalInvocationId: m_subgroup_lane = variable; break;
				case spv::BuiltInLocalInvocationId: m_lid = variable; break;
				case spv::BuiltInGlobalInvocationId: m_gid = variable; break;
				case spv::BuiltInWorkgroupId: m_wgid = variable; break;
				case spv::BuiltInWorkgroupSize:
					refusal = "module reads the workgroup size builtin";
					return false;
				case spv::BuiltInSubgroupSize:
				case spv::BuiltInNumSubgroups:
				case spv::BuiltInSubgroupId:
				case spv::BuiltInSubgroupEqMask:
				case spv::BuiltInSubgroupGeMask:
				case spv::BuiltInSubgroupGtMask:
				case spv::BuiltInSubgroupLeMask:
				case spv::BuiltInSubgroupLtMask:
					refusal = "module reads a subgroup layout builtin";
					return false;
				default: break;
			}
		}
		for (uint32_t f = 0; f < m_functions.size(); f++) {
			if (m_functions[f].id == m_entry) m_entry_function = f;
		}
		if (m_entry_function == None) {
			refusal = "entry function not found";
			return false;
		}
		for (const auto& inst: m_insts) {
			if (inst.op != spv::OpFunctionCall) continue;
			for (uint32_t k = 1; k < Operands(inst); k++) {
				const auto root = Root(Word(inst, k));
				if (m_var_class.contains(root) && m_var_class[root] == spv::StorageClassPrivate) m_stored_private.insert(root);
			}
		}
		BuildBlocks();
		return true;
	}

	uint32_t Root(uint32_t pointer) const {
		for (uint32_t guard = 0; guard < 64 && pointer < m_bound && m_def[pointer] != None; guard++) {
			const auto& def = m_insts[m_def[pointer]];
			if (def.op != spv::OpAccessChain && def.op != spv::OpInBoundsAccessChain &&
			    def.op != spv::OpPtrAccessChain && def.op != spv::OpCopyObject) {
				break;
			}
			pointer = Word(def, 0);
		}
		return pointer;
	}

	void BuildBlocks() {
		const auto& function = m_functions[m_entry_function];
		m_block_of.assign(m_insts.size(), None);
		Block* block = nullptr;
		for (uint32_t i = function.begin; i <= function.end; i++) {
			const auto& inst = m_insts[i];
			if (inst.op == spv::OpLabel) {
				m_label_block[inst.result] = static_cast<uint32_t>(m_blocks.size());
				m_blocks.push_back({inst.result, i, i, 0, {}});
				block = &m_blocks.back();
			}
			if (block == nullptr) continue;
			m_block_of[i] = static_cast<uint32_t>(m_blocks.size() - 1);
			switch (inst.op) {
				case spv::OpSelectionMerge:
				case spv::OpLoopMerge: block->merge = Word(inst, 0); break;
				case spv::OpBranch: block->successors = {Word(inst, 0)}; break;
				case spv::OpBranchConditional: block->successors = {Word(inst, 1), Word(inst, 2)}; break;
				case spv::OpSwitch:
					block->successors.push_back(Word(inst, 1));
					for (uint32_t k = 3; k < Operands(inst); k += 2) block->successors.push_back(Word(inst, k));
					break;
				default: break;
			}
			if (inst.op == spv::OpBranch || inst.op == spv::OpBranchConditional || inst.op == spv::OpSwitch ||
			    inst.op == spv::OpReturn || inst.op == spv::OpReturnValue || inst.op == spv::OpKill ||
			    inst.op == spv::OpUnreachable || inst.op == spv::OpTerminateInvocation) {
				block->end = i;
				block      = nullptr;
			}
		}
	}

	uint32_t ValueType(uint32_t id) const { return id < m_bound && m_def[id] != None ? m_insts[m_def[id]].type : 0; }

	uint32_t ScopeOf(const Inst& inst) const {
		const auto found = m_const.find(Word(inst, 0));
		return found == m_const.end() ? None : found->second;
	}

	bool Check(std::string& refusal) {
		uint64_t shared_bytes = 0;
		for (const auto& [variable, storage]: m_var_class) {
			if (storage != spv::StorageClassWorkgroup) continue;
			const auto& def  = m_insts[m_def[variable]];
			const auto  type = m_ptr_pointee[def.type];
			const auto& arr  = m_insts[m_def[type]];
			if (arr.op != spv::OpTypeArray || !m_bit32_types.contains(Word(arr, 0)) || !m_const.contains(Word(arr, 1))) {
				refusal = "workgroup memory of an unknown size";
				return false;
			}
			shared_bytes += uint64_t {m_const[Word(arr, 1)]} * 4u;
		}
		const uint64_t sync_bytes = m_waves > 1 ? uint64_t {m_waves} * 4u : 0u;
		m_batch                   = MaxBatch;
		while (m_batch > 1 && shared_bytes + sync_bytes + uint64_t {m_waves} * ExchangeWords() * 4u > MaxSharedBytes) {
			m_batch /= 2;
		}
		if (shared_bytes + sync_bytes + uint64_t {m_waves} * ExchangeWords() * 4u > MaxSharedBytes) {
			refusal = "no workgroup memory left for the exchange";
			return false;
		}
		const auto& entry = m_functions[m_entry_function];
		for (uint32_t i = m_first_function; i < m_insts.size(); i++) {
			const auto& inst     = m_insts[i];
			const bool  in_entry = i >= entry.begin && i <= entry.end;
			if (inst.op == spv::OpImageWrite || inst.op == spv::OpImageTexelPointer) m_writes_images = true;
			if (inst.op == spv::OpStore || inst.op == spv::OpCopyMemory || inst.op == spv::OpCopyMemorySized ||
			    (IsAtomic(inst.op) && inst.op != spv::OpAtomicLoad)) {
				m_writes_buffers = m_writes_buffers || IsBufferPointer(Word(inst, 0));
			}
			if (inst.op == spv::OpStore || inst.op == spv::OpCopyMemory || inst.op == spv::OpCopyMemorySized ||
			    (IsAtomic(inst.op) && inst.op != spv::OpAtomicLoad)) {
				const auto root = Root(Word(inst, 0));
				if (m_var_class.contains(root) && m_var_class[root] == spv::StorageClassWorkgroup) {
					if (!in_entry) {
						refusal = "workgroup memory written outside the entry function";
						return false;
					}
					m_shared_writes.push_back(i);
				}
			}
			if ((inst.op == spv::OpCopyObject || inst.op == spv::OpPtrAccessChain || inst.op == spv::OpFunctionCall) &&
			    Operands(inst) > 0) {
				for (uint32_t k = 0; k < Operands(inst); k++) {
					const auto id    = Word(inst, k);
					const auto found = m_builtin.find(Root(id));
					if (found != m_builtin.end() && m_var_class.contains(found->first) &&
					    (found->second == spv::BuiltInLocalInvocationIndex || found->second == spv::BuiltInLocalInvocationId ||
					     found->second == spv::BuiltInGlobalInvocationId ||
					     found->second == spv::BuiltInSubgroupLocalInvocationId) &&
					    (inst.op != spv::OpFunctionCall || k > 0)) {
						refusal = "thread id builtin used through a copied pointer";
						return false;
					}
				}
			}
			if (inst.op == spv::OpControlBarrier) {
				if (!in_entry) {
					refusal = "barrier outside the entry function";
					return false;
				}
			}
			if ((inst.op >= spv::OpGroupAll && inst.op <= spv::OpGroupSMax) ||
			    (inst.op >= spv::OpSubgroupBallotKHR && inst.op <= spv::OpSubgroupReadInvocationKHR) ||
			    (inst.op >= spv::OpGroupIAddNonUniformAMD && inst.op <= spv::OpGroupSMaxNonUniformAMD)) {
				refusal = "group operation outside the supported set";
				return false;
			}
			if (!IsSubgroupOp(inst.op)) continue;
			if (!in_entry) {
				refusal = "subgroup operation outside the entry function";
				return false;
			}
			if (ScopeOf(inst) != spv::ScopeSubgroup) {
				refusal = "subgroup operation with another scope";
				return false;
			}
			switch (inst.op) {
				case spv::OpGroupNonUniformElect:
				case spv::OpGroupNonUniformAll:
				case spv::OpGroupNonUniformAny:
				case spv::OpGroupNonUniformBallotBitExtract:
				case spv::OpGroupNonUniformBallotFindLSB:
				case spv::OpGroupNonUniformBallotFindMSB: break;
				case spv::OpGroupNonUniformBallot:
					if (ValueTypeIsUvec4(inst.type)) break;
					refusal = "ballot of another type";
					return false;
				case spv::OpGroupNonUniformBallotBitCount:
					if (Word(inst, 1) <= spv::GroupOperationExclusiveScan) break;
					refusal = "clustered ballot count";
					return false;
				case spv::OpGroupNonUniformBroadcast:
				case spv::OpGroupNonUniformBroadcastFirst:
				case spv::OpGroupNonUniformShuffle:
					if (inst.type == m_bool || m_bit32_types.contains(inst.type)) break;
					refusal = "shuffle of a value wider than 32 bits";
					return false;
				default:
					if (IsReduction(inst.op) && Word(inst, 1) == spv::GroupOperationReduce && Operands(inst) == 3u &&
					    (inst.type == m_bool || m_bit32_types.contains(inst.type))) {
						break;
					}
					refusal = "unsupported subgroup operation";
					return false;
			}
		}
		return true;
	}

	uint32_t ExchangeWords() const { return 2u * m_batch * WaveLanes; }

	bool LaneIndexLoad(uint32_t id) const {
		if (id >= m_bound || m_def[id] == None) return false;
		const auto& load = m_insts[m_def[id]];
		if (load.op != spv::OpLoad) return false;
		const auto pointer = Word(load, 0);
		if (m_lii != 0 && pointer == m_lii) return true;
		if (pointer >= m_bound || m_def[pointer] == None || m_size[0] % WaveLanes != 0) return false;
		const auto& chain = m_insts[m_def[pointer]];
		if ((chain.op != spv::OpAccessChain && chain.op != spv::OpInBoundsAccessChain) || Operands(chain) != 2) return false;
		const auto variable = Word(chain, 0);
		const auto axis     = m_const.find(Word(chain, 1));
		return (variable == m_lid || variable == m_gid) && variable != 0 && axis != m_const.end() && axis->second == 0;
	}

	bool WaveIndexOfLane(const Inst& inst) const {
		if ((inst.op != spv::OpUDiv && inst.op != spv::OpShiftRightLogical) || Operands(inst) != 2) {
			return false;
		}
		if (!LaneIndexLoad(Word(inst, 0))) return false;
		const auto divisor = m_const.find(Word(inst, 1));
		if (divisor == m_const.end()) return false;
		return inst.op == spv::OpUDiv ? divisor->second != 0 && divisor->second % WaveLanes == 0
		                              : divisor->second >= 6u && divisor->second < 32u;
	}

	bool ValueTypeIsUvec4(uint32_t type) const {
		if (type >= m_bound || m_def[type] == None) return false;
		const auto& def = m_insts[m_def[type]];
		return def.op == spv::OpTypeVector && Word(def, 0) == m_u32 && Word(def, 1) == 4u;
	}

	bool Div(uint32_t id) const { return id < m_bound && m_div[id] != 0; }

	bool IsBufferPointer(uint32_t pointer) const {
		const auto found = m_ptr_class.find(ValueType(pointer));
		return found != m_ptr_class.end() &&
		       (found->second == spv::StorageClassStorageBuffer || found->second == spv::StorageClassPhysicalStorageBuffer ||
		        found->second == spv::StorageClassUniform);
	}

	bool LoadDivergent(uint32_t pointer) const {
		if (Div(pointer)) return true;
		if (m_writes_buffers && IsBufferPointer(pointer)) return true;
		const auto root  = Root(pointer);
		const auto found = m_var_class.find(root);
		if (found == m_var_class.end()) return Div(root);
		switch (found->second) {
			case spv::StorageClassInput: {
				const auto builtin = m_builtin.find(root);
				return builtin == m_builtin.end() || (builtin->second != spv::BuiltInWorkgroupId &&
				                                      builtin->second != spv::BuiltInNumWorkgroups &&
				                                      builtin->second != spv::BuiltInWorkgroupSize);
			}
			case spv::StorageClassUniform:
			case spv::StorageClassUniformConstant:
			case spv::StorageClassPushConstant:
			case spv::StorageClassStorageBuffer:
			case spv::StorageClassPhysicalStorageBuffer: return false;
			case spv::StorageClassFunction: return m_var_div[root] != 0;
			case spv::StorageClassPrivate: return m_var_div[root] != 0;
			default: return true;
		}
	}

	void ComputeImpure() {
		m_impure.assign(m_functions.size(), 0);
		std::unordered_map<uint32_t, uint32_t> index_of;
		for (uint32_t f = 0; f < m_functions.size(); f++) index_of[m_functions[f].id] = f;
		for (bool changed = true; changed;) {
			changed = false;
			for (uint32_t f = 0; f < m_functions.size(); f++) {
				if (m_impure[f] != 0 || f == m_entry_function) continue;
				for (uint32_t i = m_functions[f].begin; i <= m_functions[f].end; i++) {
					const auto& inst = m_insts[i];
					bool        bad  = IsImageOp(inst.op) || IsAtomic(inst.op) || IsRayQuery(inst.op) ||
					           IsSubgroupOp(inst.op);
					if (inst.op == spv::OpLoad) {
						bad = bad || (m_writes_buffers && IsBufferPointer(Word(inst, 0)));
						const auto root  = Root(Word(inst, 0));
						const auto found = m_var_class.find(root);
						bad = bad || (found != m_var_class.end() &&
						              (found->second == spv::StorageClassInput ||
						               found->second == spv::StorageClassWorkgroup ||
						               (found->second == spv::StorageClassPrivate && m_stored_private.contains(root))));
					}
					if (inst.op == spv::OpFunctionCall) {
						const auto callee = index_of.find(Word(inst, 0));
						bad = bad || callee == index_of.end() || m_impure[callee->second] != 0;
					}
					if (bad) {
						m_impure[f] = 1;
						changed     = true;
						break;
					}
				}
			}
		}
		m_impure_index = std::move(index_of);
	}

	std::unordered_map<uint32_t, uint32_t> m_impure_index;

	bool CallDivergent(const Inst& inst) const {
		const auto callee = m_impure_index.find(Word(inst, 0));
		return callee == m_impure_index.end() || m_impure[callee->second] != 0;
	}

	bool AnyOperandDivergent(const Inst& inst) const {
		const auto n = Operands(inst);
		for (uint32_t k = 0; k < n; k++) {
			if (inst.op == spv::OpExtInst && k == 1u) continue;
			if (Div(Word(inst, k))) return true;
		}
		return false;
	}

	std::vector<uint32_t> Reach(const Block& from, uint32_t stop) const {
		std::vector<uint32_t> seen;
		std::vector<uint8_t>  mark(m_blocks.size(), 0);
		std::vector<uint32_t> work(from.successors.begin(), from.successors.end());
		while (!work.empty()) {
			const auto label = work.back();
			work.pop_back();
			const auto found = m_label_block.find(label);
			if (label == stop || found == m_label_block.end() || mark[found->second] != 0) continue;
			mark[found->second] = 1;
			seen.push_back(found->second);
			for (const auto next: m_blocks[found->second].successors) work.push_back(next);
		}
		return seen;
	}

	void Analyze() {
		m_div.assign(m_bound, 0);
		m_var_div.assign(m_bound, 0);
		m_block_div.assign(m_blocks.size(), 0);
		m_phi_div.assign(m_blocks.size(), 0);
		ComputeImpure();
		std::vector<std::vector<uint32_t>> loops(m_blocks.size());
		for (uint32_t b = 0; b < m_blocks.size(); b++) {
			const auto& block = m_blocks[b];
			if (block.merge != 0 && m_insts[block.end - 1].op == spv::OpLoopMerge) {
				loops[b] = Reach(block, block.merge);
			}
		}
		const auto& entry = m_functions[m_entry_function];
		for (uint32_t i = m_first_function; i < m_insts.size(); i++) {
			const auto& inst = m_insts[i];
			if (i >= entry.begin && i <= entry.end) {
				if (inst.op != spv::OpFunctionCall) continue;
				for (uint32_t k = 1; k < Operands(inst); k++) {
					const auto root = Root(Word(inst, k));
					if (m_var_class.contains(root)) m_var_div[root] = 1;
				}
				continue;
			}
			if (inst.op == spv::OpStore || inst.op == spv::OpCopyMemory || inst.op == spv::OpCopyMemorySized ||
			    IsAtomic(inst.op)) {
				const auto root = Root(m_in[inst.first]);
				if (root < m_bound) m_var_div[root] = 1;
			}
		}
		for (bool changed = true; changed;) {
			changed           = false;
			const auto divide = [&](uint32_t id) {
				if (id < m_bound && m_div[id] == 0) {
					m_div[id] = 1;
					changed   = true;
				}
			};
			const auto divide_block = [&](uint32_t b) {
				if (m_block_div[b] == 0) {
					m_block_div[b] = 1;
					changed        = true;
				}
			};
			const auto divide_phis = [&](uint32_t label) {
				const auto found = m_label_block.find(label);
				if (found != m_label_block.end() && m_phi_div[found->second] == 0) {
					m_phi_div[found->second] = 1;
					changed                  = true;
				}
			};
			for (uint32_t i = entry.begin; i <= entry.end; i++) {
				const auto& inst = m_insts[i];
				const auto  b    = m_block_of[i];
				if (b == None) continue;
				const bool in_div = m_block_div[b] != 0;
				if (inst.op == spv::OpStore) {
					const auto root = Root(Word(inst, 0));
					if ((in_div || Div(Word(inst, 1)) || Div(Word(inst, 0))) && root < m_bound && m_var_div[root] == 0) {
						m_var_div[root] = 1;
						changed         = true;
					}
					continue;
				}
				if (inst.op == spv::OpCopyMemory || inst.op == spv::OpCopyMemorySized) {
					const auto root = Root(m_in[inst.first]);
					if (root < m_bound && m_var_div[root] == 0) {
						m_var_div[root] = 1;
						changed         = true;
					}
					continue;
				}
				if (inst.result == 0 || inst.type == 0) continue;
				bool d = in_div;
				switch (inst.op) {
					case spv::OpPhi:
						d = d || m_phi_div[b] != 0;
						for (uint32_t k = 0; k < Operands(inst); k += 2) d = d || Div(Word(inst, k));
						break;
					case spv::OpLoad: d = d || LoadDivergent(Word(inst, 0)); break;
					case spv::OpFunctionCall:
						d = d || AnyOperandDivergent(inst) || CallDivergent(inst);
						for (uint32_t k = 1; k < Operands(inst); k++) {
							const auto root = Root(Word(inst, k));
							d = d || (m_var_class.contains(root) && m_var_div[root] != 0);
						}
						break;
					case spv::OpGroupNonUniformElect: d = true; break;
					case spv::OpUDiv:
					case spv::OpShiftRightLogical:
						d = d || (!WaveIndexOfLane(inst) && AnyOperandDivergent(inst));
						break;
					case spv::OpGroupNonUniformShuffle: d = d || Div(Word(inst, 1)); break;
					case spv::OpGroupNonUniformBallotBitCount:
						d = d || Word(inst, 1) != spv::GroupOperationReduce || Div(Word(inst, 2));
						break;
					default:
						if (UniformResult(inst.op)) break;
						d = d || IsAtomic(inst.op) || IsRayQuery(inst.op) || (m_writes_images && IsImageRead(inst.op)) ||
						    AnyOperandDivergent(inst);
						break;
				}
				if (d) divide(inst.result);
			}
			for (uint32_t b = 0; b < m_blocks.size(); b++) {
				const auto& block = m_blocks[b];
				const auto& term  = m_insts[block.end];
				if ((term.op != spv::OpBranchConditional && term.op != spv::OpSwitch) || !Div(Word(term, 0))) continue;
				if (block.merge != 0) {
					const auto region = Reach(block, block.merge);
					for (const auto inner: region) divide_block(inner);
					divide_phis(block.merge);
					std::vector<uint32_t> exits;
					for (const auto inner: region) {
						for (const auto next: m_blocks[inner].successors) {
							if (next != block.merge) exits.push_back(next);
						}
					}
					for (uint32_t h = 0; h < m_blocks.size(); h++) {
						const auto& outer = m_blocks[h];
						if (outer.merge == 0 || h == b) continue;
						const auto& merge = m_insts[outer.end - 1];
						const bool  loop  = merge.op == spv::OpLoopMerge;
						const auto  body  = loop ? loops[h] : Reach(outer, outer.merge);
						if (std::find(body.begin(), body.end(), b) == body.end()) continue;
						const bool left = std::any_of(exits.begin(), exits.end(), [&](uint32_t next) {
							return next == outer.merge || (loop && (next == Word(merge, 1) || next == outer.label));
						});
						if (!left) continue;
						if (!loop) {
							if (!m_unstructured_exit) {
								m_unstructured_exit = true;
								changed             = true;
							}
							continue;
						}
						for (const auto inner: body) divide_block(inner);
						divide_block(h);
						divide_phis(outer.merge);
					}
					continue;
				}
				std::vector<uint32_t> targeted;
				for (uint32_t h = 0; h < m_blocks.size(); h++) {
					if (loops[h].empty() && m_insts[m_blocks[h].end - 1].op != spv::OpLoopMerge) continue;
					const auto& merge = m_insts[m_blocks[h].end - 1];
					for (const auto next: block.successors) {
						if (next == m_blocks[h].label || next == Word(merge, 0) || next == Word(merge, 1)) {
							targeted.push_back(h);
							break;
						}
					}
				}
				if (targeted.empty() && !m_unstructured_exit) {
					m_unstructured_exit = true;
					changed             = true;
				}
				for (uint32_t h = 0; h < m_blocks.size(); h++) {
					if (std::find(loops[h].begin(), loops[h].end(), b) == loops[h].end()) continue;
					if (!targeted.empty() && std::find(targeted.begin(), targeted.end(), h) == targeted.end()) continue;
					for (const auto inner: loops[h]) divide_block(inner);
					divide_block(h);
					divide_phis(m_blocks[h].merge);
				}
			}
		}
	}

	bool m_unstructured_exit = false;

	bool CheckUniform(std::string& refusal) const {
		if (m_unstructured_exit) {
			refusal = "lanes leave a construct through a divergent branch";
			return false;
		}
		const auto& entry = m_functions[m_entry_function];
		for (uint32_t i = entry.begin; i <= entry.end; i++) {
			const auto& inst = m_insts[i];
			const auto  b    = m_block_of[i];
			if (b == None || m_block_div[b] == 0) continue;
			if (IsSubgroupOp(inst.op) || inst.op == spv::OpControlBarrier) {
				refusal = "subgroup operation in divergent control flow";
				return false;
			}
			if (inst.op == spv::OpReturn || inst.op == spv::OpReturnValue || inst.op == spv::OpKill ||
			    inst.op == spv::OpTerminateInvocation) {
				refusal = "lanes leave the wave early";
				return false;
			}
		}
		return true;
	}

	std::optional<std::vector<uint32_t>> ConstantWord(uint32_t pointer) const {
		const auto& chain = m_insts[m_def[pointer]];
		if (chain.op != spv::OpAccessChain && chain.op != spv::OpInBoundsAccessChain) return std::nullopt;
		if (m_insts[m_def[Word(chain, 0)]].op != spv::OpVariable) return std::nullopt;
		std::vector<uint32_t> word {Word(chain, 0)};
		for (uint32_t k = 1; k < Operands(chain); k++) {
			const auto value = Fold(Word(chain, k), 0);
			if (!value) return std::nullopt;
			word.push_back(*value);
		}
		return word;
	}

	std::optional<uint32_t> Fold(uint32_t id, uint32_t depth) const {
		if (const auto found = m_const.find(id); found != m_const.end()) return found->second;
		if (depth > 16 || id >= m_def.size() || m_def[id] == 0) return std::nullopt;
		const auto& inst = m_insts[m_def[id]];
		if (Operands(inst) != 2) return std::nullopt;
		const auto a = Fold(Word(inst, 0), depth + 1), b = Fold(Word(inst, 1), depth + 1);
		if (!a || !b) return std::nullopt;
		switch (inst.op) {
			case spv::OpIAdd: return *a + *b;
			case spv::OpISub: return *a - *b;
			case spv::OpIMul: return *a * *b;
			case spv::OpBitwiseAnd: return *a & *b;
			case spv::OpBitwiseOr: return *a | *b;
			case spv::OpBitwiseXor: return *a ^ *b;
			case spv::OpShiftRightLogical: return *b < 32u ? *a >> *b : 0u;
			case spv::OpShiftLeftLogical: return *b < 32u ? *a << *b : 0u;
			default: return std::nullopt;
		}
	}

	bool PlaceSharedBarriers(std::string& refusal) {
		if (m_shared_writes.empty()) return true;
		std::vector<uint32_t> owner(m_blocks.size(), None);
		for (uint32_t b = 0; b < m_blocks.size(); b++) {
			const auto& block = m_blocks[b];
			const auto& term  = m_insts[block.end];
			if (m_block_div[b] != 0 || block.merge == 0 ||
			    (term.op != spv::OpBranchConditional && term.op != spv::OpSwitch) || !Div(Word(term, 0))) {
				continue;
			}
			for (const auto inner: Reach(block, block.merge)) {
				if (owner[inner] == None) owner[inner] = b;
			}
		}
		std::unordered_set<uint32_t> checked;
		for (const auto i: m_shared_writes) {
			const auto b = m_block_of[i];
			if (m_block_div[b] == 0) {
				m_barrier_before.insert(i);
				m_barrier_after.insert(i);
				const auto& store = m_insts[i];
				if (store.op == spv::OpStore && !Div(Word(store, 0)) && Div(Word(store, 1)) &&
				    m_bit32_types.contains(ValueType(Word(store, 1)))) {
					m_last_lane_stores.insert(i);
				}
				continue;
			}
			if (owner[b] == None) {
				refusal = "workgroup memory written in a divergent loop";
				return false;
			}
			const auto& header = m_blocks[owner[b]];
			if (!checked.insert(owner[b]).second) continue;
			std::vector<std::pair<uint32_t, SplitHints::Update>> updates;
			bool                                                 plain_write = false;
			for (const auto inner: Reach(header, header.merge)) {
				for (uint32_t j = m_blocks[inner].begin; j <= m_blocks[inner].end; j++) {
					const auto& other = m_insts[j];
					if (other.op == spv::OpFunctionCall && CallDivergent(other)) {
						refusal = "call that may read workgroup memory in a divergent region that writes it";
						return false;
					}
					if (other.op != spv::OpLoad && other.op != spv::OpCopyMemory && other.op != spv::OpCopyMemorySized &&
					    other.op != spv::OpStore && !IsAtomic(other.op)) {
						continue;
					}
					const bool store = other.op == spv::OpStore;
					const auto from  = Root(other.op == spv::OpLoad || store || IsAtomic(other.op) ? Word(other, 0) : Word(other, 1));
					const bool shared = m_var_class.contains(from) && m_var_class[from] == spv::StorageClassWorkgroup;
					const bool shared_target =
					    other.op == spv::OpCopyMemory || other.op == spv::OpCopyMemorySized
					        ? m_var_class.contains(Root(Word(other, 0))) && m_var_class[Root(Word(other, 0))] == spv::StorageClassWorkgroup
					        : shared;
					if (const auto update = IsAtomic(other.op) ? m_unordered.find(Word(other, 0)) : m_unordered.end();
					    shared && update != m_unordered.end()) {
						updates.emplace_back(Word(other, 0), update->second);
						continue;
					}
					if (store || ((other.op == spv::OpCopyMemory || other.op == spv::OpCopyMemorySized) && shared_target)) {
						plain_write = plain_write || shared_target;
						if (store) continue;
					}
					if (shared) {
						refusal = "workgroup memory read and written in one divergent region";
						return false;
					}
				}
			}
			if (!updates.empty() && plain_write) {
				refusal = "workgroup memory stored and updated by atomics in one divergent region";
				return false;
			}
			for (size_t x = 0; x < updates.size(); x++) {
				for (size_t y = x + 1; y < updates.size(); y++) {
					if (updates[x].second == updates[y].second) continue;
					const auto a = ConstantWord(updates[x].first), c = ConstantWord(updates[y].first);
					if (!a || !c || a->front() != c->front() || *a == *c) {
						refusal = "workgroup memory updated by different atomics in one divergent region";
						return false;
					}
				}
			}
			m_barrier_before.insert(header.end - 1u);
			const auto merge = m_label_block.find(header.merge);
			if (merge == m_label_block.end()) {
				refusal = "divergent region without a merge block";
				return false;
			}
			auto at = m_blocks[merge->second].begin + 1u;
			while (m_insts[at].op == spv::OpPhi || m_insts[at].op == spv::OpLine || m_insts[at].op == spv::OpNoLine) at++;
			m_barrier_before.insert(at);
		}
		return true;
	}

	uint32_t LoopCarriedValues() const {
		uint32_t most = 0;
		for (const auto& block: m_blocks) {
			if (block.merge == 0 || m_insts[block.end - 1].op != spv::OpLoopMerge) continue;
			uint32_t values = 0;
			for (uint32_t i = block.begin + 1; i < block.end && m_insts[i].op == spv::OpPhi; i++) values++;
			most = std::max(most, values);
		}
		return most;
	}

	bool RelievesRegisters(std::string& refusal) const {
		if (m_waves == 1) return true;
		const auto values = LoopCarriedValues();
		if (m_threads <= MaxRelievedThreads && values >= RegisterBoundLoopValues) return true;
		refusal = "several waves whose halves would wait on each other without registers to gain: threads=" +
		          std::to_string(m_threads) + " loop_values=" + std::to_string(values);
		return false;
	}

	static bool Pure(uint32_t op) {
		switch (op) {
			case spv::OpAccessChain:
			case spv::OpInBoundsAccessChain:
			case spv::OpIAdd:
			case spv::OpISub:
			case spv::OpIMul:
			case spv::OpUDiv:
			case spv::OpUMod:
			case spv::OpShiftLeftLogical:
			case spv::OpShiftRightLogical:
			case spv::OpShiftRightArithmetic:
			case spv::OpBitwiseAnd:
			case spv::OpBitwiseOr:
			case spv::OpBitwiseXor:
			case spv::OpNot:
			case spv::OpBitcast:
			case spv::OpUConvert:
			case spv::OpSConvert:
			case spv::OpCompositeExtract:
			case spv::OpCompositeConstruct:
			case spv::OpCopyObject:
			case spv::OpSelect:
			case spv::OpIEqual:
			case spv::OpINotEqual:
			case spv::OpULessThan:
			case spv::OpULessThanEqual:
			case spv::OpUGreaterThan:
			case spv::OpUGreaterThanEqual:
			case spv::OpLogicalAnd:
			case spv::OpLogicalOr:
			case spv::OpLogicalNot:
			case spv::OpLine:
			case spv::OpNoLine: return true;
			default: return false;
		}
	}

	bool m_dirty = true;

	void EmitSharedBarrier() {
		m_dirty = false;
		if (m_waves > 1) {
			EmitHalvesSync(spv::ScopeWorkgroup, spv::MemorySemanticsWorkgroupMemoryMask);
			return;
		}
		Emit(spv::OpControlBarrier, {Const(spv::ScopeWorkgroup), Const(spv::ScopeWorkgroup),
		                             Const(spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask)});
	}

	void EmitHalvesSync(uint32_t memory_scope, uint32_t storage) {
		const uint32_t ordering = static_cast<uint32_t>(spv::MemorySemanticsAcquireMask) |
		                          static_cast<uint32_t>(spv::MemorySemanticsReleaseMask) |
		                          static_cast<uint32_t>(spv::MemorySemanticsAcquireReleaseMask) |
		                          static_cast<uint32_t>(spv::MemorySemanticsSequentiallyConsistentMask);
		const auto semantics = Const((storage & ~ordering) | static_cast<uint32_t>(spv::MemorySemanticsWorkgroupMemoryMask) |
		                             static_cast<uint32_t>(spv::MemorySemanticsAcquireReleaseMask));
		Emit(spv::OpControlBarrier, {Const(spv::ScopeSubgroup), Const(memory_scope), semantics});
		const auto target = U32(spv::OpIAdd, {U32(spv::OpLoad, {m_epoch}), Const(2)});
		Emit(spv::OpStore, {m_epoch, target});
		Value(spv::OpFunctionCall, m_void, {m_sync_fn, m_wave, target});
		Emit(spv::OpMemoryBarrier, {Const(memory_scope), semantics});
	}

	void EmitSyncFunction() {
		const auto wave   = NewId();
		const auto target = NewId();
		const auto start = NewId(), arrive = NewId(), arrived = NewId(), header = NewId(), body = NewId(), next = NewId(),
		           done = NewId();
		Put(*m_out, spv::OpFunction, {m_void, m_sync_fn, spv::FunctionControlMaskNone, m_sync_fn_type});
		Put(*m_out, spv::OpFunctionParameter, {m_u32, wave});
		Put(*m_out, spv::OpFunctionParameter, {m_u32, target});
		Put(*m_out, spv::OpLabel, {start});
		const auto counter = Value(spv::OpAccessChain, m_ptr_wg_u32, {m_sync, wave});
		const auto first   = Bool(spv::OpIEqual, {U32(spv::OpLoad, {m_real_lane}), Const(0)});
		Put(*m_out, spv::OpSelectionMerge, {arrived, spv::SelectionControlMaskNone});
		Put(*m_out, spv::OpBranchConditional, {first, arrive, arrived});
		Put(*m_out, spv::OpLabel, {arrive});
		U32(spv::OpAtomicIAdd, {counter, Const(spv::ScopeWorkgroup),
		                        Const(spv::MemorySemanticsReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask), Const(1)});
		Put(*m_out, spv::OpBranch, {arrived});
		Put(*m_out, spv::OpLabel, {arrived});
		Put(*m_out, spv::OpBranch, {header});
		Put(*m_out, spv::OpLabel, {header});
		Put(*m_out, spv::OpLoopMerge, {done, next, spv::LoopControlMaskNone});
		Put(*m_out, spv::OpBranch, {body});
		Put(*m_out, spv::OpLabel, {body});
		const auto seen = U32(spv::OpAtomicLoad, {counter, Const(spv::ScopeWorkgroup),
		                                         Const(spv::MemorySemanticsAcquireMask | spv::MemorySemanticsWorkgroupMemoryMask)});
		const auto reached = Bool(spv::OpSGreaterThanEqual, {U32(spv::OpISub, {seen, target}), Const(0)});
		Put(*m_out, spv::OpBranchConditional, {reached, done, next});
		Put(*m_out, spv::OpLabel, {next});
		Put(*m_out, spv::OpBranch, {header});
		Put(*m_out, spv::OpLabel, {done});
		Put(*m_out, spv::OpReturn, {});
		Put(*m_out, spv::OpFunctionEnd, {});
	}

	std::vector<uint32_t>*                 m_out = nullptr;
	std::vector<uint32_t>                  m_decls;
	std::unordered_map<uint32_t, uint32_t> m_new_const;

	uint32_t NewId() { return m_bound++; }

	static void Put(std::vector<uint32_t>& to, uint32_t op, std::initializer_list<uint32_t> operands) {
		to.push_back(static_cast<uint32_t>((operands.size() + 1u) << spv::WordCountShift) | op);
		to.insert(to.end(), operands.begin(), operands.end());
	}

	void Emit(uint32_t op, std::initializer_list<uint32_t> operands) { Put(*m_out, op, operands); }

	uint32_t Value(uint32_t op, uint32_t type, std::initializer_list<uint32_t> operands) {
		const auto id = NewId();
		m_out->push_back(static_cast<uint32_t>((operands.size() + 3u) << spv::WordCountShift) | op);
		m_out->push_back(type);
		m_out->push_back(id);
		m_out->insert(m_out->end(), operands.begin(), operands.end());
		return id;
	}

	uint32_t Const(uint32_t value) {
		const auto found = m_new_const.find(value);
		if (found != m_new_const.end()) return found->second;
		const auto id = NewId();
		Put(m_decls, spv::OpConstant, {m_u32, id, value});
		m_new_const[value] = id;
		return id;
	}

	uint32_t m_xchg = 0, m_slot = 0, m_ptr_wg_u32 = 0, m_ptr_priv_u32 = 0;
	uint32_t m_lane = 0, m_wave_lane = 0, m_wave = 0, m_wave_base = 0, m_bit = 0, m_in_lo = 0, m_other = 0, m_scope = 0;
	uint32_t m_sync = 0, m_epoch = 0, m_sync_fn = 0;

	uint32_t U32(uint32_t op, std::initializer_list<uint32_t> operands) { return Value(op, m_u32, operands); }
	uint32_t Bool(uint32_t op, std::initializer_list<uint32_t> operands) { return Value(op, m_bool, operands); }
	uint32_t Glsl(uint32_t instruction, std::initializer_list<uint32_t> operands) {
		const auto id = NewId();
		m_out->push_back(static_cast<uint32_t>((operands.size() + 5u) << spv::WordCountShift) | spv::OpExtInst);
		m_out->insert(m_out->end(), {m_u32, id, m_glsl, instruction});
		m_out->insert(m_out->end(), operands.begin(), operands.end());
		return id;
	}

	uint32_t m_sid = 0, m_real_lane = 0, m_wgid_in = 0, m_v3 = 0;

	void EmitSetup() {
		const auto sid  = U32(spv::OpLoad, {m_sid});
		const auto slii = U32(spv::OpLoad, {m_real_lane});
		m_lane          = U32(spv::OpIAdd, {U32(spv::OpShiftLeftLogical, {sid, Const(5)}), slii});
		m_wave_lane     = U32(spv::OpBitwiseAnd, {m_lane, Const(WaveLanes - 1u)});
		m_wave          = U32(spv::OpShiftRightLogical, {sid, Const(1)});
		m_wave_base     = U32(spv::OpIMul, {m_wave, Const(ExchangeWords())});
		m_bit           = U32(spv::OpShiftLeftLogical, {Const(1), slii});
		m_in_lo         = Bool(spv::OpIEqual, {U32(spv::OpBitwiseAnd, {sid, Const(1)}), Const(0)});
		m_other         = U32(spv::OpBitwiseXor, {m_wave_lane, Const(32)});
		if (m_waves > 1) {
			Emit(spv::OpAtomicStore, {Value(spv::OpAccessChain, m_ptr_wg_u32, {m_sync, m_wave}), Const(spv::ScopeWorkgroup),
			                          Const(spv::MemorySemanticsMaskNone), Const(0)});
			Emit(spv::OpControlBarrier, {Const(spv::ScopeWorkgroup), Const(spv::ScopeWorkgroup),
			                             Const(spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask)});
		}
		if (m_lii != 0) Emit(spv::OpStore, {m_lii, m_lane});
		if (m_subgroup_lane != 0) Emit(spv::OpStore, {m_subgroup_lane, m_wave_lane});
		if (m_lid == 0 && m_gid == 0) return;
		const uint32_t local[3] = {
		    U32(spv::OpUMod, {m_lane, Const(m_size[0])}),
		    U32(spv::OpUMod, {U32(spv::OpUDiv, {m_lane, Const(m_size[0])}), Const(m_size[1])}),
		    U32(spv::OpUDiv, {m_lane, Const(m_size[0] * m_size[1])}),
		};
		if (m_lid != 0) Emit(spv::OpStore, {m_lid, Value(spv::OpCompositeConstruct, m_v3, {local[0], local[1], local[2]})});
		if (m_gid == 0) return;
		const auto group = Value(spv::OpLoad, m_v3, {m_wgid_in});
		uint32_t   global[3] {};
		for (uint32_t k = 0; k < 3; k++) {
			global[k] = U32(spv::OpIAdd,
			                {U32(spv::OpIMul, {U32(spv::OpCompositeExtract, {group, k}), Const(m_size[k])}), local[k]});
		}
		Emit(spv::OpStore, {m_gid, Value(spv::OpCompositeConstruct, m_v3, {global[0], global[1], global[2]})});
	}

	struct Job {
		uint32_t              inst  = 0;
		uint32_t              mine  = 0;
		uint32_t              mine2 = 0;
		std::vector<uint32_t> loaded;
	};

	std::unordered_map<uint32_t, Job>                           m_jobs;
	std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> m_ballots;

	bool IsBallotOp(uint32_t op) const {
		return op == spv::OpGroupNonUniformBallot || op == spv::OpGroupNonUniformAny || op == spv::OpGroupNonUniformAll;
	}

	bool IsShuffleOp(uint32_t op) const {
		return op == spv::OpGroupNonUniformShuffle || op == spv::OpGroupNonUniformBroadcast ||
		       op == spv::OpGroupNonUniformBroadcastFirst;
	}

	bool NeedsExchange(const Inst& inst) const {
		if (IsBallotOp(inst.op)) {
			const auto predicate = Word(inst, 1);
			return Div(predicate) && !m_true.contains(predicate) && !m_false.contains(predicate) &&
			       !m_ballots.contains(predicate);
		}
		if (IsShuffleOp(inst.op)) return Div(Word(inst, 1));
		if (IsReduction(inst.op)) return Div(Word(inst, 2));
		return false;
	}

	bool AvailableAt(const Inst& inst, uint32_t at) const {
		for (uint32_t k = 0; k < Operands(inst); k++) {
			const auto id = Word(inst, k);
			if (id < m_bound && m_def[id] != None && m_def[id] >= at) return false;
		}
		return true;
	}

	uint32_t ShuffleLane(const Inst& inst) {
		return inst.op == spv::OpGroupNonUniformBroadcastFirst ? Const(0) : U32(spv::OpBitwiseAnd, {Word(inst, 2), Const(WaveLanes - 1u)});
	}

	void EmitBatch(uint32_t first) {
		std::vector<uint32_t> group {first};
		std::unordered_set<uint32_t> predicates;
		if (IsBallotOp(m_insts[first].op)) predicates.insert(Word(m_insts[first], 1));
		const auto block = m_block_of[first];
		for (uint32_t j = first + 1; j <= m_blocks[block].end && group.size() < m_batch; j++) {
			const auto& inst = m_insts[j];
			if (!IsSubgroupOp(inst.op) || !NeedsExchange(inst) || !AvailableAt(inst, first)) continue;
			if (IsBallotOp(inst.op) && !predicates.insert(Word(inst, 1)).second) continue;
			group.push_back(j);
		}
		for (const auto j: group) {
			const auto& inst = m_insts[j];
			auto&       job  = m_jobs[j];
			job.inst         = j;
			if (IsBallotOp(inst.op)) {
				const auto mask = Value(spv::OpGroupNonUniformBallot, m_uvec4, {m_scope, Word(inst, 1)});
				job.mine        = U32(spv::OpCompositeExtract, {mask, 0});
			} else if (IsShuffleOp(inst.op)) {
				job.mine = ToWord(inst.type, Word(inst, 1));
			} else {
				job.mine  = Value(inst.op, inst.type, {m_scope, spv::GroupOperationReduce, Word(inst, 2)});
				job.mine2 = ToWord(inst.type, job.mine);
			}
		}
		const auto slot = U32(spv::OpLoad, {m_slot});
		const auto base = U32(spv::OpIAdd, {m_wave_base, U32(spv::OpIMul, {slot, Const(WaveLanes * m_batch)})});
		const auto at   = U32(spv::OpIAdd, {base, m_wave_lane});
		const auto store = [&](uint32_t member, uint32_t value) {
			const auto index = U32(spv::OpIAdd, {at, Const(member * WaveLanes)});
			Emit(spv::OpStore, {Value(spv::OpAccessChain, m_ptr_wg_u32, {m_xchg, index}), value});
		};
		for (uint32_t member = 0; member < group.size(); member++) {
			const auto& inst = m_insts[group[member]];
			auto&       job  = m_jobs[group[member]];
			store(member, IsReduction(inst.op) ? job.mine2 : job.mine);
		}
		std::vector<uint32_t> sources;
		for (const auto j: group) {
			const auto& inst = m_insts[j];
			sources.push_back(IsShuffleOp(inst.op) ? ShuffleLane(inst) : IsReduction(inst.op) ? m_other : 0);
		}
		EmitSharedBarrier();
		const auto load = [&](uint32_t member, uint32_t source) {
			const auto index = U32(spv::OpIAdd, {U32(spv::OpIAdd, {base, source}), Const(member * WaveLanes)});
			return U32(spv::OpLoad, {Value(spv::OpAccessChain, m_ptr_wg_u32, {m_xchg, index})});
		};
		for (uint32_t member = 0; member < group.size(); member++) {
			const auto& inst = m_insts[group[member]];
			auto&       job  = m_jobs[group[member]];
			if (IsBallotOp(inst.op)) {
				m_ballots[Word(inst, 1)] = {load(member, Const(0)), load(member, Const(32))};
				continue;
			}
			job.loaded.push_back(load(member, sources[member]));
		}
		Emit(spv::OpStore, {m_slot, U32(spv::OpBitwiseXor, {slot, Const(1)})});
	}

	std::pair<uint32_t, uint32_t> BallotWords(uint32_t predicate) {
		if (m_true.contains(predicate)) return {Const(UINT32_MAX), Const(UINT32_MAX)};
		if (m_false.contains(predicate)) return {Const(0), Const(0)};
		if (const auto found = m_ballots.find(predicate); found != m_ballots.end()) return found->second;
		const auto all = U32(spv::OpSelect, {predicate, Const(UINT32_MAX), Const(0)});
		return {all, all};
	}

	void EmitLastLaneStore(const Inst& store) {
		const auto type  = ValueType(Word(store, 1));
		const auto slot  = U32(spv::OpLoad, {m_slot});
		const auto base  = U32(spv::OpIAdd, {m_wave_base, U32(spv::OpIMul, {slot, Const(WaveLanes * m_batch)})});
		const auto entry = [&](uint32_t lane) { return Value(spv::OpAccessChain, m_ptr_wg_u32, {m_xchg, U32(spv::OpIAdd, {base, lane})}); };
		Emit(spv::OpStore, {entry(m_wave_lane), ToWord(type, Word(store, 1))});
		EmitSharedBarrier();
		const auto last = FromWord(type, U32(spv::OpLoad, {entry(Const(WaveLanes - 1u))}));
		Emit(spv::OpStore, {m_slot, U32(spv::OpBitwiseXor, {slot, Const(1)})});
		m_out->push_back(static_cast<uint32_t>(store.count << spv::WordCountShift) | spv::OpStore);
		m_out->push_back(Word(store, 0));
		m_out->push_back(last);
		for (uint32_t k = 2; k < Operands(store); k++) m_out->push_back(Word(store, k));
	}

	uint32_t ToWord(uint32_t type, uint32_t value) {
		if (type == m_u32) return value;
		if (type == m_bool) return U32(spv::OpSelect, {value, Const(1), Const(0)});
		return U32(spv::OpBitcast, {value});
	}

	uint32_t FromWord(uint32_t type, uint32_t word) {
		if (type == m_u32) return word;
		if (type == m_bool) return Bool(spv::OpINotEqual, {word, Const(0)});
		return Value(spv::OpBitcast, type, {word});
	}

	uint32_t Combine(const Inst& inst, uint32_t mine, uint32_t theirs) {
		const auto ext = [&](uint32_t instruction) {
			const auto id = NewId();
			m_out->push_back(static_cast<uint32_t>(7u << spv::WordCountShift) | spv::OpExtInst);
			m_out->insert(m_out->end(), {inst.type, id, m_glsl, instruction, mine, theirs});
			return id;
		};
		switch (inst.op) {
			case spv::OpGroupNonUniformIAdd: return Value(spv::OpIAdd, inst.type, {mine, theirs});
			case spv::OpGroupNonUniformSMin: return ext(GLSLstd450SMin);
			case spv::OpGroupNonUniformUMin: return ext(GLSLstd450UMin);
			case spv::OpGroupNonUniformSMax: return ext(GLSLstd450SMax);
			case spv::OpGroupNonUniformUMax: return ext(GLSLstd450UMax);
			case spv::OpGroupNonUniformBitwiseAnd: return Value(spv::OpBitwiseAnd, inst.type, {mine, theirs});
			case spv::OpGroupNonUniformBitwiseOr: return Value(spv::OpBitwiseOr, inst.type, {mine, theirs});
			case spv::OpGroupNonUniformBitwiseXor: return Value(spv::OpBitwiseXor, inst.type, {mine, theirs});
			case spv::OpGroupNonUniformLogicalAnd: return Value(spv::OpLogicalAnd, inst.type, {mine, theirs});
			case spv::OpGroupNonUniformLogicalOr: return Value(spv::OpLogicalOr, inst.type, {mine, theirs});
			default: return Value(spv::OpLogicalNotEqual, inst.type, {mine, theirs});
		}
	}

	void FinishReduce(const Inst& inst, uint32_t mine, uint32_t theirs) {
		Emit(spv::OpCopyObject, {inst.type, inst.result, Combine(inst, mine, theirs)});
	}

	void UniformReduce(const Inst& inst) {
		const auto value = Word(inst, 2);
		switch (inst.op) {
			case spv::OpGroupNonUniformIAdd:
				Emit(spv::OpIMul, {inst.type, inst.result, value, FromWord(inst.type, Const(WaveLanes))});
				break;
			case spv::OpGroupNonUniformBitwiseXor:
				Emit(spv::OpCopyObject, {inst.type, inst.result, FromWord(inst.type, Const(0))});
				break;
			case spv::OpGroupNonUniformLogicalXor:
				Emit(spv::OpINotEqual, {inst.type, inst.result, Const(0), Const(0)});
				break;
			default: Emit(spv::OpCopyObject, {inst.type, inst.result, value}); break;
		}
	}

	void RewriteSubgroupOp(uint32_t index) {
		const auto& inst = m_insts[index];
		if (!m_jobs.contains(index) && NeedsExchange(inst)) EmitBatch(index);
		const auto words_of = [&](uint32_t vector) {
			return std::pair {U32(spv::OpCompositeExtract, {vector, 0}), U32(spv::OpCompositeExtract, {vector, 1})};
		};
		const auto job = m_jobs.find(index);
		switch (inst.op) {
			case spv::OpGroupNonUniformElect: Emit(spv::OpIEqual, {inst.type, inst.result, m_wave_lane, Const(0)}); break;
			case spv::OpGroupNonUniformBallot: {
				const auto [lo, hi] = BallotWords(Word(inst, 1));
				Emit(spv::OpCompositeConstruct, {inst.type, inst.result, lo, hi, Const(0), Const(0)});
				break;
			}
			case spv::OpGroupNonUniformAny: {
				const auto [lo, hi] = BallotWords(Word(inst, 1));
				Emit(spv::OpINotEqual, {inst.type, inst.result, U32(spv::OpBitwiseOr, {lo, hi}), Const(0)});
				break;
			}
			case spv::OpGroupNonUniformAll: {
				const auto [lo, hi] = BallotWords(Word(inst, 1));
				Emit(spv::OpIEqual, {inst.type, inst.result, U32(spv::OpBitwiseAnd, {lo, hi}), Const(UINT32_MAX)});
				break;
			}
			case spv::OpGroupNonUniformBroadcastFirst:
			case spv::OpGroupNonUniformBroadcast:
			case spv::OpGroupNonUniformShuffle:
				Emit(spv::OpCopyObject,
				     {inst.type, inst.result, job == m_jobs.end() ? Word(inst, 1) : FromWord(inst.type, job->second.loaded[0])});
				break;
			case spv::OpGroupNonUniformBallotBitExtract: {
				const auto index_id = Word(inst, 2);
				const auto word     = U32(spv::OpVectorExtractDynamic,
				                          {Word(inst, 1), U32(spv::OpBitwiseAnd, {U32(spv::OpShiftRightLogical, {index_id, Const(5)}), Const(3)})});
				const auto bit      = U32(spv::OpBitwiseAnd,
				                          {U32(spv::OpShiftRightLogical, {word, U32(spv::OpBitwiseAnd, {index_id, Const(31)})}),
				                           Const(1)});
				Emit(spv::OpINotEqual, {inst.type, inst.result, bit, Const(0)});
				break;
			}
			case spv::OpGroupNonUniformBallotBitCount: {
				auto [lo, hi]        = words_of(Word(inst, 2));
				const auto operation = Word(inst, 1);
				if (operation != spv::GroupOperationReduce) {
					auto below = U32(spv::OpISub, {m_bit, Const(1)});
					if (operation == spv::GroupOperationInclusiveScan) below = U32(spv::OpBitwiseOr, {below, m_bit});
					lo = U32(spv::OpBitwiseAnd, {lo, U32(spv::OpSelect, {m_in_lo, below, Const(UINT32_MAX)})});
					hi = U32(spv::OpBitwiseAnd, {hi, U32(spv::OpSelect, {m_in_lo, Const(0), below})});
				}
				Emit(spv::OpIAdd, {inst.type, inst.result, U32(spv::OpBitCount, {lo}), U32(spv::OpBitCount, {hi})});
				break;
			}
			case spv::OpGroupNonUniformBallotFindLSB: {
				const auto [lo, hi] = words_of(Word(inst, 1));
				const auto high     = U32(spv::OpSelect, {Bool(spv::OpINotEqual, {hi, Const(0)}),
				                                          U32(spv::OpIAdd, {Glsl(GLSLstd450FindILsb, {hi}), Const(32)}),
				                                          Const(UINT32_MAX)});
				Emit(spv::OpSelect, {inst.type, inst.result, Bool(spv::OpINotEqual, {lo, Const(0)}),
				                     Glsl(GLSLstd450FindILsb, {lo}), high});
				break;
			}
			case spv::OpGroupNonUniformBallotFindMSB: {
				const auto [lo, hi] = words_of(Word(inst, 1));
				const auto high     = U32(spv::OpIAdd, {Glsl(GLSLstd450FindUMsb, {hi}), Const(32)});
				Emit(spv::OpSelect, {inst.type, inst.result, Bool(spv::OpINotEqual, {hi, Const(0)}), high,
				                     Glsl(GLSLstd450FindUMsb, {lo})});
				break;
			}
			default:
				if (job == m_jobs.end()) {
					UniformReduce(inst);
				} else {
					FinishReduce(inst, job->second.mine, FromWord(inst.type, job->second.loaded[0]));
				}
				break;
		}
	}

	uint32_t FindSyncFunctionType() const {
		for (uint32_t i = 0; i < m_first_function; i++) {
			const auto& inst = m_insts[i];
			if (inst.op == spv::OpTypeFunction && Operands(inst) == 3u && Word(inst, 0) == m_void &&
			    Word(inst, 1) == m_u32 && Word(inst, 2) == m_u32) {
				return inst.result;
			}
		}
		return 0;
	}

	uint32_t Scope(uint32_t id) { return m_const.contains(id) && m_const[id] == spv::ScopeSubgroup ? Const(spv::ScopeWorkgroup) : id; }

	void Rewrite(std::vector<uint32_t>& out) {
		m_out = &out;
		out.assign(m_in.begin(), m_in.begin() + 5);
		m_scope        = Const(spv::ScopeSubgroup);
		const auto arr = NewId();
		Put(m_decls, spv::OpTypeArray, {arr, m_u32, Const(m_waves * ExchangeWords())});
		const auto ptr_arr = NewId();
		Put(m_decls, spv::OpTypePointer, {ptr_arr, spv::StorageClassWorkgroup, arr});
		m_ptr_wg_u32 = NewId();
		Put(m_decls, spv::OpTypePointer, {m_ptr_wg_u32, spv::StorageClassWorkgroup, m_u32});
		m_xchg = NewId();
		Put(m_decls, spv::OpVariable, {ptr_arr, m_xchg, spv::StorageClassWorkgroup});
		m_ptr_priv_u32 = NewId();
		Put(m_decls, spv::OpTypePointer, {m_ptr_priv_u32, spv::StorageClassPrivate, m_u32});
		m_slot = NewId();
		Put(m_decls, spv::OpVariable, {m_ptr_priv_u32, m_slot, spv::StorageClassPrivate, Const(0)});
		if (m_waves > 1) {
			const auto sync_arr = NewId();
			Put(m_decls, spv::OpTypeArray, {sync_arr, m_u32, Const(m_waves)});
			const auto ptr_sync = NewId();
			Put(m_decls, spv::OpTypePointer, {ptr_sync, spv::StorageClassWorkgroup, sync_arr});
			m_sync = NewId();
			Put(m_decls, spv::OpVariable, {ptr_sync, m_sync, spv::StorageClassWorkgroup});
			m_epoch = NewId();
			Put(m_decls, spv::OpVariable, {m_ptr_priv_u32, m_epoch, spv::StorageClassPrivate, Const(0)});
			m_sync_fn_type = FindSyncFunctionType();
			if (m_sync_fn_type == 0) {
				m_sync_fn_type = NewId();
				Put(m_decls, spv::OpTypeFunction, {m_sync_fn_type, m_void, m_u32, m_u32});
			}
			m_sync_fn = NewId();
		}
		std::vector<uint32_t> decorations;
		std::vector<uint32_t> interface_ids;
		uint32_t              first_global = m_memory_model + 1u;
		while (first_global < m_insts.size()) {
			const auto op = m_insts[first_global].op;
			if (op != spv::OpEntryPoint && op != spv::OpExecutionMode && op != spv::OpExecutionModeId &&
			    op != spv::OpString && op != spv::OpSourceExtension && op != spv::OpSource &&
			    op != spv::OpSourceContinued && op != spv::OpName && op != spv::OpMemberName &&
			    op != spv::OpModuleProcessed && op != spv::OpLine && op != spv::OpNoLine) {
				break;
			}
			first_global++;
		}
		if (m_uvec4 == 0) {
			m_uvec4 = NewId();
			Put(m_decls, spv::OpTypeVector, {m_uvec4, m_u32, 4});
		}
		const auto ptr_in_u32 = NewId();
		Put(m_decls, spv::OpTypePointer, {ptr_in_u32, spv::StorageClassInput, m_u32});
		const auto input = [&](uint32_t pointer, uint32_t builtin) {
			const auto id = NewId();
			Put(m_decls, spv::OpVariable, {pointer, id, spv::StorageClassInput});
			Put(decorations, spv::OpDecorate, {id, spv::DecorationBuiltIn, builtin});
			return id;
		};
		m_sid       = input(ptr_in_u32, spv::BuiltInSubgroupId);
		m_real_lane = input(ptr_in_u32, spv::BuiltInSubgroupLocalInvocationId);
		std::unordered_set<uint32_t> moved;
		for (const auto variable: {m_lii, m_subgroup_lane, m_lid, m_gid}) {
			if (variable != 0) moved.insert(variable);
		}
		const auto pointee_of = [&](uint32_t variable) { return m_ptr_pointee[m_insts[m_def[variable]].type]; };
		if (m_lid != 0) m_v3 = pointee_of(m_lid);
		if (m_gid != 0) {
			m_v3 = pointee_of(m_gid);
			if (m_wgid != 0) {
				m_wgid_in = m_wgid;
			} else {
				const auto ptr_in_v3 = NewId();
				Put(m_decls, spv::OpTypePointer, {ptr_in_v3, spv::StorageClassInput, m_v3});
				m_wgid_in = input(ptr_in_v3, spv::BuiltInWorkgroupId);
			}
		}
		std::unordered_map<uint32_t, uint32_t> private_ptr;
		const auto private_of = [&](uint32_t input_ptr, std::vector<uint32_t>& to) {
			const auto id = NewId();
			Put(to, spv::OpTypePointer, {id, spv::StorageClassPrivate, m_ptr_pointee[input_ptr]});
			return id;
		};
		std::unordered_map<uint32_t, uint32_t> chain_type;
		for (uint32_t i = m_first_function; i < m_insts.size(); i++) {
			const auto& inst = m_insts[i];
			if ((inst.op == spv::OpAccessChain || inst.op == spv::OpInBoundsAccessChain) && moved.contains(Root(Word(inst, 0)))) {
				auto& pointer = private_ptr[inst.type];
				if (pointer == 0) pointer = private_of(inst.type, m_decls);
				chain_type[i] = pointer;
			}
		}
		std::vector<uint32_t> globals;
		for (const auto& [variable, storage]: m_var_class) {
			if (storage != spv::StorageClassFunction) globals.push_back(variable);
		}
		std::sort(globals.begin(), globals.end());
		interface_ids = globals;
		interface_ids.insert(interface_ids.end(), {m_xchg, m_slot, m_sid, m_real_lane});
		if (m_waves > 1) interface_ids.insert(interface_ids.end(), {m_sync, m_epoch});
		if (m_gid != 0 && m_wgid == 0) interface_ids.push_back(m_wgid_in);
		bool need_glsl = m_glsl == 0;
		if (need_glsl) m_glsl = NewId();
		const auto& entry      = m_functions[m_entry_function];
		const auto  entry_block = m_blocks.empty() ? None : m_blocks[0].begin;
		bool        setup_due   = false;
		size_t      decl_at = 0, deco_at = 0;
		for (uint32_t i = 0; i < m_insts.size(); i++) {
			const auto& inst = m_insts[i];
			const auto  src  = std::span(m_in).subspan(inst.offset, inst.count);
			if (i == m_memory_model && need_glsl) {
				out.push_back((6u << spv::WordCountShift) | spv::OpExtInstImport);
				out.push_back(m_glsl);
				const char name[16] = "GLSL.std.450";
				uint32_t   packed[4] {};
				std::memcpy(packed, name, sizeof(packed));
				out.insert(out.end(), packed, packed + 4);
			}
			if (i == m_first_function) decl_at = out.size();
			if (m_last_decoration == None ? i == first_global : i == m_last_decoration + 1u) deco_at = out.size();
			if (inst.op == spv::OpCapability && !m_caps_done) {
				out.insert(out.end(), src.begin(), src.end());
				for (const auto capability: {spv::CapabilityGroupNonUniform, spv::CapabilityGroupNonUniformBallot,
				                             spv::CapabilityGroupNonUniformArithmetic}) {
					if (!m_caps.contains(capability)) Put(out, spv::OpCapability, {capability});
				}
				m_caps_done = true;
				continue;
			}
			if (inst.op == spv::OpEntryPoint) {
				uint32_t name_end = 3;
				while (name_end < inst.count && (src[name_end] >> 24u) != 0u) name_end++;
				name_end++;
				out.push_back(static_cast<uint32_t>((name_end + interface_ids.size()) << spv::WordCountShift) | inst.op);
				out.insert(out.end(), src.begin() + 1, src.begin() + name_end);
				out.insert(out.end(), interface_ids.begin(), interface_ids.end());
				continue;
			}
			if (i == m_local_size) {
				Put(out, spv::OpExecutionMode, {m_entry, spv::ExecutionModeLocalSize, m_threads, 1, 1});
				continue;
			}
			if (inst.op == spv::OpDecorate && moved.contains(Word(inst, 0)) && Word(inst, 1) == spv::DecorationBuiltIn) {
				continue;
			}
			if (inst.op == spv::OpVariable && moved.contains(inst.result)) {
				std::vector<uint32_t> type_decl;
				const auto            pointer = private_of(inst.type, type_decl);
				out.insert(out.end(), type_decl.begin(), type_decl.end());
				Put(out, spv::OpVariable, {pointer, inst.result, spv::StorageClassPrivate});
				continue;
			}
			const bool in_entry = i >= entry.begin && i <= entry.end;
			if (setup_due && inst.op != spv::OpVariable && inst.op != spv::OpLine && inst.op != spv::OpNoLine) {
				EmitSetup();
				setup_due = false;
			}
			if (in_entry && inst.op == spv::OpLabel) m_ballots.clear();
			if (m_barrier_before.contains(i) && m_dirty && !m_last_lane_stores.contains(i)) EmitSharedBarrier();
			if (const auto found = chain_type.find(i); found != chain_type.end()) {
				out.insert(out.end(), src.begin(), src.end());
				out[out.size() - inst.count + 1] = found->second;
			} else if (m_last_lane_stores.contains(i)) {
				EmitLastLaneStore(inst);
			} else if (in_entry && IsSubgroupOp(inst.op)) {
				RewriteSubgroupOp(i);
			} else if (inst.op == spv::OpControlBarrier && m_waves > 1 && ScopeOf(inst) == spv::ScopeSubgroup) {
				const auto semantics = m_const.find(m_in[inst.first + 2]);
				EmitHalvesSync(m_const.contains(m_in[inst.first + 1]) && m_const[m_in[inst.first + 1]] == spv::ScopeDevice
				                   ? spv::ScopeDevice
				                   : spv::ScopeWorkgroup,
				               semantics == m_const.end() ? 0u : semantics->second);
			} else if (inst.op == spv::OpControlBarrier) {
				Put(out, spv::OpControlBarrier, {Scope(m_in[inst.first]), Scope(m_in[inst.first + 1]), m_in[inst.first + 2]});
			} else if (inst.op == spv::OpMemoryBarrier) {
				Put(out, spv::OpMemoryBarrier, {Scope(m_in[inst.first]), m_in[inst.first + 1]});
			} else {
				out.insert(out.end(), src.begin(), src.end());
			}
			if (!Pure(inst.op)) m_dirty = true;
			if (m_barrier_after.contains(i)) EmitSharedBarrier();
			if (i == entry_block) setup_due = true;
		}
		if (m_waves > 1) EmitSyncFunction();
		out.insert(out.begin() + static_cast<std::ptrdiff_t>(decl_at), m_decls.begin(), m_decls.end());
		if (!decorations.empty()) {
			const auto at = deco_at;
			out.insert(out.begin() + static_cast<std::ptrdiff_t>(at), decorations.begin(), decorations.end());
		}
		out[1] = std::max(out[1], 0x00010600u);
		out[3] = m_bound;
	}
};

}

bool WaveSplitThreads(const uint32_t (&threads)[3]) {
	const uint64_t total = uint64_t {std::max(threads[0], 1u)} * std::max(threads[1], 1u) * std::max(threads[2], 1u);
	return total % WaveLanes == 0 && total <= MaxThreads;
}

bool SplitWaveAcrossSubgroups(std::vector<uint32_t>& module, std::string& refusal, const SplitHints* hints) {
	std::vector<uint32_t> out;
	WaveSplit             split(module, hints);
	if (!split.Run(out, refusal)) {
		return false;
	}
	module = std::move(out);
	return true;
}

}
