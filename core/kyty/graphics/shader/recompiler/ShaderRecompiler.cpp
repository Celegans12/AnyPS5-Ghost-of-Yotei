#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ShaderCalls.h"
#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvWaveSplit.h"
#include "graphics/shader/recompiler/ir/passes/TableMode.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryOracle.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "graphics/shader/recompiler/ir/passes/RelativeVgprLowering.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <span>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler {

namespace {

std::string MakeIrDump(std::string_view cfg, const IR::Program& ir) {
	std::string dump = "CFG:\n";
	dump += cfg;
	dump += "\nIR:\n";
	dump += fmt::format("mode={} scratch_dwords={}\n",
	                    ir.dispatcher_fallback ? "dispatcher" : "structured", ir.scratch_dwords);
	dump += IR::ProgramToString(ir);
	return dump;
}

void LogDispatcherFallback(const CompileOptions& options, const CFG::Graph& cfg, const char* phase) {
	const auto* block        = cfg.FindBlock(cfg.failure_block);
	const auto  start        = block != nullptr ? block->start_pc : UINT32_MAX;
	const auto  end          = block != nullptr ? block->end_pc : UINT32_MAX;
	const auto  predecessors = block != nullptr ? block->predecessors.size() : 0u;
	const auto  successors   = block != nullptr ? block->successors.size() : 0u;
	LOGF("ShaderRecompiler %s CFG dispatcher fallback: hash=0x%016" PRIx64
	     " phase=%s failure=%s block=%" PRIu32 " pc=0x%08" PRIx32 "..0x%08" PRIx32 " preds=%" PRIu64
	     " succs=%" PRIu64 " blocks=%" PRIu64 " loops=%" PRIu64 " back_edges=%" PRIu64
	     " reason=%s\n",
	     StageName(options.stage), options.shader_hash, phase,
	     CFG::FailureKindToString(cfg.failure_kind).c_str(), cfg.failure_block, start, end,
	     static_cast<uint64_t>(predecessors), static_cast<uint64_t>(successors),
	     static_cast<uint64_t>(cfg.blocks.size()), static_cast<uint64_t>(cfg.natural_loops.size()),
	     static_cast<uint64_t>(cfg.back_edges.size()), cfg.unsupported_reason.c_str());
}

enum class EmbeddedFetchValueType {
	Unknown,
	Constant,
	AttribTable,
	Attrib,
	BufferTable,
	Buffer
};

struct EmbeddedFetchSgprInfo {
	EmbeddedFetchValueType type      = EmbeddedFetchValueType::Unknown;
	int                    attrib_id = 0;
	uint32_t               value     = 0;
};

using EmbeddedFetchVectorLanes = std::map<uint64_t, EmbeddedFetchSgprInfo>;

uint64_t EmbeddedFetchVectorLaneKey(uint32_t reg, uint32_t lane) {
	return (static_cast<uint64_t>(reg) << 32u) | lane;
}

uint32_t EmbeddedFetchLane(uint32_t lane, uint32_t wave_size) {
	return wave_size == 32 || wave_size == 64 ? lane % wave_size : lane;
}

void ClearEmbeddedFetchVectorLanes(EmbeddedFetchVectorLanes* lanes, uint32_t reg) {
	const auto first = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg, 0));
	const auto last  = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg + 1u, 0));
	lanes->erase(first, last);
}

bool IsDecodedSgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Sgpr || op.kind == Decoder::OperandKind::VccLo ||
	       op.kind == Decoder::OperandKind::VccHi;
}

uint32_t DecodedSgprReg(const Decoder::Operand& op) {
	switch (op.kind) {
		case Decoder::OperandKind::VccLo: return 106u;
		case Decoder::OperandKind::VccHi: return 107u;
		default: return op.reg;
	}
}

bool IsDecodedVgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Vgpr;
}

uint32_t DecodedDstSize(const Decoder::Instruction& inst) {
	return std::max(inst.data_dwords, 1u);
}

uint32_t EmbeddedFetchDstSize(const Decoder::Instruction& inst) {
	return inst.opcode == Decoder::Opcode::V_MAD_U64_U32 ? 2u : DecodedDstSize(inst);
}

void ClearEmbeddedFetchSgprs(std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                             const Decoder::Operand& dst, uint32_t size) {
	if (!IsDecodedSgpr(dst)) {
		return;
	}
	const auto register_id = DecodedSgprReg(dst);
	for (uint32_t i = 0; i < size && register_id + i < sgprs.size(); i++) {
		sgprs[register_id + i] = {};
	}
}

bool TryDecodedOperandConstant(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                               const Decoder::Operand& op, uint32_t& value) {
	switch (op.kind) {
		case Decoder::OperandKind::LiteralConstant:
		case Decoder::OperandKind::IntegerInlineConstant:
		case Decoder::OperandKind::FloatInlineConstant: value = op.value; return true;
		case Decoder::OperandKind::Null: value = 0; return true;
		default: break;
	}
	if (IsDecodedSgpr(op) && DecodedSgprReg(op) < sgprs.size() &&
	    sgprs[DecodedSgprReg(op)].type == EmbeddedFetchValueType::Constant) {
		value = sgprs[DecodedSgprReg(op)].value;
		return true;
	}
	return false;
}

bool TryDecodedSmemOffset(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                          const Decoder::Instruction& inst, uint32_t& raw_offset) {
	uint32_t base = 0;
	if (!TryDecodedOperandConstant(sgprs, inst.src1, base)) {
		return false;
	}
	const auto value = static_cast<uint64_t>(base) + inst.offset;
	if (value > 0xffffffffull) {
		return false;
	}
	raw_offset = static_cast<uint32_t>(value);
	return true;
}

bool IsEmbeddedFetchSLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_LOAD_DWORD:
		case Decoder::Opcode::S_LOAD_DWORDX2:
		case Decoder::Opcode::S_LOAD_DWORDX4:
		case Decoder::Opcode::S_LOAD_DWORDX8:
		case Decoder::Opcode::S_LOAD_DWORDX16: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchBufferLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_X:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XY:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZ:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZW: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchAttribPropagationAlu(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_BFE_U32:
		case Decoder::Opcode::S_AND_B32:
		case Decoder::Opcode::S_ADD_I32:
		case Decoder::Opcode::S_ADD_U32:
		case Decoder::Opcode::S_LSHL_B32: return true;
		default: return false;
	}
}

int BufferTableAttribFromOffset(uint32_t raw_offset, int dword) {
	return static_cast<int>((raw_offset + static_cast<uint32_t>(dword) * 4u) / 16u);
}

Frontend::EmbeddedFetchPlan DetectEmbeddedVertexFetch(
    const Decoder::Program& decoded, const ShaderVertexInputInfo& input_info,
    uint32_t user_data_base, uint32_t user_data_count, uint32_t wave_size) {
	const uint32_t    vertex_index_reg   = input_info.logical_stage == ShaderType::Local ? 2u : 5u;
	const uint32_t    instance_index_reg = input_info.logical_stage == ShaderType::Local ? 5u : 8u;
	Frontend::EmbeddedFetchPlan data;
	data.loads.reserve(input_info.resources_num);
	int32_t vertex_offset_candidate   = -1;
	int32_t instance_offset_candidate = -1;
	bool    vertex_offset_conflict    = false;
	bool    instance_offset_conflict  = false;

	const int shift_regs = 8;
	const int attrib_reg = input_info.fetch_attrib_reg + shift_regs;
	const int buffer_reg = input_info.fetch_buffer_reg + shift_regs;

	std::array<EmbeddedFetchSgprInfo, 108> sgprs {};
	std::array<bool, 256>                 vgpr_is_index {};
	EmbeddedFetchVectorLanes               vector_lanes;
	const bool                             track_vector_lanes =
	    std::none_of(decoded.instructions.begin(), decoded.instructions.end(),
	                 [](const auto& inst) {
		                 return Decoder::IsDirectBranch(inst.opcode) ||
		                        inst.opcode == Decoder::Opcode::S_SETPC_B64;
	                 });

	if (attrib_reg >= 0 && attrib_reg < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg].type = EmbeddedFetchValueType::AttribTable;
	}
	if (attrib_reg + 1 >= 0 && attrib_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg + 1].type = EmbeddedFetchValueType::AttribTable;
	}
	if (buffer_reg >= 0 && buffer_reg < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg].type = EmbeddedFetchValueType::BufferTable;
	}
	if (buffer_reg + 1 >= 0 && buffer_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg + 1].type = EmbeddedFetchValueType::BufferTable;
	}

	for (const auto& inst: decoded.instructions) {
		const bool vertex_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == 0 || (user_data_base == 8 && inst.dst.reg == vertex_index_reg));
		const bool instance_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == (user_data_base == 8 ? instance_index_reg : 3u));
		uint32_t   sad_zero = 0;
		const bool index_offset_add =
		    (vertex_index_accumulator || instance_index_accumulator) && IsDecodedSgpr(inst.src0) &&
		    ((inst.opcode == Decoder::Opcode::V_ADD_I32 && IsDecodedVgpr(inst.src1) &&
		      inst.src1.reg == inst.dst.reg) ||
		     (user_data_base == 8 &&
		      (inst.dst.reg == vertex_index_reg || inst.dst.reg == instance_index_reg) &&
		      inst.opcode == Decoder::Opcode::V_SAD_U32 && !inst.dst.clamp &&
		      IsDecodedVgpr(inst.src2) && inst.src2.reg == inst.dst.reg &&
		      TryDecodedOperandConstant(sgprs, inst.src1, sad_zero) && sad_zero == 0));
		if (data.loads.empty() && index_offset_add) {
			const auto reg = DecodedSgprReg(inst.src0);
			if (reg >= user_data_base && reg - user_data_base < user_data_count) {
				auto& candidate = vertex_index_accumulator ? vertex_offset_candidate
				                                           : instance_offset_candidate;
				auto& conflict  = vertex_index_accumulator ? vertex_offset_conflict
				                                           : instance_offset_conflict;
				if (candidate >= 0 && candidate != static_cast<int32_t>(reg)) {
					conflict = true;
				} else {
					candidate = static_cast<int32_t>(reg);
				}
			}
		}
		switch (inst.opcode) {
			case Decoder::Opcode::V_WRITELANE_B32: {
				uint32_t lane = 0;
				if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
					vgpr_is_index[inst.dst.reg] = false;
				}
				if (track_vector_lanes && IsDecodedVgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size() &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					vector_lanes[EmbeddedFetchVectorLaneKey(inst.dst.reg,
					                                        EmbeddedFetchLane(lane, wave_size))] =
					    sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedVgpr(inst.dst)) {
					ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
				}
				break;
			}
			case Decoder::Opcode::V_READLANE_B32: {
				uint32_t lane = 0;
				if (track_vector_lanes && IsDecodedSgpr(inst.dst) &&
				    DecodedSgprReg(inst.dst) < sgprs.size() && IsDecodedVgpr(inst.src0) &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					const auto found = vector_lanes.find(EmbeddedFetchVectorLaneKey(
					    inst.src0.reg, EmbeddedFetchLane(lane, wave_size)));
					sgprs[DecodedSgprReg(inst.dst)] =
					    found != vector_lanes.end() ? found->second : EmbeddedFetchSgprInfo {};
				} else if (IsDecodedSgpr(inst.dst)) {
					ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
				}
				break;
			}
			case Decoder::Opcode::S_MOV_B32:
				if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size()) {
					sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedSgpr(inst.dst)) {
					uint32_t value = 0;
					if (TryDecodedOperandConstant(sgprs, inst.src0, value)) {
						auto& dst = sgprs[DecodedSgprReg(inst.dst)];
						dst.type  = EmbeddedFetchValueType::Constant;
						dst.value = value;
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
					}
				}
				break;
			case Decoder::Opcode::S_MOVK_I32:
				if (IsDecodedSgpr(inst.dst)) {
					auto& dst = sgprs[DecodedSgprReg(inst.dst)];
					dst.type  = EmbeddedFetchValueType::Constant;
					dst.value = inst.src0.value;
				}
				break;
			default:
				if (IsEmbeddedFetchSLoad(inst)) {
					if (IsDecodedSgpr(inst.src0) && DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type ==
					        EmbeddedFetchValueType::AttribTable) {
						uint32_t raw_offset = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							const auto register_id = DecodedSgprReg(inst.dst);
							const int  index       = static_cast<int>(raw_offset / 4u);
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Attrib;
								dst.attrib_id    = index + static_cast<int>(i);
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else if (IsDecodedSgpr(inst.src0) &&
					           DecodedSgprReg(inst.src0) < sgprs.size() &&
					           sgprs[DecodedSgprReg(inst.src0)].type ==
					               EmbeddedFetchValueType::BufferTable) {
						const auto register_id = DecodedSgprReg(inst.dst);
						uint32_t   raw_offset  = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst = sgprs[register_id + i];
								dst.type  = EmbeddedFetchValueType::Buffer;
								dst.attrib_id =
								    BufferTableAttribFromOffset(raw_offset, static_cast<int>(i));
							}
						} else if (IsDecodedSgpr(inst.src1) &&
						           DecodedSgprReg(inst.src1) < sgprs.size() &&
						           sgprs[DecodedSgprReg(inst.src1)].type ==
						               EmbeddedFetchValueType::Attrib &&
						           (inst.offset & 0x3u) == 0) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Buffer;
								dst.attrib_id    = sgprs[DecodedSgprReg(inst.src1)].attrib_id;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
					}
				} else if (inst.opcode == Decoder::Opcode::V_CNDMASK_B32) {
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
						ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
					}
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size() &&
					    IsDecodedVgpr(inst.src0) && inst.src0.reg == instance_index_reg &&
					    IsDecodedVgpr(inst.src1) && inst.src1.reg == vertex_index_reg) {
						vgpr_is_index[inst.dst.reg] = true;
					}
				} else if (IsEmbeddedFetchAttribPropagationAlu(inst)) {
					if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
					    DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type == EmbeddedFetchValueType::Attrib) {
						sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
					} else if (IsDecodedSgpr(inst.dst)) {
						uint32_t src0 = 0;
						uint32_t src1 = 0;
						if (TryDecodedOperandConstant(sgprs, inst.src0, src0) &&
						    TryDecodedOperandConstant(sgprs, inst.src1, src1)) {
							auto& dst = sgprs[DecodedSgprReg(inst.dst)];
							dst.type  = EmbeddedFetchValueType::Constant;
							switch (inst.opcode) {
								case Decoder::Opcode::S_AND_B32: dst.value = src0 & src1; break;
								case Decoder::Opcode::S_LSHL_B32:
									dst.value = src0 << (src1 & 31u);
									break;
								case Decoder::Opcode::S_BFE_U32:
									dst.value = src0 >> (src1 & 31u);
									break;
								default: dst.value = src0 + src1; break;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
						}
					}
				} else if (IsEmbeddedFetchBufferLoad(inst)) {
					if (IsDecodedVgpr(inst.src0) && inst.src0.reg < vgpr_is_index.size() &&
					    vgpr_is_index[inst.src0.reg] &&
					    IsDecodedSgpr(inst.src1) && DecodedSgprReg(inst.src1) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src1)].type == EmbeddedFetchValueType::Buffer) {
						const auto& buffer = sgprs[DecodedSgprReg(inst.src1)];
						if (data.loads.empty()) {
							if (!vertex_offset_conflict) {
								data.vertex_offset_sgpr = vertex_offset_candidate;
							}
							if (!instance_offset_conflict) {
								data.instance_offset_sgpr = instance_offset_candidate;
							}
						}
						auto& load        = data.loads.emplace_back();
						load.pc           = inst.pc;
						load.attrib_id    = buffer.attrib_id;
						load.components   = DecodedDstSize(inst);
					}
				}
				break;
		}
		if (inst.opcode == Decoder::Opcode::V_MOVRELD_B32) {
			vector_lanes.clear();
		} else if (inst.opcode != Decoder::Opcode::V_WRITELANE_B32 && IsDecodedVgpr(inst.dst)) {
			for (uint32_t i = 0;
			     i < EmbeddedFetchDstSize(inst) && inst.dst.reg + i < vgpr_is_index.size();
			     i++) {
				ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg + i);
			}
		}
	}

	return data;
}

Decoder::Program DecodeFusedProgram(std::span<const uint32_t> front, std::span<const uint32_t> back,
                                    std::vector<uint32_t>& joined_code) {
	EXIT_IF(back.empty());
	auto       result      = Decoder::DecodeFrontProgram(front);
	const auto front_words = static_cast<uint32_t>(result.code.size());
	joined_code.assign(result.code.begin(), result.code.end());
	joined_code.insert(joined_code.end(), back.begin(), back.end());
	joined_code[front_words - 1u] = 0xbf820000u;
	result.instructions.back()    = {};
	Decoder::DecodeInstruction(joined_code, front_words - 1u, result.instructions.back());
	Decoder::Program back_program;
	Decoder::DecodeProgram(back, back_program);
	result.has_swap_pc |= back_program.has_swap_pc;
	const auto back_pc = front_words * sizeof(uint32_t);
	for (auto& inst: back_program.instructions) {
		EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_GETPC_B64);
		inst.pc += back_pc;
		inst.branch_target += back_pc;
		result.instructions.push_back(std::move(inst));
	}
	result.code = joined_code;
	return result;
}

}

namespace {

std::mutex                                  g_ray_tracing_programs_mutex;
std::array<std::unordered_set<uint64_t>, static_cast<size_t>(RayTracingProgramKind::Count)> g_ray_tracing_programs;
RayTracingPrograms                          g_ray_tracing_counts;

}

bool NoteRayTracingProgram(RayTracingProgramKind kind, uint64_t shader_hash) {
	const std::lock_guard lock(g_ray_tracing_programs_mutex);
	if (!g_ray_tracing_programs[static_cast<size_t>(kind)].insert(shader_hash).second) {
		return false;
	}
	switch (kind) {
		case RayTracingProgramKind::Native: g_ray_tracing_counts.native++; break;
		case RayTracingProgramKind::Empty:
			g_ray_tracing_counts.empty++;
			g_ray_tracing_counts.last_empty = shader_hash;
			break;
		case RayTracingProgramKind::AlwaysGuest:
			g_ray_tracing_counts.always_guest++;
			g_ray_tracing_counts.last_always_guest = shader_hash;
			break;
		case RayTracingProgramKind::PixelGuest:
			g_ray_tracing_counts.pixel_guest++;
			g_ray_tracing_counts.last_pixel_guest = shader_hash;
			break;
		case RayTracingProgramKind::Count: break;
	}
	return true;
}

RayTracingPrograms GetRayTracingPrograms() {
	const std::lock_guard lock(g_ray_tracing_programs_mutex);
	return g_ray_tracing_counts;
}

ShaderSource PrepareShaderSource(std::span<const uint32_t> code, const CompileOptions& options) {
	EXIT_IF(code.empty());
	ShaderSource source;
#if defined(ANYPS5_KYTY)
	static constexpr std::array<uint32_t, 1> EndProgram {0xbf810000u};
	const bool merged_hull =
	    (options.stage == ShaderType::Local || options.stage == ShaderType::TessellationControl) &&
	    options.input_info.vertex != nullptr && options.input_info.vertex->tess.merged_hull != 0;
	if (merged_hull && options.stage == ShaderType::Local) {
		Decoder::DecodeProgram(EndProgram, source.decoded);
	} else if (merged_hull) {
		Decoder::DecodeProgram(code, source.decoded);
		if (source.decoded.has_swap_pc) {
			EXIT("a merged hull program that calls a subroutine is unsupported\n");
		}
	} else
#endif
	if (!options.back_code.empty()) {
		source.decoded = DecodeFusedProgram(code, options.back_code, source.code);
	} else if (options.stage == ShaderType::Local) {
		source.decoded = Decoder::DecodeFrontProgram(code);
		auto& handoff = source.decoded.instructions.back();
		handoff.opcode = Decoder::Opcode::S_ENDPGM;
		handoff.src_count = 0;
	} else {
		Decoder::DecodeProgram(code, source.decoded);
	}
	if (source.decoded.has_swap_pc) {
		PrepareCallTargets(source, options);
		source.code.assign(code.begin(), code.end());
		source.decoded.code = source.code;
	}
	if (options.stage == ShaderType::Compute) {
		(void)Decoder::DeriveRecordReach(source.decoded, options.wave_size, source.decoded.record_reach);
	}
	return source;
}

TranslateResult TranslateProgram(std::span<const uint32_t> code, const CompileOptions& options) {
	auto source = PrepareShaderSource(code, options);
	return TranslateProgram(source.decoded, options);
}

TranslateResult TranslateProgram(const Decoder::Program& decoded, const CompileOptions& options) {
	EXIT_IF(decoded.instructions.empty());
	if (decoded.has_swap_pc) {
		for (const auto& inst: decoded.instructions) {
			if (inst.opcode == Decoder::Opcode::S_SWAPPC_B64 && inst.branch_target == UINT32_MAX)
				EXIT("shader call at pc 0x%08x must be linked before translation", inst.pc);
		}
	}

	for (const auto& inst: decoded.instructions) {
		const auto mode_bits = Decoder::FloatModeBitsWritten(inst);
		if (mode_bits == 0u) continue;
		Common::HackNotice::Report("S_SETREG_B32 float MODE write ignored", "ShaderRecompiler::TranslateProgram",
		                           options.shader_hash, inst.pc,
		                           fmt::format("round=0x{:x} denorm=0x{:x}", mode_bits & 0xfu, (mode_bits >> 4u) & 0xfu));
	}

	const auto* stage        = StageName(options.stage);
	const auto compile_begin = std::chrono::steady_clock::now();
	const auto phase_ms = [&compile_begin]() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		    std::chrono::steady_clock::now() - compile_begin).count());
	};

	std::string decoded_dump;
	if (options.dump_ir) {
		decoded_dump = Decoder::ProgramToString(decoded);
		LOGF("ShaderRecompiler %s decoded RDNA2:\n%s", stage, decoded_dump.c_str());
	}

	LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " CFG BuildGraph\n", stage,
	     options.shader_hash);
	auto native_cfg = CFG::BuildGraph(decoded);
	CFG::Graph structured_cfg;
	auto* selected_cfg = &native_cfg;
	LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " CFG BuildGraph blocks=%" PRIu64
	     " loops=%" PRIu64 " back_edges=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
	     stage, options.shader_hash,
	     static_cast<uint64_t>(native_cfg.blocks.size()),
	     static_cast<uint64_t>(native_cfg.natural_loops.size()),
	     static_cast<uint64_t>(native_cfg.back_edges.size()), phase_ms());
	if (native_cfg.irreducible) {
		LogDispatcherFallback(options, native_cfg, "build");
	} else {
		LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " CFG Structurize\n", stage,
		     options.shader_hash);
		structured_cfg = CFG::Structurize(native_cfg);
		if (structured_cfg.unsupported) {
			native_cfg.unsupported = true;
			native_cfg.failure_kind = structured_cfg.failure_kind;
			native_cfg.failure_block = structured_cfg.failure_block;
			native_cfg.unsupported_reason = structured_cfg.unsupported_reason;
			LogDispatcherFallback(options, native_cfg, "structurize");
		} else {
			selected_cfg = &structured_cfg;
			LOGF("ShaderRecompiler %s structured CFG success: blocks=%" PRIu64 "\n", stage,
			     static_cast<uint64_t>(selected_cfg->blocks.size()));
		}
		LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " CFG Structurize blocks=%" PRIu64
		     " loops=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
		     stage, options.shader_hash,
		     static_cast<uint64_t>(selected_cfg->blocks.size()),
		     static_cast<uint64_t>(selected_cfg->natural_loops.size()), phase_ms());
	}

	const auto& cfg = *selected_cfg;
	Frontend::EmbeddedFetchPlan embedded_fetch;
	if ((options.stage == ShaderType::Vertex || options.stage == ShaderType::Local) &&
	    options.input_info.vertex->has_dynamic_vertex_fetch) {
		embedded_fetch = DetectEmbeddedVertexFetch(
		    decoded, *options.input_info.vertex, options.user_data_base,
		    static_cast<uint32_t>(options.user_data.size()), options.wave_size);
		if (!embedded_fetch.loads.empty()) {
			LOGF("ShaderRecompiler %s embedded vertex fetch plan: detected=%" PRIu64 "\n", stage,
			     static_cast<uint64_t>(embedded_fetch.loads.size()));
		}
	}
	Frontend::TranslateOptions translate_options {
	    .stage            = options.stage,
	    .wave_size        = options.wave_size,
	    .shader_hash      = options.shader_hash,
	    .user_data_base   = options.user_data_base,
	    .user_data_count  = static_cast<uint32_t>(options.user_data.size()),
	    .input_info       = options.input_info,
	    .embedded_fetch   = embedded_fetch.loads.empty() ? nullptr : &embedded_fetch,
	};
	LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " IR TranslateProgram\n", stage,
	     options.shader_hash);
	auto ir = Frontend::TranslateProgram(decoded, cfg, translate_options);
	for (const auto& body: decoded.call_bodies)
		ir.inlined_call_entries.push_back(body.begin);
	LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " IR TranslateProgram blocks=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     stage, options.shader_hash,
	     static_cast<uint64_t>(ir.blocks.size()), phase_ms());
	IR::MarkUndefinedLaneSpills(ir);
	IR::RewriteToSsa(ir.blocks);
	const auto simplify = [&ir] {
		IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
		IR::RemoveIdentities(ir);
		IR::EliminateDeadCode(ir.blocks);
	};
	simplify();
	const auto read_lane_stats = IR::EliminateReadLane(ir, ir.wave_size);
	IR::ReleaseUndefinedLaneSpills(ir);
	if (read_lane_stats.rewritten_reads != 0 || read_lane_stats.rewritten_reductions != 0) {
		LOGF("ShaderRecompiler %s read-lane elimination: reads=%" PRIu32
		     " reductions=%" PRIu32 "\n",
		     stage, read_lane_stats.rewritten_reads, read_lane_stats.rewritten_reductions);
	}
	simplify();
	if (decoded.has_swap_pc) {
		for (const auto* block: ir.blocks)
			for (const auto& inst: *block)
				EXIT_NOT_IMPLEMENTED(inst.GetOpcode() == IR::ValueOpcode::GetShaderBase);
	}
	if (!LowerTessellationMemory(ir, options)) {
		EXIT("ShaderRecompiler %s failed hash=0x%016" PRIx64
		     ": unsupported tessellation memory shape\n",
		     stage, options.shader_hash);
	}
	std::string cfg_dump;
	if (options.dump_ir) {
		cfg_dump = CFG::GraphToString(cfg);
		LOGF("ShaderRecompiler %s native IR before resource tracking:\n%s", stage,
		     MakeIrDump(cfg_dump, ir).c_str());
	}
	const auto*      tracked = &decoded;
	Decoder::Program traced_source;
	std::vector<IR::NativeTraversals>  oracle_sources;
	std::vector<IR::OracleImprovement> oracle_improvements;
	const bool bvh = std::ranges::any_of(decoded.instructions, [](const auto& inst) {
		return inst.opcode == Decoder::Opcode::IMAGE_BVH_INTERSECT_RAY;
	});
	constexpr uint32_t pixel_oracle = IR::OracleRemaining | IR::OraclePixel;
	if (options.stage == ShaderType::Pixel && options.native_ray_tracing && bvh &&
	    (options.rt_oracle & pixel_oracle) != pixel_oracle &&
	    NoteRayTracingProgram(RayTracingProgramKind::PixelGuest, options.shader_hash)) {
		::Log::WriteToConsoleAndLog(fmt::format("[RT] ps 0x{:016x}: pixel walk kept on the guest traversal: pixel oracle "
		                                        "off (KYTY_RT_ORACLE bit 8)\n",
		                                        options.shader_hash));
	}
	if (options.stage == ShaderType::Compute ||
	    (options.stage == ShaderType::Pixel && options.native_ray_tracing && options.rt_oracle != 0)) {
		const auto traced = IR::TraceNativeTraversals(
		    ir, {.wave_size = options.wave_size,
		         .rewrite   = options.native_ray_tracing,
		         .oracle    = options.native_ray_tracing ? options.rt_oracle : 0u,
		         .decoded   = &decoded});
		oracle_sources      = traced.oracle_sources;
		oracle_improvements = traced.oracle_improvements;
		if (traced.source_valid) {
			ir.info.ray_query_user_sgpr      = traced.user_sgpr;
			ir.info.ray_query_header_offset  = traced.header_offset;
			ir.info.ray_query_header_vsharp  = traced.header_vsharp;
			ir.info.ray_query_root           = traced.root;
			ir.info.ray_query_layout         = traced.layout;
			traced_source                    = decoded;
			traced_source.ray_tracing_source = {.user_sgpr     = traced.user_sgpr,
			                                    .header_offset = traced.header_offset,
			                                    .header_vsharp = traced.header_vsharp,
			                                    .root          = traced.root,
			                                    .layout        = traced.layout,
			                                    .valid         = true};
			if (traced.sites != 0 && !traced.reach.empty()) {
				auto& reach = traced_source.ray_tracing_reach;
				for (const auto& inst: decoded.instructions) {
					if (std::ranges::any_of(traced.reach, [&](const auto& range) {
						    return inst.pc >= range.first && inst.pc < range.second;
					    })) {
						reach.push_back(inst.pc);
					}
				}
				std::ranges::sort(reach);
				reach.erase(std::unique(reach.begin(), reach.end()), reach.end());
			}
			tracked = &traced_source;
		}
		if (traced.sites != 0) {
			LOGF("%s native ray tracing traversals: hash=0x%016" PRIx64 " sites=%u\n", stage,
			     options.shader_hash, traced.sites);
			NoteRayTracingProgram(RayTracingProgramKind::Native, options.shader_hash);
		}
		size_t intersections = 0, empty = 0;
		for (const auto* block: ir.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != IR::ValueOpcode::BvhIntersect) continue;
				++intersections;
				empty += (inst.Flags<uint32_t>() & IR::BvhIntersectEmpty) != 0 ? 1u : 0u;
			}
		}
		if (empty != 0 && NoteRayTracingProgram(RayTracingProgramKind::Empty, options.shader_hash)) {
			::Log::WriteToConsoleAndLog(fmt::format("[RT] {} 0x{:016x}: {} of {} BVH intersections run empty: no form "
			                                        "and no host query serves them, their rays miss\n",
			                                        options.stage == ShaderType::Compute ? "cs" : "ps",
			                                        options.shader_hash, empty, intersections));
		}
	}
	IR::TrackResources(ir, *tracked, native_cfg, options.input_info);
	TranslateResult result;
	result.program        = std::move(ir);
	result.oracle_sources      = std::move(oracle_sources);
	result.oracle_improvements = std::move(oracle_improvements);
	if (options.dump_ir) {
		result.decoded_dump = std::move(decoded_dump);
		result.cfg_dump     = std::move(cfg_dump);
	}
	return result;
}

CompileResult CompileProgram(TranslateResult translated, const CompileOptions& options,
                             const IR::ResourceSpecialization& specialization,
                             uint32_t push_data_start_dword) {
	const auto emit_begin = std::chrono::steady_clock::now();
	auto& ir = translated.program;
	const auto* stage = StageName(ir.stage);
	IR::ApplyResourceSpecialization(ir, specialization);
	if (options.table_mode) {
		CompileResult refused;
		if (!IR::EnterTableMode(ir, refused.table_refusal)) {
			return refused;
		}
	}
	for (auto& inst: ir.value_storage) {
		inst.Invalidate();
	}
	for (auto* block: ir.blocks) {
		for (auto& inst: *block) {
			const auto op = inst.GetOpcode();
			uint32_t first = 0;
			if (op == IR::ValueOpcode::GetBufferResource) {
				if (std::ranges::any_of(inst.Uses(), [&](const IR::Use& use) {
					return ir.memory_info[use.user->Flags<IR::MemoryFlags>().index].kind ==
					       IR::ResourceKind::IndirectBuffer;
				})) {
					continue;
				}
				const auto resource = inst.Flags<uint32_t>();
				first               = resource < ir.info.buffers.size() &&
				                              ir.info.buffers[resource].indirect_root == resource
				                          ? 1u
				                          : 0u;
			} else if (op == IR::ValueOpcode::GetImageResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.images.size() &&
				                (ir.info.images[resource].indirect_root == resource ||
				                 ir.info.images[resource].table_capacity != 0u) ? 1u : 0u;
			} else if (op == IR::ValueOpcode::GetSamplerResource) {
				const auto resource = inst.Flags<uint32_t>();
				first = resource < ir.info.samplers.size() &&
				                !ir.info.samplers[resource].indirect_resources.empty() ? 1u : 0u;
			} else {
				continue;
			}
			for (size_t index = first; index < inst.NumArgs(); index++) {
				inst.SetArg(index, IR::Value(0u));
			}
		}
	}
	ir.value_storage.clear();
	IR::RemoveIdentities(ir);
	IR::EliminateDeadCode(ir.blocks);
	if (options.profile->ray_query) {
		if (const auto traced = IR::TraceTraversalsWithRayQueries(ir); traced != 0) {
			LOGF("ShaderRecompiler %s ray query traversals: hash=0x%016" PRIx64 " loops=%u\n",
			     stage, ir.shader_hash, traced);
		}
	}
	if (const auto selects = IR::EliminateMaskedSelects(ir.blocks); selects != 0) {
		LOGF("ShaderRecompiler %s masked select elimination: selects=%" PRIu32 "\n", stage, selects);
	}
	IR::RemoveIdentities(ir);
	IR::EliminateDeadCode(ir.blocks);
	if (IR::LowerRelativeVgprs(ir) != 0) {
		IR::RemoveIdentities(ir);
		IR::EliminateDeadCode(ir.blocks);
	}

	IR::CollectShaderInfo(ir, options.input_info);
	std::string ir_dump;
	if (options.dump_ir) {
		ir_dump = MakeIrDump(translated.cfg_dump, ir);
		LOGF("ShaderRecompiler %s native IR:\n%s", stage, ir_dump.c_str());
	}

	LOGF("ShaderRecompiler %s phase begin: hash=0x%016" PRIx64 " SPIR-V EmitProgram\n", stage,
	     ir.shader_hash);
	std::vector<uint32_t> spirv;
	if (const auto* group = options.stage == ShaderType::Compute ? options.input_info.compute : nullptr;
	    group != nullptr && !options.wave_lane_pairs && ir.wave_size == 64u && group->host_subgroup_size == 32u &&
	    !ir.lane_local && !(ir.info.uses_lds && (group->lds_storage || ir.info.function_lds)) &&
	    Spirv::WaveSplitThreads(group->threads_num)) {
		auto wide               = *group;
		wide.host_subgroup_size = 64u;
		auto wide_input         = options.input_info;
		wide_input.compute      = &wide;
		Spirv::SplitHints hints;
		spirv = Spirv::EmitProgram(ir, wide_input, push_data_start_dword, *options.profile, options.emit_debug_info,
		                           true, options.ray_stats, &hints);
		std::string refusal;
		if (!Spirv::SplitWaveAcrossSubgroups(spirv, refusal, &hints)) {
			LOGF("ShaderRecompiler %s wave64 kept in lane pairs: hash=0x%016" PRIx64 " reason=%s\n", stage,
			     ir.shader_hash, refusal.c_str());
			spirv.clear();
		} else {
			LOGF("ShaderRecompiler %s wave64 split across subgroups: hash=0x%016" PRIx64 " threads=%u\n", stage,
			     ir.shader_hash,
			     std::max(group->threads_num[0], 1u) * std::max(group->threads_num[1], 1u) *
			         std::max(group->threads_num[2], 1u));
		}
	}
	if (spirv.empty()) {
		spirv = Spirv::EmitProgram(ir, options.input_info, push_data_start_dword, *options.profile,
		                           options.emit_debug_info, true, options.ray_stats);
	}
	LOGF("ShaderRecompiler %s phase end: hash=0x%016" PRIx64 " SPIR-V EmitProgram words=%" PRIu64
	     " wave_size=%u elapsed_ms=%" PRIu64 "\n",
	     stage, ir.shader_hash, static_cast<uint64_t>(spirv.size()), ir.wave_size,
	     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - emit_begin)
	                               .count()));
	CompileResult result;
	if (options.plain_mip_stats_variant && Spirv::UsesMipStats(ir)) {
		result.spirv_plain =
		    Spirv::EmitProgram(ir, options.input_info, push_data_start_dword, *options.profile,
		                       options.emit_debug_info, false, options.ray_stats);
	}
	result.spirv          = std::move(spirv);
	result.program        = std::move(ir);
	result.oracle_sources      = std::move(translated.oracle_sources);
	result.oracle_improvements = std::move(translated.oracle_improvements);
	if (options.dump_ir) {
		result.decoded_dump = std::move(translated.decoded_dump);
		result.ir_dump      = std::move(ir_dump);
	}
	return result;
}

}
