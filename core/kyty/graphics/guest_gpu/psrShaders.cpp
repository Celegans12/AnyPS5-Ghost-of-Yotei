#include "graphics/guest_gpu/psrShaders.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <iterator>
#include <string_view>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics::Psr {

namespace {

struct NamedShader {
	std::string_view name;
	Shader           shader;
};

constexpr NamedShader Names[] = {
    {"psr_morton_compute_bounding_box_for_instances", Shader::MortonComputeBoundingBoxForInstances},
    {"psr_morton_initialize_indirect_args_top", Shader::MortonInitializeIndirectArgsTop},
    {"psr_single_pass_scan_initialize_descriptors_uint32", Shader::SinglePassScanInitializeDescriptors},
    {"psr_single_pass_scan_initialize_descriptors_float", Shader::SinglePassScanInitializeDescriptors},
    {"psr_single_pass_scan_uint32", Shader::SinglePassScanUint32},
    {"psr_morton_compute_valid_instance_indices", Shader::MortonComputeValidInstanceIndices},
    {"psr_morton_setup_indirect_args_top", Shader::MortonSetupIndirectArgsTop},
    {"psr_morton_init_one_instance_root_node", Shader::MortonInitOneInstanceRootNode},
    {"psr_morton_compute_code_for_instances", Shader::MortonComputeCodeForInstances},
    {"psr_bitonic_sort_pass_key_value_indirect", Shader::BitonicSortPassKeyValueIndirect},
    {"psr_morton_build_qbvh_hierarchy_for_instances", Shader::MortonBuildQbvhHierarchyForInstances},
    {"psr_morton_build_instance_node", Shader::MortonBuildInstanceNode},
    {"psr_morton_refit_qbvh_hierarchy_for_instances", Shader::MortonRefitQbvhHierarchyForInstances},
    {"psr_morton_rotate_tree_first_run_top", Shader::MortonRotateTreeFirstRunTop},
    {"psr_morton_rotate_tree_top", Shader::MortonRotateTreeTop},
    {"psr_collapse_to_qbvh_simple_compute_compacted_nodes_indirect",
     Shader::CollapseToQbvhSimpleComputeCompactedNodesIndirect},
    {"psr_single_pass_scan_setup_indirect_args_vec2u", Shader::SinglePassScanSetupIndirectArgsVec2u},
    {"psr_single_pass_scan_initialize_descriptors_vec2u_indirect",
     Shader::SinglePassScanInitializeDescriptorsVec2uIndirect},
    {"psr_single_pass_scan_vec2u_indirect", Shader::SinglePassScanVec2uIndirect},
    {"psr_collapse_to_qbvh_simple_top", Shader::CollapseToQbvhSimpleTop},
    {"psr_qbvh_fan_refit_copy_dispatch_args", Shader::QbvhFanRefitCopyDispatchArgs},
    {"psr_qbvh_fan_refit_async", Shader::QbvhFanRefitAsync},
    {"psr_qbvh_fan_refit_recover_aabbs", Shader::QbvhFanRefitRecoverAabbs},
    {"psr_qbvh_refit_bottom_level", Shader::QbvhRefitBottomLevel},
    {"psr_morton_compute_bounding_box_for_vertices", Shader::MortonComputeBoundingBoxForVertices},
    {"psr_morton_compute_code_for_mesh_uint8x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_uint16x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_uint32x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_low_precision_uint8x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_low_precision_uint16x3", Shader::MortonComputeCodeForMesh},
    {"psr_morton_compute_code_for_mesh_low_precision_uint32x3", Shader::MortonComputeCodeForMesh},
    {"psr_collapse_to_qbvh_simple_bottom", Shader::CollapseToQbvhSimpleBottom},
    {"psr_collapse_to_qbvh_simple_bottom_with_refit", Shader::CollapseToQbvhSimpleBottom},
    {"psr_morton_capture_memory", Shader::MortonCaptureMemory},
    {"psr_decompress_header_base", Shader::DecompressHeaderBase},
    {"psr_decompress_compute_fan_node_uint32x3", Shader::DecompressComputeFanNode},
    {"psr_decompress_compute_fan_node_uint16x3", Shader::DecompressComputeFanNode},
    {"psr_decompress_compute_fan_node_uint8x3", Shader::DecompressComputeFanNode},
    {"psr_decompress_compute_fan_node_shader_mesh_0", Shader::DecompressComputeFanNode},
    {"psr_decompress_compute_fan_node_shader_mesh_1", Shader::DecompressComputeFanNode},
    {"psr_decompress_compute_fan_node_shader_mesh_2", Shader::DecompressComputeFanNode},
    {"psr_decompress_compute_fan_node_shader_mesh_3", Shader::DecompressComputeFanNode},
};

std::atomic<Generation> g_generation {Generation::Unknown};
std::atomic<bool>       g_generation_known {false};
std::atomic<bool>       g_replacement_stopped {false};
std::atomic<bool>       g_unknown_build_logged {false};

struct CodeEntry {
	uint64_t fingerprint;
	Shader   shader;
};
#include "graphics/guest_gpu/psrShaderCodes.inc"

struct GenerationCodes {
	Generation                 generation;
	std::span<const CodeEntry> codes;
	std::span<const uint32_t>  revisions;
};
constexpr std::array KnownCodes {
    GenerationCodes {Generation::Base460, Base460Codes, Base460Revisions},
    GenerationCodes {Generation::Base252, Base252Codes, Base252Revisions},
#if defined(ANYPS5_KYTY)
    GenerationCodes {Generation::Base148, Base148Codes, Base148Revisions},
#endif
};

const char* GenerationName(Generation generation) {
	switch (generation) {
		case Generation::Base460: return "Base460";
		case Generation::Base252: return "Base252";
		case Generation::Base515: return "Base515";
#if defined(ANYPS5_KYTY)
		case Generation::Base148: return "Base148";
#endif
		default: return "unknown";
	}
}

Shader FindCode(const GenerationCodes& generation, uint64_t fingerprint) {
	const auto it = std::ranges::lower_bound(generation.codes, fingerprint, {}, &CodeEntry::fingerprint);
	return it != generation.codes.end() && it->fingerprint == fingerprint ? it->shader : Shader::None;
}

uint32_t CompilerRevision(std::span<const uint32_t> binary) {
	constexpr size_t FooterWords = 12;
	if (binary.size() < FooterWords) {
		return 0;
	}
	const auto footer = binary.last(FooterWords);
	return std::memcmp(footer.data(), "barefoot", 8) == 0 ? footer[9] : 0;
}

const char* RegisteredName(uint64_t code_address) {
	return ShaderRegisteredName(code_address);
}

bool LibraryNamed() {
	return ShaderNameRegistered("psr_morton_capture_memory");
}

constexpr std::array NewerOpcodes {ShaderRecompiler::Decoder::Opcode::V_PERM_B32,
                                   ShaderRecompiler::Decoder::Opcode::V_SAD_U8,
                                   ShaderRecompiler::Decoder::Opcode::V_SAD_HI_U8,
                                   ShaderRecompiler::Decoder::Opcode::V_SAD_U16,
                                   ShaderRecompiler::Decoder::Opcode::V_MSAD_U8,
                                   ShaderRecompiler::Decoder::Opcode::V_LERP_U8,
                                   ShaderRecompiler::Decoder::Opcode::V_QSAD_PK_U16_U8,
                                   ShaderRecompiler::Decoder::Opcode::V_MQSAD_PK_U16_U8,
                                   ShaderRecompiler::Decoder::Opcode::V_MQSAD_U32_U8,
                                   ShaderRecompiler::Decoder::Opcode::V_CMP_U_F16};

int64_t TableOpcode(ShaderRecompiler::Decoder::Opcode opcode) {
	int64_t earlier = 0;
	for (const auto newer: NewerOpcodes) {
		if (opcode == newer) {
			return -1;
		}
		earlier += opcode > newer ? 1 : 0;
	}
	return static_cast<int64_t>(opcode) - earlier;
}

uint64_t InstructionKey(const ShaderRecompiler::Decoder::Instruction& inst, bool pc_relative) {
	using ShaderRecompiler::Decoder::Family;
	using ShaderRecompiler::Decoder::Opcode;
	using ShaderRecompiler::Decoder::Operand;
	using ShaderRecompiler::Decoder::OperandKind;
	const auto operand_key = [&](const Operand& operand) {
		auto kind = operand.kind;
		if (kind == OperandKind::VccLo || kind == OperandKind::VccHi) {
			kind = OperandKind::Sgpr;
		}
		uint64_t   key      = static_cast<uint32_t>(kind);
		const bool constant = kind == OperandKind::IntegerInlineConstant ||
		                      kind == OperandKind::FloatInlineConstant ||
		                      (kind == OperandKind::LiteralConstant && !pc_relative);
		if (constant) {
			key |= uint64_t {operand.value} << 32u;
		}
		const uint32_t modifiers = (operand.negate ? 1u : 0u) | (operand.absolute ? 2u : 0u) |
		                           (operand.clamp ? 4u : 0u) | (operand.op_sel ? 8u : 0u) |
		                           (operand.op_sel_hi ? 16u : 0u) | (operand.negate_hi ? 32u : 0u) |
		                           ((operand.omod & 3u) << 6u) | ((operand.sdwa_sel & 15u) << 8u) |
		                           ((operand.dpp ? (operand.dpp_ctrl & 0x1ffu) + 1u : 0u) << 12u);
		return key ^ (uint64_t {modifiers} << 8u);
	};
	std::array<uint64_t, 15> fields {};
	const auto table_opcode = inst.opcode != Opcode::UNKNOWN && inst.opcode != Opcode::UNSUPPORTED
	                              ? TableOpcode(inst.opcode)
	                              : -1;
	const bool known        = table_opcode >= 0;
	fields[0]               = known ? static_cast<uint64_t>(table_opcode)
	                                : (0x100000000ull | (static_cast<uint64_t>(inst.family) << 16u) | inst.opcode_id);
	if (inst.family == Family::SOPP || !known) {
		return XXH3_64bits(fields.data(), sizeof(uint64_t));
	}
	fields[1] = operand_key(inst.dst);
	fields[2] = operand_key(inst.dst2);
	std::array<uint64_t, 3> sources {operand_key(inst.src0), operand_key(inst.src1), operand_key(inst.src2)};
	std::ranges::sort(sources);
	fields[3]  = sources[0];
	fields[4]  = sources[1];
	fields[5]  = sources[2];
	fields[6]  = operand_key(inst.src3);
	fields[7]  = (uint64_t {inst.offset} << 32u) | inst.secondary_offset;
	fields[8]  = (uint64_t {inst.dmask} << 32u) | inst.data_format;
	fields[9]  = (uint64_t {inst.number_format} << 32u) | inst.image_sample_flags;
	fields[10] = (uint64_t {inst.data_dwords} << 32u) | inst.data_components;
	fields[11] = (uint64_t {inst.data_bits} << 32u) | static_cast<uint32_t>(inst.image_dimension);
	fields[12] = (uint64_t {inst.memory_segment} << 32u) | inst.image_address_components;
	fields[13] = (inst.data_signed ? 1u : 0u) | (inst.typed ? 2u : 0u) | (inst.formatted ? 4u : 0u) |
	             (inst.gds ? 8u : 0u) | (inst.glc ? 16u : 0u) | (inst.dlc ? 32u : 0u) |
	             (inst.slc ? 64u : 0u) | (inst.idxen ? 128u : 0u) | (inst.offen ? 256u : 0u) |
	             (inst.image_r128 ? 512u : 0u);
	fields[14] = inst.src_count;
	return XXH3_64bits(fields.data(), sizeof(fields));
}

Generation DetectGeneration() {
	const bool older = ShaderNameRegistered("psr_morton_compute_bounding_box_for_vertices") &&
	                   ShaderNameRegistered("psr_rebraiding_copy_srt_instance");
	const bool newer = ShaderNameRegistered("psr_count_nested_bvh") &&
	                   ShaderNameRegistered("psr_create_flattened_header");
	if (older && !newer) {
		return Generation::Base460;
	}
	return newer && !older ? Generation::Base515 : Generation::Unknown;
}

constexpr uint32_t Base515Revision = 0x5371c534u;
struct Base515Code {
	std::string_view name;
	uint64_t         fingerprint;
	Shader           shader;
};
constexpr Base515Code Base515Codes[] = {
    {"psr_qbvh_fan_refit_copy_dispatch_args", 0xa2d5a06017eb9c03ull, Shader::QbvhFanRefitCopyDispatchArgs},
    {"psr_qbvh_fan_refit_async_shader_mesh_0", 0xc416e109a7bbc897ull, Shader::QbvhFanRefitAsyncShaderMesh},
    {"psr_qbvh_fan_refit_async_shader_mesh_1", 0xc416e109a7bbc897ull, Shader::QbvhFanRefitAsyncShaderMesh},
    {"psr_qbvh_fan_refit_async_shader_mesh_2", 0xc416e109a7bbc897ull, Shader::QbvhFanRefitAsyncShaderMesh},
    {"psr_qbvh_fan_refit_async_shader_mesh_3", 0xc416e109a7bbc897ull, Shader::QbvhFanRefitAsyncShaderMesh},
    {"psr_qbvh_fan_refit_recover_aabbs", 0xba9376f980d9e6c6ull, Shader::QbvhFanRefitRecoverAabbs},
    {"psr_qbvh_refit_bottom_level", 0x688fc60a8000d2edull, Shader::QbvhRefitBottomLevel},
};

Shader IdentifyBase515(uint64_t code_address, std::string_view name) {
	for (const auto& code: Base515Codes) {
		if (code.name != name) {
			continue;
		}
		const auto binary = ShaderMappedBinary(code_address);
		if (CompilerRevision(binary) == Base515Revision && CodeFingerprint(binary) == code.fingerprint) {
			return code.shader;
		}
		LOGF("Psr: %s is not the code the emulator knows: it is emulated\n", code.name.data());
		return Shader::Other;
	}
	return Shader::Other;
}

}

Generation LibraryGeneration() {
	return g_generation.load(std::memory_order_acquire);
}

uint64_t CodeFingerprint(std::span<const uint32_t> binary) {
	namespace D = ShaderRecompiler::Decoder;
	std::vector<uint32_t> code(binary.begin(), binary.end());
	code.resize(code.size() + D::MaxInstructionRawWords, 0);
	std::vector<uint64_t> keys;
	uint32_t              getpc_sgpr = UINT32_MAX;
	for (uint32_t word = 0; word < binary.size();) {
		if (D::GetInstructionFamily(code[word]) == D::Family::Unknown) {
			return 0;
		}
		D::Instruction inst;
		D::DecodeInstruction(code, word, inst);
		word += std::max(inst.word_count, 1u);
		switch (inst.opcode) {
			case D::Opcode::S_NOP:
			case D::Opcode::S_WAITCNT:
			case D::Opcode::S_WAITCNT_VSCNT:
			case D::Opcode::S_WAITCNT_DEPCTR:
			case D::Opcode::S_INST_PREFETCH:
			case D::Opcode::S_CLAUSE: continue;
			case D::Opcode::S_SETPC_B64: return 0;
			default: break;
		}
		const auto reads = [&](uint32_t sgpr) {
			return (inst.src0.kind == D::OperandKind::Sgpr && inst.src0.reg == sgpr) ||
			       (inst.src1.kind == D::OperandKind::Sgpr && inst.src1.reg == sgpr);
		};
		const bool pc_relative = getpc_sgpr != UINT32_MAX &&
		                         ((inst.opcode == D::Opcode::S_ADD_U32 && reads(getpc_sgpr)) ||
		                          (inst.opcode == D::Opcode::S_ADDC_U32 && reads(getpc_sgpr + 1)));
		if (inst.opcode == D::Opcode::S_GETPC_B64 && inst.dst.kind == D::OperandKind::Sgpr) {
			getpc_sgpr = inst.dst.reg;
		}
		keys.push_back(InstructionKey(inst, pc_relative));
		if (inst.opcode == D::Opcode::S_ENDPGM) {
			std::ranges::sort(keys);
			const auto fingerprint = XXH3_64bits(keys.data(), keys.size() * sizeof(uint64_t));
			return fingerprint != 0 ? fingerprint : 1;
		}
	}
	return 0;
}

static bool AcceptedRevision(uint32_t revision) {
	return std::ranges::any_of(KnownCodes, [&](const GenerationCodes& generation) {
		return std::ranges::find(generation.revisions, revision) != generation.revisions.end();
	});
}

static CodeMatch IdentifyCode(uint32_t revision, uint64_t fingerprint) {
	for (const auto& generation: KnownCodes) {
		if (std::ranges::find(generation.revisions, revision) != generation.revisions.end()) {
			const auto shader = FindCode(generation, fingerprint);
			return shader != Shader::None ? CodeMatch {generation.generation, shader} : CodeMatch {};
		}
	}
	return {};
}

CodeMatch IdentifyCode(std::span<const uint32_t> binary) {
	return IdentifyCode(CompilerRevision(binary), CodeFingerprint(binary));
}

static Shader IdentifyShaderCode(uint64_t code_address) {
	const auto binary = ShaderMappedBinary(code_address);
	if (binary.empty()) {
		return Shader::None;
	}
	const auto revision = CompilerRevision(binary);
	if (!AcceptedRevision(revision)) {
		if (std::ranges::find(DisabledRevisions, revision) != std::end(DisabledRevisions) &&
		    !g_unknown_build_logged.load(std::memory_order_acquire)) {
			const auto fingerprint = CodeFingerprint(binary);
			const bool library     = std::ranges::any_of(KnownCodes, [&](const GenerationCodes& generation) {
				return FindCode(generation, fingerprint) != Shader::None;
			});
			if (library && !g_unknown_build_logged.exchange(true, std::memory_order_acq_rel)) {
				LOGF("Psr: a build of the library whose replacements are not enabled (compiler revision "
				     "%08x): its shaders are emulated\n",
				     revision);
			}
		}
		return Shader::None;
	}
	const auto match = IdentifyCode(revision, CodeFingerprint(binary));
	if (match.generation == Generation::Unknown) {
		return Shader::None;
	}
	auto known = Generation::Unknown;
	if (g_generation.compare_exchange_strong(known, match.generation, std::memory_order_acq_rel)) {
		g_generation_known.store(true, std::memory_order_release);
#if defined(ANYPS5_KYTY)
		if (match.generation == Generation::Base148) {
			LOGF("Psr: library generation %s (shaders recognized by their code): its BLAS refits are "
			     "replaced, its BLAS builds are emulated and followed, its top level builds are emulated\n",
			     GenerationName(match.generation));
		} else
#endif
		LOGF("Psr: library generation %s (shaders recognized by their code)%s\n",
		     GenerationName(match.generation),
		     match.generation == Generation::Base252
		         ? ": its BLAS refits and top level builds are replaced, its BLAS builds are emulated "
		           "and followed"
		         : "");
	} else if (known != match.generation) {
		return Shader::None;
	}
	return match.shader;
}

Shader IdentifyShader(uint64_t code_address) {
	const char* name = RegisteredName(code_address);
	if (name == nullptr || std::strncmp(name, "psr_", 4) != 0) {
		return LibraryNamed() ? Shader::None : IdentifyShaderCode(code_address);
	}
	if (!g_generation_known.load(std::memory_order_acquire)) {
		const auto generation = DetectGeneration();
		g_generation.store(generation, std::memory_order_release);
		g_generation_known.store(true, std::memory_order_release);
		LOGF("Psr: library generation %s\n",
		     generation == Generation::Base460   ? "Base460"
		     : generation == Generation::Base515 ? "Base515 (its BLAS refits are replaced)"
		                                         : "unknown (its shaders are emulated)");
	}
	const auto generation = g_generation.load(std::memory_order_acquire);
	if (generation == Generation::Base515) {
		return IdentifyBase515(code_address, name);
	}
	if (generation != Generation::Base460) {
		return Shader::None;
	}
	for (const auto& entry: Names) {
		if (entry.name == name) {
			return entry.shader;
		}
	}
	return std::string_view(name).find("trinity") != std::string_view::npos ? Shader::Trinity : Shader::Other;
}

bool ReplacementEnabled() {
	return !g_replacement_stopped.load(std::memory_order_acquire);
}

void StopReplacement(const char* reason, const char* shader_name) {
	if (!g_replacement_stopped.exchange(true, std::memory_order_acq_rel)) {
		LOGF("Psr: %s (%s): its BVH builds and refits are emulated from now on\n", reason,
		     shader_name != nullptr ? shader_name : "?");
	}
}

bool IsTopLevelBuildShader(Shader shader) {
	return shader >= Shader::MortonComputeBoundingBoxForInstances &&
	       shader <= Shader::CollapseToQbvhSimpleTop;
}

}
