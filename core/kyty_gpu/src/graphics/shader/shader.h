#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

namespace ShaderError {

[[nodiscard]] inline bool Fail(std::string* error, std::string_view message) {
	if (error != nullptr) {
		error->assign(message.data(), message.size());
	}
	return false;
}

} // namespace ShaderError

namespace HW {
struct VertexShaderInfo;
struct PixelShaderInfo;
struct ComputeShaderInfo;
struct ShaderRegisters;
} // namespace HW

enum class ShaderType {
	Unknown,
	Vertex,
	Pixel,
	Fetch,
	Compute,
	Mesh,
	Local,
	TessellationControl,
	TessellationEvaluation
};

namespace ShaderRecompiler::IR {
struct CompiledShaderInfo;
} // namespace ShaderRecompiler::IR

namespace StageF {
struct ResolveRecord;
} // namespace StageF

struct ShaderStageRuntime {
	const ShaderRecompiler::IR::CompiledShaderInfo* program = nullptr;
	const ShaderRecompiler::IR::ResourceSnapshot*   resources = nullptr;
	// The resolve thread's lookups (stageFLookup.h): what the resolve thread already found for `resources` when ProgramCache::Get
	// adopted the resolve thread's snapshot for this draw (null otherwise). Only FindBuffers of the same draw reads it.
	const StageF::ResolveRecord*                    stagef_record = nullptr;

	[[nodiscard]] explicit operator bool() const {
		return program != nullptr && resources != nullptr;
	}
};

constexpr uint32_t DstSel(uint32_t x, uint32_t y = 0, uint32_t z = 0, uint32_t w = 0) {
	return x | (y << 3u) | (z << 6u) | (w << 9u);
}

inline uint8_t GetDstSel(uint32_t swizzle, uint32_t channel) {
	return (swizzle >> (channel * 3u)) & 0x7u;
}

[[nodiscard]] inline bool IsValidImageSwizzle(uint32_t swizzle,
                                             uint32_t component_count = 4) noexcept {
	if ((swizzle & ~0xfffu) != 0) {
		return false;
	}
	for (uint32_t channel = 0; channel < 4; channel++) {
		const auto selector = GetDstSel(swizzle, channel);
		if (selector > 1 && (selector < 4 || selector >= 4 + component_count)) return false;
	}
	return true;
}

struct ShaderClipSpaceTransform {
	float scale[2]       = {};
	float offset[2]      = {};
	float half_extent[2] = {};
	bool  enabled        = false;
};

struct ShaderWorkgroupInputInfo {
	uint32_t threads_num[3]      = {0, 0, 0};
	uint32_t lds_size_dwords     = 0;
	uint32_t scratch_size_dwords = 0;
	uint32_t host_subgroup_size  = 64;
	uint32_t wave_size           = 64;
};

struct ShaderMeshInputInfo: ShaderWorkgroupInputInfo {
	uint32_t input_primitive      = 0;
	uint32_t primitives_per_group = 0;
	uint32_t vertices_per_group   = 0;
	uint32_t max_vertices         = 0;
	uint32_t max_primitives       = 0;
	uint32_t provoking_vertex     = 0;
	bool     fast_launch          = false;

	[[nodiscard]] constexpr uint32_t InputPrimitiveSize() const {
		switch (static_cast<Prospero::PrimitiveType>(input_primitive)) {
			case Prospero::PrimitiveType::kPointList: return 1u;
			case Prospero::PrimitiveType::kLineList: return 2u;
			default: return 3u;
		}
	}
	[[nodiscard]] constexpr uint32_t InputPrimitiveStep() const {
		switch (static_cast<Prospero::PrimitiveType>(input_primitive)) {
			case Prospero::PrimitiveType::kTriFan:
			case Prospero::PrimitiveType::kTriStrip: return 1u;
			default: return InputPrimitiveSize();
		}
	}
	[[nodiscard]] constexpr uint32_t InputPrimitiveCount(uint32_t vertices) const {
		const auto size = InputPrimitiveSize();
		return vertices < size ? 0u : (vertices - size) / InputPrimitiveStep() + 1u;
	}
	[[nodiscard]] constexpr uint32_t InputVertexCount(uint32_t primitives) const {
		return primitives == 0u ? 0u : (primitives - 1u) * InputPrimitiveStep() + InputPrimitiveSize();
	}
};

struct ShaderTessellationInputInfo {
	uint32_t input_control_points  = 0;
	uint32_t output_control_points = 0;
	uint32_t ls_stride             = 0;
	uint32_t hs_stride             = 0;
	uint32_t patch_base            = 0;
	uint32_t patch_size            = 0;
	uint32_t domain                = 0;
	uint32_t partitioning          = 0;
	uint32_t output_topology       = 0;
	// Ghost of Yotei 2.000's merged LS+HS program, split in two (SplitMergedHullProgram, shader.cpp): the VGPRs its HS
	// part reads before writing them, which on the hardware still hold what the LS part left in the same lane (one wave
	// runs both parts); bit k = v<k>. The LS stage exports them after its outputs and the HS stage starts with them
	// (CarriedOutput/CarriedInput, Translate.cpp). Empty for every other pair (AnalyzeTessellationPrograms).
	std::array<uint32_t, 8> carried_vgprs {};
};

// GPU indirect draws: vertex program variant for indirect
// draws the GPU executes. The command processor would patch the user SGPRs the packet names with
// each draw's base vertex, start instance and draw index; the variant reads those SGPRs from the
// SPIR-V draw parameters instead, and a natively fetching shader sees zero-based vertex and instance
// indices, as with the patched SGPRs (the host draw passes the offsets in the indirect command).
// With draw base once every vertex program of the variant does (draw_base_once).
struct ShaderIndirectDrawInfo {
	static constexpr uint8_t NoSgpr = 0xff;

	bool    enabled       = false;
	uint8_t base_vertex   = NoSgpr; // user-data index (SPI_SHADER_USER_DATA_GS_<n>)
	uint8_t base_instance = NoSgpr;
	uint8_t draw_index    = NoSgpr;

	[[nodiscard]] uint32_t Key() const {
		return enabled ? 1u | (static_cast<uint32_t>(base_vertex) << 8u) |
		                     (static_cast<uint32_t>(base_instance) << 16u) |
		                     (static_cast<uint32_t>(draw_index) << 24u)
		               : 0u;
	}
};

struct ShaderVertexInputInfo {
	static constexpr int RES_MAX = 32;

	ShaderBufferResource    resources[RES_MAX];
	ShaderVertexDestination resources_dst[RES_MAX];
	ShaderVertexInputBuffer buffers[RES_MAX];
	ShaderStageRuntime      stage;
	ShaderType                  logical_stage        = ShaderType::Vertex;
	int                     resources_num       = 0;
	int                     fetch_attrib_reg    = 0;
	int                     fetch_buffer_reg    = 0;
	int                     buffers_num         = 0;
	uint32_t                wave_size           = 64;
	uint32_t                scratch_size_dwords = 0;
	// LDS_SIZE of the GS stage a non-NGG vertex shader runs on; usually zero.
	uint32_t                lds_size_dwords     = 0;
	uint32_t                pa_cl_vs_out_cntl    = 0;
	// PA_CL_CLIP_CNTL.UCP_ENA; set only on the last pre-rasterization stage.
	uint32_t                user_clip_planes     = 0;
	ShaderClipSpaceTransform clip_space;
	ShaderMeshInputInfo      mesh;
	ShaderTessellationInputInfo tess;
	ShaderIndirectDrawInfo  indirect_draw;
	// Set by the program cache for the indirect_draw variant: one of its SGPRs also feeds
	// descriptors or SRT reads the host evaluates, which would see the unpatched register value.
	bool                    indirect_draw_resource_sgprs = false;
	bool                    fetch_external      = false;
	bool                    fetch_embedded      = false;
	// Draw base once: v5/v8
	// start as the vertex and instance index without the draw's base vertex and start instance, as
	// the hardware seeds them, so the base the shader adds from its user SGPRs counts once.
	bool                    draw_base_once      = false;
	// The place of the former table-mode key (KYTY_PERF_TABLE_BUFFERS / project D), always 0: the graphics
	// journal's schema fingerprint takes sizeof (shaderPrecompile.cpp GraphicsSchemaFingerprint), so existing journals stay.
	uint8_t                 reserved_table_key  = 0;
};

inline constexpr uint8_t ComputeVariantPairOff     = 2u;
inline constexpr uint8_t ComputeVariantPairSpin    = 4u; // KYTY_DIAG_TWIN_DISPATCH: spin exchanges

struct ShaderComputeInputInfo: ShaderWorkgroupInputInfo {
	bool               async_compute              = false;
	uint8_t            float_mode                 = 0xc0;
	// Program variants of one dispatch (part of the key only when set; one byte, so sizeof stays):
	// ComputeVariantPairOff – KYTY_DIAG_TWIN_DISPATCH's default twin: a shader of the
	// pair list gets the two-lanes-per-invocation module instead of the paired one.
	uint8_t            compute_variant            = 0;
	uint32_t           dispatch_threads_num[3]    = {0, 0, 0};
	uint32_t           workgroup_counts[3]        = {0, 0, 0};
	uint64_t           indirect_args_address       = 0;
	bool               group_id[3]                = {false, false, false};
	bool               dispatch_thread_dimensions = false;
	bool               lds_storage                = false;
	int                thread_ids_num             = 0;
	int                workgroup_register         = 0;
	bool               tg_size_en                 = false;
	// COMPUTE_PGM_RSRC1.DX10_CLAMP (KYTY_FIX_CLAMP_NAN; part of the key only when that switch is on and the bit is 0).
	// In the padding before `stage`: sizeof (the precompile journal's schema) stays; the journal does not record it
	// (its replays compile with the usual 1).
	bool               dx10_clamp                 = true;
	ShaderStageRuntime stage;
};

enum class ShaderAlphaBlendSource : uint8_t { None, SourceAlpha, SourceAlphaOne, SourceAlphaZero };
enum class ShaderPixelParameterMode : uint8_t { FirstVertex, LastVertex, Rectangle };

struct ShaderPixelInputInfo {
	uint32_t                                       interpolator_settings[32]    = {0};
	uint32_t                                       input_num                    = 0;
	uint32_t                                       wave_size                    = 64;
	uint32_t                                       ps_system_input_base         = 0;
	uint32_t                                       custom_interpolation_mask    = 0;
	uint32_t                                       ps_perspective_center_vgpr   = UINT32_MAX;
	uint32_t                                       ps_perspective_sample_vgpr   = UINT32_MAX;
	uint32_t                                       ps_perspective_centroid_vgpr = UINT32_MAX;
	uint8_t                                        target_output_mode[8]        = {};
	uint32_t                                       target_shader_mask           = UINT32_MAX;
	std::array<Prospero::ColorComponentMapping, 8> target_export_mapping        = {};
	uint8_t                                        target_packed_unorm_mask      = 0;
	// CB round mode per converted export: zero adds half an LSB, one truncates.
	uint8_t                                        target_round_mode            = 0;
	uint32_t                                       scratch_size_dwords          = 0;
	// The place of the former table-mode key (KYTY_PERF_TABLE_BUFFERS / project D), always 0: the graphics
	// journal's schema fingerprint takes sizeof (shaderPrecompile.cpp GraphicsSchemaFingerprint), so existing journals stay.
	uint8_t                                        reserved_table_key           = 0;
	// EXTRA_LDS_SIZE: LDS reserved beyond the interpolants (VGPR spills to LDS).
	uint32_t                                       lds_size_dwords              = 0;
	bool                                           ps_pos_x                     = false;
	bool                                           ps_pos_y                     = false;
	bool                                           ps_pos_z                     = false;
	bool                                           ps_pos_w                     = false;
	bool                                           ps_front_face                = false;
	bool                                           ps_front_face_all_bits       = false;
	bool                                           ps_ancillary                 = false;
	bool                                           ps_no_perspective            = false;
	ShaderPixelParameterMode                       parameter_mode = ShaderPixelParameterMode::FirstVertex;
	bool                                           ps_pixel_kill_enable         = false;
	bool                                           ps_depth_export_enable       = false;
	bool                                           ps_sample_mask_export_enable = false;
	bool                                           ps_sample_shading            = false;
	bool                                           dual_source_blending         = false;
	bool                                           object_primitive_id          = false;
	// Export logical alpha or per-channel source factors through MRT1 after channel swizzling.
	ShaderAlphaBlendSource                         alpha_blend_source = ShaderAlphaBlendSource::None;
	bool                                           ps_early_z                   = false;
	bool                                           ps_execute_on_noop           = false;
	// The ObjPrimID write: the colour slot that receives this draw's hardware ObjPrimID write (VGT_OBJECT_ID plus
	// the primitive index when ADD_PIPED_PRIM_ID is set), 0xff for none. VGT_OBJECT_ID changes every draw, so it
	// is push constant dword objprim_push_dword, not part of the program. Bytes in the padding before `stage`,
	// so the precompile journal's schema (sizeof) stays; the journal stores them in its flag word.
	uint8_t                                        objprim_slot                 = 0xffu;
	uint8_t                                        objprim_push_dword           = 0xffu;
	bool                                           objprim_add_prim_id          = false;
	ShaderStageRuntime                             stage;

	bool HasPositionInput() const { return ps_pos_x || ps_pos_y || ps_pos_z || ps_pos_w; }
	bool HasObjprimOutput() const { return objprim_slot < 8u; }
};

union ShaderStageInputInfo {
	const ShaderVertexInputInfo*  vertex;
	const ShaderPixelInputInfo*   pixel;
	const ShaderComputeInputInfo* compute = nullptr;
};

inline const ShaderWorkgroupInputInfo* ShaderWorkgroupInput(ShaderType           stage,
                                                            ShaderStageInputInfo input) {
	switch (stage) {
		case ShaderType::Compute: return input.compute;
		case ShaderType::Mesh: return &input.vertex->mesh;
		default: return nullptr;
	}
}

uint32_t ShaderPixelParameterMappedLocation(const ShaderPixelInputInfo& info, uint32_t input);
uint32_t ShaderPixelParameterLocation(const ShaderPixelInputInfo& info,
                                      std::span<const uint32_t> active_inputs, uint32_t input);
bool     ShaderPixelParameterIsFlat(const ShaderPixelInputInfo& info, uint32_t input);
bool     ShaderPixelParameterIsCustom(const ShaderPixelInputInfo& info, uint32_t input);
// Draw base once: the hardware gives a vertex shader
// its vertex and instance index (v5/v8) without the draw's base vertex and start instance; the
// shader adds them itself from user SGPRs (s30/s31: V_SAD_U32 v5, s30, 0, v5). Without the switch
// v5/v8 are gl_VertexIndex/gl_InstanceIndex, which carry the host draw's offsets, and the base counts
// twice wherever the host draw carries it: indirect draws of vertex shaders with embedded-fetch
// metadata (CPU expansion and the GPU indirect variant), direct draws whose embedded-fetch
// plan moved the base SGPRs into the host draw. The switch takes those offsets off v5/v8 (vertex
// entry in Translate.cpp, the variant in spirvEmitterFlow.cpp) and leaves out of a CPU-expanded
// indirect draw the offsets no vertex input needs (IndirectDrawNativeOffsets, renderDraw.cpp).
// Mesh, LS and the other tessellation stages are unchanged.

struct ShaderSharp {
	uint16_t offset_dw : 15;
	uint16_t size      : 1;
};

struct ShaderUserData {
	uint16_t*    direct_resource_offset;
	ShaderSharp* sharp_resource_offset[4];
	uint16_t     eud_size_dw;
	uint16_t     srt_size_dw;
	uint16_t     direct_resource_count;
	uint16_t     sharp_resource_count[4];
};

struct ShaderRegisterRange {
	uint16_t start;
	uint16_t end;
};

struct ShaderDrawModifier {
	uint32_t enbl_start_vertex_offset   : 1;
	uint32_t enbl_start_index_offset    : 1;
	uint32_t enbl_start_instance_offset : 1;
	uint32_t enbl_draw_index            : 1;
	uint32_t enbl_user_vgprs            : 1;
	uint32_t render_target_slice_offset : 3;
	uint32_t fuse_draws                 : 1;
	uint32_t compiler_flags             : 23;
	uint32_t is_default                 : 1;
	uint32_t reserved                   : 31;
};

struct ShaderRegister {
	uint32_t offset;
	uint32_t value;
};

struct ShaderSpecialRegs {
	ShaderRegister      ge_cntl;
	ShaderRegister      vgt_shader_stages_en;
	uint32_t            dispatch_modifier;
	ShaderRegisterRange user_data_range;
	ShaderDrawModifier  draw_modifier;
	ShaderRegister      vgt_gs_out_prim_type;
	ShaderRegister      ge_user_vgpr_en;
};

struct ShaderSemantic {
	uint32_t semantic         : 8;
	uint32_t hardware_mapping : 8;
	uint32_t size_in_elements : 4;
	uint32_t is_f16           : 2;
	uint32_t is_flat_shaded   : 1;
	uint32_t is_linear        : 1;
	uint32_t is_custom        : 1;
	uint32_t static_vb_index  : 1;
	uint32_t static_attribute : 1;
	uint32_t reserved         : 1;
	uint32_t default_value    : 2;
	uint32_t default_value_hi : 2;
};

struct Shader {
	uint32_t             file_header;
	uint32_t             version;
	ShaderUserData*      user_data;
	const volatile void* code;
	ShaderRegister*      cx_registers;
	ShaderRegister*      sh_registers;
	ShaderSpecialRegs*   specials;
	ShaderSemantic*      input_semantics;
	ShaderSemantic*      output_semantics;
	uint32_t             header_size;
	uint32_t             shader_size;
	uint32_t             embedded_constant_buffer_size_dqw;
	uint32_t             target;
	uint32_t             num_input_semantics;
	uint16_t             scratch_size_dw_per_thread;
	uint16_t             num_output_semantics;
	uint16_t             special_sizes_bytes;
	uint8_t              type;
	uint8_t              num_cx_registers;
	uint8_t              num_sh_registers;
};

struct ShaderMappedData {
	Prospero::ShaderBinaryType type {};
	ShaderUserData* user_data           = nullptr;
	ShaderSemantic* input_semantics     = nullptr;
	uint32_t        num_input_semantics = 0;
	uint32_t        code_size_bytes     = 0;
	uint32_t        scratch_size_dwords = 0;
};

void ShaderInit();
void ShaderMapUserData(uint64_t addr, const ShaderMappedData& data);
// Milliseconds since the game first created the shader with this code hash (AgcCreateShader), -1 if never.
[[nodiscard]] double ShaderCreatedMsAgo(uint64_t hash);
// The shader mapped at addr and the hash of its code, if there is one (the preparation of a program exits
// when there is none). Callable from any thread; used by the resolve thread.
bool ShaderLookupMapped(uint64_t addr, ShaderMappedData* data, uint64_t* hash);
uint32_t ShaderPixelExportTarget(uint32_t shader_mask, uint32_t export_index);

void     ShaderDbgDumpInputInfo(const ShaderVertexInputInfo& info);
void     ShaderDbgDumpInputInfo(const ShaderPixelInputInfo& info);
void     ShaderDbgDumpInputInfo(const ShaderComputeInputInfo& info);

// KYTY_DIAG_SKIP_UNSUPPORTED_CS=1 (logged APPROXIMATION, default on, Common::EnvDefaults): a compute
// shader whose code contains instructions the decoder does not implement is not dispatched (=0: the translation stops
// the emulator naming the instruction). Also KYTY_DIAG_SKIP_CS_HASHES. KYTY_TRACE_PERF counts both (PERF workaround_hits
// skip_cs / skip_unsupported).
bool ShaderComputeSkippedAsUnsupported(const HW::ComputeShaderInfo& regs);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_H_ */
