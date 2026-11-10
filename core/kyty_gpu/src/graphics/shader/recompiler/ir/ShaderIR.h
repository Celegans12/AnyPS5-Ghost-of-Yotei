#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "common/liveAb.h"
#include "common/onceValue.h"
#include "common/stringUtils.h"
#include "common/shaderEnv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <cstdio>
#include <cstdlib>

#include <array>
#include <atomic>
#include <bit>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace Xbyak {
class CodeGenerator;
}

namespace Libs::Graphics::ShaderRecompiler::IR {

class SrtWalker;

// Finite descriptor domains also bound the arithmetic used to prove their indices.
constexpr uint32_t MaxIndirectDescriptorProbes = 65536u;

enum class ResourceKind {
	None,
	ScalarBuffer,
	ScalarAddress,
	Buffer,
	IndirectBuffer,
	Flat,
	FlatLocal,
	Global,
	Scratch,
	Lds,
	Gds,
	Image,
	Sampler
};

[[nodiscard]] constexpr bool IsAddressResourceKind(ResourceKind kind) {
	return kind == ResourceKind::ScalarAddress || kind == ResourceKind::Flat ||
	       kind == ResourceKind::FlatLocal || kind == ResourceKind::Global ||
	       kind == ResourceKind::Scratch;
}

struct MemoryInfo {
	ResourceKind            kind                     = ResourceKind::None;
	uint32_t                resource                 = 0;
	uint32_t                sampler                  = 0;
	uint32_t                offset                   = 0;
	uint32_t                secondary_offset         = 0;
	uint32_t                dmask                    = 0;
	uint32_t                data_dwords              = 1;
	uint32_t                data_bits                = 32;
	uint32_t                component_index          = 0;
	uint32_t                component_count          = 1;
	uint32_t                data_format              = 0;
	uint32_t                number_format            = 0;
	uint32_t                image_sample_flags       = 0;
	Decoder::ImageDimension image_dimension          = Decoder::ImageDimension::Unknown;
	uint32_t                image_address_components = 0;
	bool                    address_is_full                                       = false;
	bool                    data_signed                                           = false;
	bool                    typed                                                 = false;
	bool                    formatted                                             = false;
	bool                    image_has_mip                                         = false;
	bool                    image_r128                                            = false;
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;
	// Record-keyed V#s: a BDA access through a record-keyed V# (its guest ranges are listed per
	// materialization), so its stores and atomics are no writes through unknown addresses.
	bool                    record_keyed                                          = false;

	[[nodiscard]] bool IsCoherentBufferPairLoad(ValueOpcode opcode) const {
		return kind == ResourceKind::Buffer && opcode == ValueOpcode::LoadBufferU32x2 && coherent &&
		       !typed && !formatted && data_bits == 32u;
	}

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		// Raw byte/short loads read their bytes through the same GPU-selected address.
		if (!typed && !formatted &&
		    ((opcode == ValueOpcode::LoadBufferU8 && data_bits == 8u) ||
		     (opcode == ValueOpcode::LoadBufferU16 && data_bits == 16u)))
			return true;
		return !typed && (data_bits == 32u || (formatted && data_bits == 16u)) &&
		       (formatted ? opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4
		                  : opcode == ValueOpcode::ReadConstBuffer ||
		                        opcode == ValueOpcode::LoadBufferU32 ||
		                        opcode == ValueOpcode::LoadBufferU32x2 ||
		                        opcode == ValueOpcode::LoadBufferU32x3 ||
		                        opcode == ValueOpcode::LoadBufferU32x4);
	}

	// record-keyed (D6): raw DWORD stores through a GPU-selected descriptor.
	// The written pages are not reported to the host (no GPU ownership tracking).
	// record-keyed (D6): 32-bit raw atomics through a GPU-selected descriptor (BDA).
	[[nodiscard]] bool SupportsUntrackedIndirectBufferAtomic(ValueOpcode opcode) const {
		if (typed || formatted || data_bits != 32u) return false;
		switch (opcode) {
			case ValueOpcode::BufferAtomicIAdd32:
			case ValueOpcode::BufferAtomicISub32:
			case ValueOpcode::BufferAtomicSMin32:
			case ValueOpcode::BufferAtomicUMin32:
			case ValueOpcode::BufferAtomicSMax32:
			case ValueOpcode::BufferAtomicUMax32:
			case ValueOpcode::BufferAtomicAnd32:
			case ValueOpcode::BufferAtomicOr32:
			case ValueOpcode::BufferAtomicXor32:
			case ValueOpcode::BufferAtomicSwap32:
			case ValueOpcode::BufferAtomicCmpSwap32: return true;
			default: return false;
		}
	}

	// A formatted X..XYZW store of 32-bit VGPR data through
	// a record-keyed V#; StoreIndirectFormatted encodes it in the V#'s runtime format (at most 16 bytes per element).
	[[nodiscard]] bool SupportsRecordKeyedFormattedStore(ValueOpcode opcode) const {
		return formatted && !typed && data_bits == 32u &&
		       (opcode == ValueOpcode::StoreBufferU32 || opcode == ValueOpcode::StoreBufferU32x2 ||
		        opcode == ValueOpcode::StoreBufferU32x3 || opcode == ValueOpcode::StoreBufferU32x4);
	}

	[[nodiscard]] bool SupportsUntrackedIndirectBufferStore(ValueOpcode opcode) const {
		if (!typed && !formatted &&
		    ((opcode == ValueOpcode::StoreBufferU8 && data_bits == 8u) ||
		     (opcode == ValueOpcode::StoreBufferU16 && data_bits == 16u)))
			return true;
		return !typed && !formatted && data_bits == 32u &&
		       (opcode == ValueOpcode::StoreBufferU32 || opcode == ValueOpcode::StoreBufferU32x2 ||
		        opcode == ValueOpcode::StoreBufferU32x3 || opcode == ValueOpcode::StoreBufferU32x4);
	}

	bool operator==(const MemoryInfo& other) const = default;
};

// Record-keyed V#s. B1: formatted stores through a GPU-selected V# that a bounded loop reads
// from a record table are record-keyed like raw stores (their guest ranges are listed per materialization and obtained for
// writing) and written through BDA in the runtime format. B2: a record-keyed V# whose key no bounded loop limits lists every
// record of its table (a read past the table's bytes reads a zero, i.e. empty, V#). Candidates are the records at whole
// strides from the field (the key times the stride is assumed not to wrap 32 bits). B3: the record table of a record-keyed
// V# may be a 64-bit address (scalar loads from memory) instead of a V#, when a bounded loop limits the key: the host reads
// the count's records from that address (no size: without a loop the V# stays untracked). B4: a record-keyed key that no
// loop bounds is bounded by its own form when that proves a largest value (BoundedU32Max: e.g. Water Lighting's select of
// 0..4). B5 (game counts): a record-keyed address table whose filled record count the game keeps in CPU memory the shader
// does not read (e.g. Update Oct Maps 34e090c6: the CPU fills GI-volume slots 0..count-1 and the job list names only those)
// lists exactly those records. B6 (path proofs): a V# whose words are phis of reads chosen by the path through the shader
// (e.g. CutHeight e3030eb1: fixed SRT reads, a bounding box before its loop and the counter V# inside it) is the one read
// that can hold where its accesses run: every access requires a VGPR state equal to a constant, and pairing that state with
// the V# along the same phis leaves a single read (a direct candidate source).
// D6 step K1 (composed V#s): a store or atomic through a V# the shader composes from uniform values is record-keyed. Its
// words are values the host can evaluate once the selects of VGPR writes made under a full EXEC give their new value (a
// predicated write with every lane on holds that value on every lane, so a read of any lane reads it: exact). Example: the
// radix sort 32acc45f's look-back records, V# {SRT+32 address, stride 1024, NUM_RECORDS = ceil(N / T), format}. Such a V# is
// a direct candidate source (as a path proof's), evaluated at every materialization; the GPU keeps reading its words (BDA).
// Its accesses may add a vector offset whose value bits bound it (e.g. (lane & 0xff) << 2) instead of a fixed one.

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind   = ExportTargetKind::Unknown;
	uint32_t         target = 0;
	uint32_t         index  = 0;
	uint32_t         en     = 0;
	bool             done   = false;
	bool             compr  = false;
	bool             vm     = false;

	bool operator==(const ExportInfo& other) const = default;
};

struct BufferResource {
	static constexpr uint32_t NoImageAlias = UINT32_MAX;
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               descriptor_index   = UINT32_MAX;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               write_extent       = 0; // UINT32_MAX once a write is runtime-addressed or formatted
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	uint32_t               image_alias        = NoImageAlias;
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	// While every write has an address fixed in the shader (raw data, no index or vector offset, immediate scalar
	// offset): true, with the written bytes [first, second) from the descriptor base, sorted and coalesced. False
	// (and empty) once any write does not, or before the first write.
	bool                                       fixed_writes = false;
	std::vector<std::pair<uint32_t, uint32_t>> fixed_write_ranges;
	// D4b step 1 (ResourceTracking.cpp AppendRange): every write of this pool is an append store
	// whose record index comes from the 4-byte atomic counter of buffer resource `append_counter`; a dispatch writes
	// bytes [append_min_offset, append_max_end) of records [counter before, counter after) (stride from the V#).
	static constexpr uint32_t NoAppendCounter = UINT32_MAX;
	uint32_t append_counter    = NoAppendCounter;
	uint32_t append_min_offset = 0;
	uint32_t append_max_end    = 0;
	bool                   live               = false;
	uint32_t               indirect_root              = NoIndirectBuffer;
	uint32_t               indirect_mapping_offset    = 0;
	uint32_t               indirect_search_iterations = 0;
	std::vector<uint32_t>  indirect_resources;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;

	uint32_t                      source            = 0;
	uint32_t                      descriptor_index  = UINT32_MAX;
	uint32_t                      first_use_pc      = 0;
	ImageResourceClass            resource_class    = ImageResourceClass::None;
	Prospero::TextureNumericClass numeric_class     = Prospero::TextureNumericClass::Unsupported;
	Decoder::ImageDimension       dimension         = Decoder::ImageDimension::Unknown;
	ImageMipMode                  mip_mode          = ImageMipMode::None;
	uint32_t                      mip_count         = 1;
	Prospero::BufferFormat        conversion_format = Prospero::BufferFormat::kInvalid;
	uint32_t                      shader_swizzle    = ShaderImageIdentitySwizzle;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          atomic64          = false;
	bool                          unfiltered_access_only = true;
	bool                          depth_compare     = false;
	bool                          cube              = false;
	bool                          r128              = false;
	bool                          heap              = false; // Resident image heap root or kind.
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	// KYTY_BINDLESS: sampled from the set-1 arrays at the slot translation[region + key/4] gives;
	// flattened_srt[indirect_mapping_offset] = region, [+1] = entry count (patched by the host).
	bool                          bindless          = false;
	// Non-zero: the key is a record index and entry = key * bindless_stride_dwords (record tables).
	uint32_t                      bindless_stride_dwords = 0;
	std::vector<uint32_t>         indirect_resources;
	// The bindless tables: where the handles of a bounded heap table (a V# the shader
	// indexes by byte offset, IsBoundedDescriptorTable) read their descriptors, as {record stride in bytes (0: the
	// V#'s own stride, read at draw time), byte offset of the descriptor in the record}. The host translates exactly
	// these fields of every record before the draw. bindless_fields_complete is false when some handle's key is not
	// of that form; the host then leaves the heap to the feedback path. Recorded by ResourceTracking.
	std::vector<std::pair<uint32_t, uint32_t>> bindless_fields;
	bool                                       bindless_fields_complete = true;
	// KYTY_PERF_BINDLESS_CLASSES (BindlessClassesRequested): the resource serves exactly one field of a bounded heap
	// table (its only bindless_fields entry), so its numeric class can follow what that field's records hold.
	// Set by ResourceTracking only while the switch is on (false otherwise).
	bool                                       bindless_field_split = false;

	bool operator==(const ImageResource& other) const = default;
};

[[nodiscard]] constexpr Prospero::BufferFormat
RawSignedStorageFormat(const ImageResource& image, Prospero::BufferFormat format) {
	if (image.resource_class != ImageResourceClass::Storage || !image.written)
		return Prospero::BufferFormat::kInvalid;
	if (format == Prospero::BufferFormat::k32SInt && (image.atomic ? !image.atomic64 : !image.read))
		return Prospero::BufferFormat::k32UInt;
	if (format == Prospero::BufferFormat::k16SInt && !image.read && !image.atomic)
		return Prospero::BufferFormat::k16UInt;
	return Prospero::BufferFormat::kInvalid;
}

// A raw load of a kDepth k16UInt T# reads a D16 surface's codes. Vulkan cannot view D16 as
// R16Uint, so the host binds it as UNorm and the shader converts back.
[[nodiscard]] constexpr bool ReadsDepthBits(const ImageResource& image) {
	return image.conversion_format == Prospero::BufferFormat::k16UInt;
}

struct SamplerResource {
	static constexpr uint32_t NoIndirectSampler = UINT32_MAX;

	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	// Native filtering/border variants share the original sampler's runtime descriptor.
	uint32_t snapshot_index        = 0;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;
	// GPU-selected sampler: like an indirect image, a runtime key selects one of
	// indirect_resources through the flattened key/ordinal mapping.
	uint32_t              indirect_root              = NoIndirectSampler;
	uint32_t              indirect_mapping_offset    = 0;
	uint32_t              indirect_search_iterations = 0;
	std::vector<uint32_t> indirect_resources;

	bool operator==(const SamplerResource& other) const = default;
};

struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

// Record-keyed V#s: the BDA accesses through one record-keyed V# source
// (DescriptorSource::IndirectDescriptor::record_keyed). Every access has a fixed vector and scalar offset, so the
// bytes it touches lie in [access_begin, access_end) plus index * stride from a candidate V#'s base; `indexed` when
// some access adds a lane index. Each materialization turns it into ResourceSnapshot::indirect_ranges.
struct IndirectRangeResource {
	uint32_t source       = 0;
	uint32_t first_use_pc = 0;
	uint32_t access_begin = UINT32_MAX;
	uint64_t access_end   = 0;
	bool     indexed      = false;
	bool     read         = false;
	bool     written      = false;
	bool     atomic       = false;

	bool operator==(const IndirectRangeResource& other) const = default;
};

enum class TessellationAttribute {
	LocalOutput,
	ControlInput,
	ControlOutput,
	EvaluationInput,
	PatchOutput,
	PatchInput,
	Factor,
	// The VGPRs a split merged LS+HS program carries from its LS part into its HS part (ShaderTessellationInputInfo::
	// carried_vgprs): the LS stage's per-vertex output and the HS stage's per-control-point input, one u32 each, packed
	// four per vec4 after the LS outputs; address = 4 * the register's rank in the set.
	CarriedOutput,
	CarriedInput
};
constexpr uint32_t NumTessellationAttributes =
    static_cast<uint32_t>(TessellationAttribute::CarriedInput) + 1u;

enum class StageInputKind {
	VertexIndex,
	InvocationId,
	PrimitiveId,
	TessCoord,
	InstanceIndex,
	FragCoord,
	FrontFacing,
	PackedAncillary,
	Layer,
	SampleId,
	BaryCoordSmooth,
	BaryCoordSmoothSample,
	BaryCoordSmoothCentroid,
	BaryCoordNoPerspective,
	WorkgroupId,
	NumWorkgroups,
	LocalInvocationId,
	LocalInvocationIndex,
	GlobalInvocationId,
	Parameter,
	// SPIR-V draw parameters, read only by the GPU indirect vertex variant
	// (ShaderIndirectDrawInfo); the IR itself never names them.
	BaseVertex,
	BaseInstance,
	DrawIndex,
};

enum class StageOutputKind {
	Position,
	Parameter,
	Mrt,
	ObjectPrimitiveId,
	Depth,
	SampleMask,
	PointSize,
	ClipDistance,
	CullDistance,
	Layer,
	ViewportIndex,
	// The ObjPrimID write: the 32-bit ObjPrimID the hardware writes to a colour slot the pixel shader does not
	// export (location = the slot).
	ObjPrimId
};

struct PositionExportComponent {
	uint32_t clip_distance = UINT32_MAX;
	uint32_t cull_distance = UINT32_MAX;
	bool     point_size     = false;
	bool     layer          = false;
	bool     viewport       = false;
};

inline PositionExportComponent DecodePositionExportComponent(uint32_t control,
	                                                           uint32_t pos_index,
	                                                           uint32_t component) {
	PositionExportComponent result;
	if (pos_index == 0 || component >= 4) {
		return result;
	}

	uint32_t slot   = pos_index - 1;
	uint32_t vector = 3;
	for (uint32_t i = 0; i < 3; i++) {
		if ((control & (1u << (21u + i))) != 0) {
			if (slot == 0) {
				vector = i;
				break;
			}
			slot--;
		}
	}
	if (vector == 3) {
		return result;
	}

	if (vector == 0) {
		result.point_size = component == 0 && (control & (1u << 16u)) != 0;
		result.layer      = component == 2 && (control & (1u << 18u)) != 0;
		result.viewport   = component == 2 && (control & (1u << 19u)) != 0;
		return result;
	}

	const auto scalar = (vector - 1) * 4 + component;
	const auto lower  = (1u << scalar) - 1u;
	const auto clip   = control & 0xffu;
	const auto cull   = (control >> 8u) & 0xffu;
	if ((clip & (1u << scalar)) != 0) {
		result.clip_distance = std::popcount(clip & lower);
	}
	if ((cull & (1u << scalar)) != 0) {
		result.cull_distance = std::popcount(cull & lower);
	}
	return result;
}

struct StageInput {
	StageInputKind kind            = StageInputKind::VertexIndex;
	uint32_t       location        = 0;
	uint32_t       component_count = 1;
	std::string    debug_name;
	bool           per_vertex = false;

	bool operator==(const StageInput& other) const = default;
};

struct StageOutput {
	StageOutputKind kind     = StageOutputKind::Parameter;
	uint32_t        index    = 0;
	uint32_t        location = 0;
	std::string     debug_name;

	bool operator==(const StageOutput& other) const = default;
};

inline constexpr uint32_t FirstImageBinding           = 1u;
inline constexpr uint32_t FirstComparisonImageBinding = 22u;
inline constexpr uint32_t FirstStorageImageBinding    = 29u;
inline constexpr uint32_t ImageBindingCount           = 50u;
// Resident image heaps share one sampled-image array at binding 0 of this set. A heap slot
// word is kind << ImageHeapKindShift | element; element 0 is the null descriptor.
inline constexpr uint32_t ImageHeapSet                = 1u;
inline constexpr uint32_t ImageHeapCapacity           = 65536u;
inline constexpr uint32_t ImageHeapKindShift          = 20u;

enum class DescriptorBindingKind : uint32_t {
	Buffers  = 0u,
	Samplers = FirstImageBinding + ImageBindingCount,
	Gds,
	BdaPagetable,
	FaultBuffer,
	FlattenedSrt,
	ShaderData,
	SharedMemory,
	// Uniform const buffers: the flattened SRT words ReadConst reads, as a uniform buffer.
	FlattenedSrtUniform,
	Count,
};

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 51u);
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 59u);

// The bindings of one stage group: kinds before FlattenedSrtUniform keep kind + group * 58, FlattenedSrtUniform
// follows all four groups (232 + group), so no other binding number differs from upstream's kind + group * 58.
inline constexpr uint32_t BindingGroupStride = 58u;

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 6;
	static constexpr uint32_t NoStart    = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

	// User clip planes 0..N follow the mesh draw parameters or object ID, four dwords each.
	static constexpr uint32_t UserClipPlaneDword = MeshDrawDwordCount;

	[[nodiscard]] static constexpr uint32_t GraphicsDrawDwords(bool mesh, bool object_id,
	                                                           uint32_t user_clip_planes = 0) {
		if (user_clip_planes != 0) {
			return UserClipPlaneDword + 4u * std::bit_width(user_clip_planes);
		}
		return mesh ? MeshDrawDwordCount : static_cast<uint32_t>(object_id);
	}

	[[nodiscard]] static constexpr bool CanFit(uint32_t start, uint32_t size) {
		return size != 0 && start <= DwordCount && size <= DwordCount - start;
	}

	// Stage slots: a push cursor may carry where a
	// stage's data goes besides the next free dword. FromEnd (the pixel stage): its data ends at `end` (bits 8-15), the
	// first `front` dwords (bits 0-7) are the draw's; Limited (the vertex stages after it): from `front`, up to `limit`
	// (bits 8-15, the pixel data's start). So each stage's start depends on its own data and the draw's dwords only, not
	// on the other stage's size. A plain cursor (no flag) is today's: from the cursor to the end of the block.
	static constexpr uint32_t CursorFromEnd  = 1u << 30u;
	static constexpr uint32_t CursorLimited  = 1u << 29u;
	[[nodiscard]] static constexpr bool     IsSlotCursor(uint32_t cursor) {
        return cursor != NoStart && (cursor & (CursorFromEnd | CursorLimited)) != 0u;
	}
	[[nodiscard]] static constexpr uint32_t CursorFront(uint32_t cursor) {
		return IsSlotCursor(cursor) ? cursor & 0xffu : cursor;
	}
	[[nodiscard]] static constexpr uint32_t CursorBound(uint32_t cursor) {
		return IsSlotCursor(cursor) ? (cursor >> 8u) & 0xffu : DwordCount;
	}
	[[nodiscard]] static constexpr uint32_t FromEndCursor(uint32_t front, uint32_t end) {
		return CursorFromEnd | end << 8u | front;
	}
	[[nodiscard]] static constexpr uint32_t LimitedCursor(uint32_t front, uint32_t limit) {
		return CursorLimited | limit << 8u | front;
	}
	[[nodiscard]] static constexpr uint32_t StartFor(uint32_t cursor, uint32_t size) {
		if (!IsSlotCursor(cursor)) return CanFit(cursor, size) ? cursor : NoStart;
		const uint32_t front = CursorFront(cursor);
		const uint32_t bound = CursorBound(cursor);
		if (size == 0 || bound > DwordCount || front > bound || size > bound - front) return NoStart;
		return (cursor & CursorFromEnd) != 0u ? bound - size : front;
	}
};

static_assert(sizeof(PushData) == 128);
constexpr uint32_t NativePushConstantSize = sizeof(PushData);

[[nodiscard]] constexpr uint32_t NativeBinding(ShaderType stage, DescriptorBindingKind kind) {
	const uint32_t group = stage == ShaderType::Pixel                    ? 1u
	                       : stage == ShaderType::TessellationControl    ? 2u
	                       : stage == ShaderType::TessellationEvaluation ? 3u
	                                                                     : 0u;
	if (kind == DescriptorBindingKind::FlattenedSrtUniform) return 4u * BindingGroupStride + group;
	return static_cast<uint32_t>(kind) + group * BindingGroupStride;
}

[[nodiscard]] constexpr ImageResourceClass ImageBindingResourceClass(DescriptorBindingKind kind) {
	const auto value = static_cast<uint32_t>(kind);
	if (value >= FirstImageBinding && value < FirstStorageImageBinding) {
		return ImageResourceClass::Sampled;
	}
	if (value >= FirstStorageImageBinding &&
	    value < static_cast<uint32_t>(DescriptorBindingKind::Samplers)) {
		return ImageResourceClass::Storage;
	}
	return ImageResourceClass::None;
}

[[nodiscard]] constexpr uint32_t ImageBindingIndex(DescriptorBindingKind kind) {
	return static_cast<uint32_t>(kind) - FirstImageBinding;
}

[[nodiscard]] constexpr std::optional<DescriptorBindingKind>
DescriptorBindingForImage(const ImageResource& image) {
	constexpr uint32_t SampledFloatBinding = 1u;
	constexpr uint32_t SampledUintBinding  = 8u;
	constexpr uint32_t SampledSintBinding  = 15u;
	constexpr uint32_t StorageFloatBinding = FirstStorageImageBinding;
	constexpr uint32_t StorageUintBinding  = StorageFloatBinding + 6u;
	constexpr uint32_t AtomicUintBinding   = StorageUintBinding + 6u;

	uint32_t base    = 0;
	bool     sampled = false;
	if (image.resource_class == ImageResourceClass::Sampled) {
		if (image.atomic) {
			return std::nullopt;
		}
		sampled = true;
		switch (image.numeric_class) {
			case Prospero::TextureNumericClass::Float:
				base = image.depth_compare ? FirstComparisonImageBinding : SampledFloatBinding;
				break;
			case Prospero::TextureNumericClass::Uint: base = SampledUintBinding; break;
			case Prospero::TextureNumericClass::Sint: base = SampledSintBinding; break;
			case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
			default: return std::nullopt;
		}
		if (image.depth_compare && image.numeric_class != Prospero::TextureNumericClass::Float) {
			return std::nullopt;
		}
	} else if (image.resource_class == ImageResourceClass::Storage) {
		if (image.atomic) {
			if (image.numeric_class != Prospero::TextureNumericClass::Uint) {
				return std::nullopt;
			}
			base = AtomicUintBinding + (image.atomic64 ? 5u : 0u);
		} else {
			switch (image.numeric_class) {
				case Prospero::TextureNumericClass::Float: base = StorageFloatBinding; break;
				case Prospero::TextureNumericClass::Uint: base = StorageUintBinding; break;
				case Prospero::TextureNumericClass::Sint:
				case Prospero::TextureNumericClass::Unsupported: return std::nullopt;
				default: return std::nullopt;
			}
		}
	} else {
		return std::nullopt;
	}

	uint32_t dimension = 0;
	switch (image.dimension) {
		case Decoder::ImageDimension::Dim1D: break;
		case Decoder::ImageDimension::Dim1DArray: dimension = 1u; break;
		case Decoder::ImageDimension::Dim2D: dimension = 2u; break;
		case Decoder::ImageDimension::Dim2DArray: dimension = 3u; break;
		case Decoder::ImageDimension::Dim2DMsaa:
			if (image.atomic) {
				return std::nullopt;
			}
			dimension = sampled ? 4u : 5u;
			break;
		case Decoder::ImageDimension::Dim2DMsaaArray:
			if (!sampled) {
				return std::nullopt;
			}
			dimension = 5u;
			break;
		case Decoder::ImageDimension::Dim3D: dimension = sampled ? 6u : 4u; break;
		case Decoder::ImageDimension::Unknown: return std::nullopt;
		default: return std::nullopt;
	}
	return static_cast<DescriptorBindingKind>(base + dimension);
}

// KYTY_BINDLESS=1: GPU-selected sampled images from a bounded descriptor heap use bindless arrays.
inline bool BindlessImagesRequested() {
	static const bool enabled = [] {
		// Default from Common::EnvDefaults; =0 turns it off.
		const char* value = Common::ShaderEnv::Read("KYTY_BINDLESS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// M120: an image table the GPU indexes (indirect_root with >= 2 candidates) is written the way it is read: the write
// selects its candidate (EmitImageAccess, as ImageRead does; "Update Oct Maps", CS 6cc64dee, writes the octahedral probe
// maps of every GI volume), and a candidate that differs from the table's exemplar only in dimension
// (coordinates taken as the candidate's own type needs them) or in its store swizzle keeps its own type instead of
// being bound as null.

// The bindless tables (the only way; they act with KYTY_BINDLESS=1): a bindless heap is translated as the
// table it is, before the draw that reads it, instead of key by key after a draw asked for a key (feedback). The table is
// the draw's V# (or record pointer); its descriptor fields come from the shader (ImageResource::bindless_fields for heap
// tables, the record stride and field offset for record tables); every field of every record is read and translated
// through the texture cache when the draw is prepared, so a key never waits for feedback and never samples the
// pending placeholder (RenderExecutor::PrepareBindlessTable).

// KYTY_PERF_BINDLESS_CLASSES=1|2 (default 1, bindless E5: at the gate 3 table fields read with loads and
// gathers hold 8_UINT terrain maps that sampled a black placeholder; =0 off; acts with KYTY_BINDLESS=1): bindless table
// fields that hold integer textures. Every bindless resource was specialized as float, and one resource served every
// field of a table with the same image properties: Ghost of Yotei's terrain tile record (440 bytes) holds two 8_UINT maps
// (+96, a 513x513 material-index map the terrain gathers, and +192, 33x33) among eight float textures, and every one of
// those T#s sampled a placeholder ("t9f5" in the PERF bindless_tables incompatible list). With the switch:
//  * a field of a bounded heap table gets an image resource of its own (ResourceTracking): =1 the fields a shader reads
//    with an image load or a gather (the instructions that read integer texels; an integer texture is never filtered),
//    =2 every field (more resources per draw);
//  * the class of such a resource follows its field's records: Uint or Sint when they hold valid integer T#s of one
//    class and no float T# besides tiny (at most 4x4) defaults, Float otherwise. It is decided once per (shader,
//    resource, field) at the field's first materialization (one read of the field on the GPU thread) and raised when the
//    host's table translation later finds integer T#s in a field typed float; a registry keeps it (ResourceMaterialization);
//  * an integer field samples integer views of its own class in the set-1 arrays (the SPIR-V declares one runtime array
//    per image type, aliasing the dimension's binding) with integer samplers (point, integer border), as a bound integer
//    T# does; its records that hold no T# of its class read the zero placeholder of the class (the console's gather of a
//    1x1 black default reads zeros too);
//  * every array gets a 1x1 placeholder per class: float opaque black (0, 0, 0, 1) instead of transparent black (the
//    game's defaults are opaque), uint and sint zero (KYTY_BINDLESS_DEBUG_COLORS=1 keeps its colours).
// It changes the SPIR-V, so it stays in the warm-list / pipeline-cache environment fingerprint; it changes the number
// of image resources too, so each mode keeps precompile journals and a warm list of its own (PipelineCache), and a
// shader the field resources would take past the resource limits is tracked without them.
inline uint32_t BindlessClassesMode() {
	static constinit Common::OnceValue<uint32_t> mode_once;
	const uint32_t mode = mode_once.Get([] {
		const char* value = Common::ShaderEnv::Read("KYTY_PERF_BINDLESS_CLASSES");
		const auto  m     = value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 1u;
		if (m != 0)
			std::fprintf(stderr, "PERF_BINDLESS_CLASSES enabled mode=%u (%s)%s\n", m,
			             m >= 2u ? "every table field its own resource" : "loaded and gathered table fields",
			             BindlessImagesRequested() ? "" : " (no effect without KYTY_BINDLESS=1)");
		return m;
	});
	return mode;
}

inline bool BindlessClassesRequested() {
	return BindlessClassesMode() != 0u && BindlessImagesRequested();
}

// KYTY_PERF_BINDLESS_CLASSES: what the records of one bindless table field hold, counted per descriptor slot
// (TallyBindlessFieldDescriptor and DecideBindlessFieldClass, ResourceMaterialization.h).
struct BindlessFieldTally {
	uint32_t uint_count  = 0; // valid T#s of the array's shape in an unsigned-integer format
	uint32_t sint_count  = 0; // ... in a signed-integer format
	uint32_t float_count = 0; // ... in a float-class format (unorm, snorm, float, depth, block-compressed)
	uint32_t float_tiny  = 0; // of those, textures of at most 4x4 texels (the defaults a game writes for a missing one)
	uint32_t other       = 0; // words that are no such T# (data, null, another shape, an integer format the host
	                          // converts in the shader)
};

// Uniform const buffers: the constant-index reads of the two host-made buffers are uniform buffer reads instead of
// storage buffer reads (NVIDIA serves them from the constant cache, often as instruction operands, instead of one L1
// load per read):
//  * ReadConst: flattened SRT slot `i` (an immediate below srt_reads.size(), which the host's flat buffer always
//    holds, see SrtWalker::RefreshFlatBuffer) through a second binding of the same uploaded words
//    (FlattenedSrtUniform); the SRT mapping reads (GPU-selected descriptors) keep the storage binding;
//  * shader data dword `i` (an immediate below ShaderDataDwords()) when it is not push data: the ShaderData binding
//    itself is a uniform buffer.
// Every read is a constant index inside the bound range, so the same bytes are read: same results.

// The smallest maxUniformBufferRange Vulkan allows: a uniform view of this size binds on every device.
inline constexpr uint32_t UniformBufferMaxBytes = 16384u;

// The flattened SRT words of ReadConst go through FlattenedSrtUniform (when they fit one uniform buffer).
inline bool ReadConstUniform(size_t srt_reads) {
	return srt_reads != 0u && srt_reads <= UniformBufferMaxBytes / 4u;
}

// M115: a formatted buffer whose V# is all zeros (a
// null buffer) takes the specialization of its binding's last live V# (format, swizzle, run-time stride), so null and
// live share one permutation; the emitter then returns what the null V# gives (zero in every channel) when the run-time
// stride is 0. Without it every null/live combination of a shader's formatted buffers is its own SPIR-V (072a51c6:
// ~820k words, 86-96 s per cold compile).
// The null share journal: a session whose first call of a program has a null formatted buffer
// would compile a null/live combination of its own (two runs: 072a51c6 had 4 recorded permutations
// differing only in b15/b16 null or live, 66-67 s per cold compile). At a program's first use its last
// live values are taken from the precompile journal's records of the same program (file order, the later record wins,
// as the later call wins in a session), so a null buffer gets the permutation an earlier session compiled. The share
// itself is unchanged (a null V# reads zeros and drops stores in any live permutation); only which equivalent
// permutation is compiled changes. Counted at each seeded binding's first live V# of the session: whether its format and
// swizzle are the journal's (checks, mismatches: must stay 0).
// The shader is in the pair list (WavePairRequested), or the pair rule takes it (a compute workgroup of `threads`
// = 64); it may still fall back to two lanes per invocation.
bool WavePairPlanned(uint64_t shader_hash, uint32_t threads, size_t code_words);
// KYTY_DIAG_TWIN_DISPATCH=rule: the emitter paired the shader by the pair rule (not by the list).
bool WavePairRulePaired(uint64_t shader_hash);

// KYTY_FIX_CLAMP_NAN (M124, default 0 = today; 1 on; 2 check): the GCN `.clamp` output modifier with DX10_CLAMP set
// (SPI_SHADER_PGM_RSRC1_* / COMPUTE_PGM_RSRC1 bit 21) turns NaN into 0. Today it is GLSLstd450FClamp, whose result for
// NaN is undefined (the module declares SignedZeroInfNanPreserve). =1: NClamp (NMin/NMax semantics: NaN -> 0).
// =2: FClamp kept; every saturate whose FClamp result is NaN is counted in the fault buffer's diagnostic words
// ClampNanDiagWord: +0 f32 hits, +1 packed-f16 hits, +2 hit records taken, +3.. the first ClampNanDiagSlots records
// (shader hash low, high) (PERF clamp_nan verify).
inline constexpr uint32_t ClampNanDiagWord  = 24;
inline constexpr uint32_t ClampNanDiagSlots = 2;
// F1 (grouped IndirectBuffer reads; the only path since 26.10, measured at the Atsu
// house: cloth CS 29.4 -> 14.6 ms GPU per frame, 12.1 -> 14.3 fps; the test switch's check mode compared 10.7e9 dwords with
// the per-dword path, 0 differ): a raw BUFFER_LOAD_DWORDX2..X4 through a GPU-selected V# (ResourceKind::IndirectBuffer)
// is one call per wave half of indirect_raw_x<N>: the V# is decoded once, every dword keeps its own bounds test, and
// when all N dwords are in bounds and lie in one 16 KiB page the page is looked up once and the N dwords are loaded back
// to back; otherwise the N per-dword calls of indirect_raw_dword run. Same addresses, same values, same fault bits (one
// page, so N identical lookups set the same bit once).
inline int ClampNanMode() {
	static const int mode = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_FIX_CLAMP_NAN");
		const int   m     = value == nullptr ? 0 : value[0] == '1' ? 1 : value[0] == '2' ? 2 : 0;
		if (m != 0) std::fprintf(stderr, "FIX_CLAMP_NAN mode=%d\n", m);
		return m;
	}();
	return mode;
}
// KYTY_FIX_CLAMP_NAN_HASHES=<hex,...> (diagnostic scope): the switch applies only to these shaders (empty = all), so a
// test does not recompile every program with a saturate.
inline int ClampNanModeFor(uint64_t shader_hash) {
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		const char*           value = Common::ShaderEnv::Read("KYTY_FIX_CLAMP_NAN_HASHES");
		while (value != nullptr && *value != 0) {
			char* end = nullptr;
			list.push_back(std::strtoull(value, &end, 16));
			if (end == nullptr || *end != ',') break;
			value = end + 1;
		}
		return list;
	}();
	const int mode = ClampNanMode();
	if (mode == 0 || hashes.empty()) return mode;
	return std::find(hashes.begin(), hashes.end(), shader_hash) != hashes.end() ? mode : 0;
}

struct Bindings {
	std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)> descriptor_counts {};
	uint64_t descriptor_mask = 0;
	uint32_t push_data_start_dword = PushData::NoStart;
	uint32_t dispatch_thread_dword = PushData::NoStart;
	uint32_t memory_offset_dword = 0;
	// One dword per buffer binding with its descriptor's record
	// stride, so the stride is shader data instead of part of the pipeline specialization.
	uint32_t memory_stride_dword = PushData::NoStart;

	// The buffer bindings the stage has.
	[[nodiscard]] uint32_t BufferBindings() const {
		return descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)];
	}
	[[nodiscard]] uint32_t ShaderDataDwords() const {
		const auto buffers = BufferBindings();
		if (memory_stride_dword != PushData::NoStart) return memory_stride_dword + buffers;
		return memory_offset_dword + (buffers + 3u) / 4u;
	}
	[[nodiscard]] bool UsesPushData() const {
		return push_data_start_dword != PushData::NoStart;
	}
	// Whether a permutation made at push cursor `cursor` lays its push dwords as this one does.
	[[nodiscard]] bool PushStartMatches(uint32_t cursor) const {
		return push_data_start_dword == PushData::StartFor(cursor, ShaderDataDwords());
	}
	void AdvancePushData(uint32_t& cursor) const {
		if (PushData::IsSlotCursor(cursor) && (cursor & PushData::CursorFromEnd) != 0u) {
			// Stage slots: after the pixel stage the vertex stages fill from the draw's dwords up to its data.
			const uint32_t limit = UsesPushData() ? push_data_start_dword : PushData::CursorBound(cursor);
			cursor = PushData::LimitedCursor(PushData::CursorFront(cursor), limit);
			return;
		}
		const auto advance = [&](uint32_t next) {
			cursor = PushData::IsSlotCursor(cursor) ? PushData::LimitedCursor(next, PushData::CursorBound(cursor)) : next;
		};
		if (UsesPushData()) {
			advance(push_data_start_dword + ShaderDataDwords());
		}
	}

	bool operator==(const Bindings& other) const = default;
};

struct ShaderInfo {
	static constexpr uint32_t MaxBufferRoots  = 64;
	// Our ResourceTracking / emitter name (upstream renamed it MaxBufferRoots).
	static constexpr uint32_t MaxBuffers      = MaxBufferRoots;
	// Our ResourceTracking still bounds the sampled image/sampler pairs (upstream 33208a545 removed the limit with its
	// own tracking, not taken - SESJA-SYNC1).
	static constexpr uint32_t MaxSampledPairs = 64;
	static constexpr uint32_t MaxImages       = 64;
	static constexpr uint32_t MaxSamplers     = 32;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	std::vector<IndirectRangeResource> indirect_ranges;
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::vector<uint32_t>            user_data_registers;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                            has_bitwise_xor               = false;
	bool                            uses_dma                      = false;
	bool                            uses_swizzle                  = false;
	bool                            uses_lds                      = false;
	bool                            uses_gds                      = false;
	bool                            uses_flattened_srt            = false;
	// ReadConst only (uses_flattened_srt also covers the SRT mapping reads of GPU-selected descriptors).
	bool                            uses_read_const               = false;
	bool                            uses_image_heap               = false;
	bool                            uses_dispatch_threads         = false;
	bool                            bvh                           = false;
	bool                            bvh8                          = false;
	bool                            bvh_stack                     = false;
	bool                            helper_invocation             = false; // upstream 5429888d4: LiveMask
	// DS_ADD_RTN_U32 / DS_SUB_RTN_U32 whose old value is used: emitted lane-ordered (EmitAtomic32).
	bool                            ordered_lds_atomics           = false;
	bool                            subgroup_ballot               = false;
	bool                            subgroup_vote                 = false;
	bool                            subgroup_barrier              = false;
	bool                            subgroup_shuffle              = false;
	bool                            subgroup_arithmetic           = false;
	bool                            subgroup_local_invocation_id  = false;
	bool                            compute_derivatives           = false;
	bool                            image_gather_extended         = false;
	bool                            function_lds                  = false;
	bool                            function_scratch              = false;
	bool                            pixel_valid_mask              = false;
	bool                            buffer_int64_atomics          = false;
	bool                            buffer_u8                     = false;
	bool                            buffer_u16                    = false;
	bool                            shared_int64_atomics          = false;
	bool                            coherent_buffers              = false;
	// The shader stores to (or atomically updates) buffer or address memory, or one of its
	// cache-bypassing (GLC) buffer accesses is narrower than 32 bits.
	bool                            buffer_writes                 = false;
	bool                            coherent_narrow               = false;
	bool                            float64                       = false;

	bool operator==(const ShaderInfo& other) const = default;
};

struct DescriptorSource {
	struct IndirectDescriptor {
		struct SelectorRead {
			struct CountedRows {
				uint32_t source = UINT32_MAX;
				uint32_t stride = 0; // Byte spacing between selector rows, not their length.

				bool operator==(const CountedRows& other) const = default;
			};
			uint32_t source = UINT32_MAX;
			uint32_t stride = 0;
			uint32_t offset = 0;
			uint32_t maximum = UINT32_MAX;
			uint32_t bit_offset = 0;
			uint32_t bit_mask = UINT32_MAX;
			bool global = false;
			// Four DWORD bitmap words per record; set-bit ranks subtract from the last key.
			bool bitset = false;
			std::optional<CountedRows> rows;
			std::vector<std::pair<uint32_t, uint32_t>> byte_ranges;

			bool operator==(const SelectorRead& other) const = default;
		};
		struct PointerTable {
			uint32_t offset = 0;
			uint32_t stride = 0;
			uint32_t count = 1;
			bool indexed = false; // Parent pointers use raw MUBUF record addressing.

			bool operator==(const PointerTable& other) const = default;
		};
		std::optional<SelectorRead> selector;
		std::optional<uint32_t> table_maximum;
		std::optional<PointerTable> pointer_table;
		uint32_t table_source    = 0;
		uint32_t table_offset    = 0;
		uint32_t table_immediate = 0;
		uint32_t table_stride    = 0;
		uint32_t table_record_bytes = 0;
		bool     table_scalar = false;
		bool     indexed = false;
		bool     signed_count = false;
		uint32_t workgroup_axis  = UINT32_MAX;
		// The key is an arbitrary GPU value (e.g. a waterfall ReadLane or loop Phi): every
		// record inside the table buffer's bounds is a candidate.
		bool     all_records     = false;
		// Two-level table: each record holds a 64-bit pointer at
		// table_offset; the descriptor lies at that pointer + pointer_offset.
		uint32_t pointer_offset  = UINT32_MAX;
		// Record-keyed V#s: a V# that a bounded loop reads from the records of table_source (a V#) at
		// table_offset + table_immediate + key * table_stride for every key below key_count. The GPU keeps the V#
		// words (its accesses stay BDA); the host lists the guest ranges the candidates' accesses can touch.
		bool     record_keyed    = false;
		// record_keyed: the loop compares key < key_count unsigned (signed otherwise).
		bool     unsigned_bound  = false;
		// record_keyed: the candidates are a superset (all-records lists / _VALUE_BOUND), so a record
		// that is no usable V# makes the listing incomplete (unknown address writes for that call) instead of failing.
		bool     widened         = false;
		// record_keyed (game counts): the record count is the dword at count_field of the block a 64-bit
		// pointer at table + count_pointer points to (game data the shader does not read), at most count_limit; the
		// dword at count_check of the same block must hold the same count (a block the game reused meanwhile does not).
		uint32_t count_pointer   = UINT32_MAX;
		uint32_t count_field     = 0;
		uint32_t count_check     = 0;
		uint32_t count_limit     = 0;
		// record_keyed (composed V#s): `sources` holds the V# the shader composes from uniform values (not
		// a path proof's reads); counted per materialization.
		bool     composed        = false;
		Value    key_count;
		// First selector index, or first descriptor key for a direct or bitmap range.
		Value                 selector_first;
		Value    selector_mask;
		std::vector<uint32_t> sources;

		bool operator==(const IndirectDescriptor& other) const = default;
	};

	std::array<Value, 8>         dwords {};
	uint32_t                     dword_count = 0;
	std::optional<IndirectDescriptor> indirect_descriptor;

	bool operator==(const DescriptorSource& other) const = default;
};

struct SrtRead {
	Value    value;
	uint32_t flat_offset = 0;

	bool operator==(const SrtRead& other) const = default;
};

struct ResourceBlock {
	// Conditional successors are ordered true, false; an empty condition follows every edge.
	Value                 condition;
	std::vector<uint32_t> successors;
	std::vector<uint32_t> sources;
	std::vector<uint32_t> srt_reads;
};

// Stable shader metadata consumed by the renderer after native IR has been discarded.
struct CompiledShaderInfo {
	ShaderType                    stage               = ShaderType::Unknown;
	uint64_t                      shader_hash         = 0;
	uint32_t                      wave_size           = 64;
	uint32_t                      user_data_base      = 0;
	uint32_t                      user_data_count     = 64;
	uint32_t                      scratch_dwords      = 0;
	uint32_t                      param_export_mask   = 0;
	bool                          has_address_writes  = false;
	ShaderInfo                    info;
	Bindings                      bindings;
};

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
	Value                pattern_size;
	Value                element_count;
};

class SrtTape;

// Number of evaluation scratch slots of a ResourcePlan (and of an SrtTape). A thread that materializes
// resources uses the slot t_resource_scratch_slot names: 0 is the GPU thread (and tests), 1 the thread
// that materializes ahead of it (MaterializeResourcesAhead selects it). A slot must be used by one
// thread at a time; the plan itself is immutable while it is materialized.
inline constexpr uint32_t ResourceScratchSlots = 12;
inline thread_local uint32_t t_resource_scratch_slot = 0;
// The slot MaterializeResourcesAhead uses on this thread (1: the resolve thread).
inline thread_local uint32_t t_ahead_scratch_slot = 1;

// The resolve thread (slot 1) records which formatted buffers take or set the
// shared null-buffer specialization instead of reading and
// updating the shared state ahead of the GPU thread; an adoption applies it in stream order (ApplyNullFormattedShare).
// An adopted result must not carry a share from a later point of the stream (PLAN_STAGES=2: the
// specialization differences of [CS Cloth] dispatches and of replayed queued [Deferred] draws).
// Whether the calling thread's materialization records the null-buffer share instead of applying it (slots that run
// out of stream order: the resolve thread's).
inline bool RecordsNullShare() {
	return t_resource_scratch_slot >= 3u || t_resource_scratch_slot == 1u;
}

// Resource analysis retained by the shader cache. It owns immutable descriptor/SRT,
// condition and fill values without translated blocks, plus reusable evaluation scratch.
struct ResourcePlan {
	struct SelectorKeys {
		uint32_t              source = UINT32_MAX;
		std::vector<uint32_t> values;
		bool                  resolved = false;
	};

	struct EvaluationContext {
		struct Entry {
			uint64_t value      = 0;
			uint64_t generation = 0;
		};

		std::vector<Entry> values;
		uint64_t           generation = 0;
		const SrtWalker*    session = nullptr;
		Value              active_mask;
		uint8_t            read_policy = 0;
	};

	ResourcePlan();
	~ResourcePlan();

	ResourcePlan(const ResourcePlan&)            = delete;
	ResourcePlan& operator=(const ResourcePlan&) = delete;
	ResourcePlan(ResourcePlan&&) noexcept;
	ResourcePlan& operator=(ResourcePlan&& other) noexcept;

	ShaderType                    stage           = ShaderType::Unknown;
	uint64_t                      shader_hash     = 0;
	uint32_t                      user_data_base  = 0;
	uint32_t                      user_data_count = 64;
	std::list<Inst>                     value_storage;
	std::vector<MemoryInfo>             memory_info;
	std::vector<DescriptorSource>       descriptor_sources;
	std::vector<uint32_t>               selector_groups;
	std::vector<ResourceBlock>          control_flow;
	std::unordered_map<const Inst*, std::vector<uint32_t>> control_phi_predecessors;
	std::vector<SrtRead>                srt_reads;
	bool                                requires_specialization_memory = false;
	std::vector<std::pair<uint64_t, uint64_t>> source_reads;
	bool                                capture_specialization_reads = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	ShaderInfo                          info;
	UniformFillPlan                     uniform_fill;
	// Dense evaluator of the values above (null: SrtWalker evaluates them); see SrtTape.h.
	std::shared_ptr<SrtTape>            srt_tape;
	// Instruction numbering of the reference walker (SrtWalker): Inst::EvaluationIndex caches the number in
	// the instruction itself, so the reference walker runs on the GPU thread (slot 0) only.
	mutable uint32_t                       evaluation_value_count = 0;
	// Evaluation scratch: nested clean/EXEC memos, activity and material keys. One per thread slot
	// (t_resource_scratch_slot); everything a materialization writes into the plan lives here.
	struct Scratch {
		std::deque<EvaluationContext> evaluation_contexts;
		uint32_t                      evaluation_depth = 0;
		std::vector<uint8_t>          active_sources;
		std::vector<uint8_t>          visited_blocks;
		std::vector<uint32_t>         pending_blocks;
		std::vector<uint32_t>         material_keys;
		// KYTY_DIAG_NULL_FORMATTED_STICKY: last live {format, swizzle} of each formatted buffer. State kept
		// across calls, so materialization ahead of the GPU thread refuses while that flag is on.
		std::vector<std::array<uint32_t, 2>> sticky_buffer_formats;
		// the null formatted share on the resolve thread: per buffer, 1 = a null formatted V#
		// that takes the binding's last live specialization, 2 = a live one that sets it. The worker neither reads
		// nor writes the shared state (it is order-dependent); ApplyNullFormattedShare does it in order.
		std::vector<uint8_t>                 null_share_kinds;
	};
	mutable std::array<Scratch, ResourceScratchSlots> scratch;
	[[nodiscard]] Scratch& ThreadScratch() const { return scratch[t_resource_scratch_slot]; }
	// The null formatted share: last live {format, swizzle} of each formatted buffer binding, shared by every
	// thread that materializes this plan (execution and resolve threads); {0, 0} = no live V# seen yet.
	struct NullFormattedLive {
		std::mutex                           mutex;
		std::vector<std::array<uint32_t, 2>> live;
		// The null share journal: the journal's last live values ({0, 0}: none; empty: no record of the program),
		// and per binding whether this session has had a live V# of it (its first is checked against the journal).
		std::vector<std::array<uint32_t, 2>> journal;
		std::vector<uint8_t>                 journal_checked;
	};
	std::shared_ptr<NullFormattedLive> null_formatted_live = std::make_shared<NullFormattedLive>();
};

struct Program: ResourcePlan {
	Program() = default;
	~Program();

	Program(const Program&)            = delete;
	Program& operator=(const Program&) = delete;
	Program(Program&&) noexcept         = default;
	Program& operator=(Program&& other) noexcept;
	CompiledShaderInfo TakeCompiledInfo() &&;

	std::vector<std::unique_ptr<Block>> block_storage;
	BlockList                           blocks;
	uint32_t                      wave_size      = 64;
	uint32_t                      scratch_dwords = 0;
	bool                          dispatcher_fallback = false;
	// The pair rule took it (ShaderRecompiler: 64 threads and 2048 or more code words); the emitter pairs what the
	// read-lane pass planned.
	bool                          pair_by_rule        = false;
	CFG::FailureKind              cfg_failure_kind    = CFG::FailureKind::None;
	std::string                   fallback_reason;
	struct ScalarWrite { uint32_t pc; ScalarReg reg; Value value; };
	std::vector<ScalarWrite>      scalar_writes;
	// Typed memory and export instructions reference shader-local metadata by dense index.
	// Decoder-only details (such as NSA register numbers) have already become IR operands.
	std::vector<ExportInfo>       export_info;
	bool                          has_address_writes = false;
	bool                          shader_info_complete = false;
	Bindings                      bindings;

};

uint32_t StorageBufferElementBits(const Program& program, const MemoryInfo& memory);

std::string ProgramToString(const Program& program);
bool        HasShaderMemoryWrites(const Program& program);

void  ValidateProgram(const Program& program); // upstream 4919fdb6e: once, at SPIR-V emission
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
bool  EquivalentSelectorDomain(const ResourcePlan& program,
                               const DescriptorSource::IndirectDescriptor& left,
                               const DescriptorSource::IndirectDescriptor& right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);
Value ResolveActiveU32(Value value, Value active);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_ */
