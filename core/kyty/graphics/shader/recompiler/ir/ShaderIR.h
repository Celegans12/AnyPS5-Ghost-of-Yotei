#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERIR_H_

#include "common/common.h"
#include "graphics/shader/recompiler/ir/BdaReadPlan.h"
#include "common/stringUtils.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bit>
#include <list>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Xbyak {
class CodeGenerator;
}

namespace Libs::Graphics::ShaderRecompiler::IR {

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
	bool                    image_cube                                            = false;
	bool                    idxen                                                 = false;
	bool                    offen                                                 = false;
	bool                    coherent                                              = false;
	bool                    planning_only                                         = false;

	[[nodiscard]] bool IsCoherentBufferPairLoad(ValueOpcode opcode) const {
		return kind == ResourceKind::Buffer && opcode == ValueOpcode::LoadBufferU32x2 && coherent &&
		       !typed && !formatted && data_bits == 32u;
	}

	[[nodiscard]] bool IsCoherentBufferPairStore(ValueOpcode opcode) const {
		return kind == ResourceKind::Buffer && opcode == ValueOpcode::StoreBufferU32x2 && coherent &&
		       !typed && !formatted && data_bits == 32u;
	}

	[[nodiscard]] bool SupportsIndirectBufferLoad(ValueOpcode opcode) const {
		if (typed) return false;
		if (data_bits == 8u) return !formatted && opcode == ValueOpcode::LoadBufferU8;
		if (data_bits == 16u) return !formatted && opcode == ValueOpcode::LoadBufferU16;
		return data_bits == 32u &&
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
	[[nodiscard]] bool SupportsIndirectBufferStore(ValueOpcode opcode) const {
		return !formatted && !typed && data_bits == 32u &&
		       (opcode == ValueOpcode::StoreBufferU32 || opcode == ValueOpcode::StoreBufferU32x2 ||
		        opcode == ValueOpcode::StoreBufferU32x3 || opcode == ValueOpcode::StoreBufferU32x4);
	}
	[[nodiscard]] bool SupportsIndirectBufferAtomic(ValueOpcode opcode) const {
		if (formatted || typed || data_bits != 32u) return false;
		switch (opcode) {
			case ValueOpcode::BufferAtomicSwap32:
			case ValueOpcode::BufferAtomicCmpSwap32:
			case ValueOpcode::BufferAtomicIAdd32:
			case ValueOpcode::BufferAtomicISub32:
			case ValueOpcode::BufferAtomicSMin32:
			case ValueOpcode::BufferAtomicUMin32:
			case ValueOpcode::BufferAtomicSMax32:
			case ValueOpcode::BufferAtomicUMax32:
			case ValueOpcode::BufferAtomicAnd32:
			case ValueOpcode::BufferAtomicOr32:
			case ValueOpcode::BufferAtomicXor32: return true;
			default: return false;
		}
	}
	[[nodiscard]] bool SupportsIndirectBufferWrite(ValueOpcode opcode) const {
		return SupportsIndirectBufferStore(opcode) || SupportsIndirectBufferAtomic(opcode);
	}
	[[nodiscard]] bool SupportsIndirectBufferAccess(ValueOpcode opcode) const {
		return SupportsIndirectBufferLoad(opcode) || SupportsIndirectBufferWrite(opcode);
	}

	bool operator==(const MemoryInfo& other) const = default;
};

enum class ExportTargetKind { Unknown, Null, Position, Primitive, Parameter, Mrt, MrtZ };

struct ExportInfo {
	ExportTargetKind kind  = ExportTargetKind::Unknown;
	uint32_t         index = 0;
	uint32_t         en    = 0;
	bool             compr = false;
	bool             vm    = false;

	bool operator==(const ExportInfo& other) const = default;
};

inline constexpr uint32_t BufferStrideMask    = 0x3fffu;
inline constexpr uint32_t RuntimeBufferStride = BufferStrideMask;
[[nodiscard]] constexpr bool HasRuntimeStride(uint32_t packed_stride) {
	return (packed_stride & BufferStrideMask) == RuntimeBufferStride;
}

struct BufferWriteSpan {
	uint32_t offset         = 0;
	bool     per_invocation = false;

	auto operator<=>(const BufferWriteSpan&) const = default;
};

struct BufferResource {
	static constexpr uint32_t NoIndirectBuffer = UINT32_MAX;

	uint32_t               source             = 0;
	uint32_t               descriptor_index   = UINT32_MAX;
	uint32_t               uniform_index      = UINT32_MAX;
	uint32_t               first_use_pc       = 0;
	uint32_t               max_byte_extent    = 0;
	uint32_t               packed_stride      = 0;
	Prospero::BufferFormat descriptor_format  = Prospero::BufferFormat::kInvalid;
	uint32_t               descriptor_swizzle = DstSel(4, 5, 6, 7);
	bool                   read               = false;
	bool                   written            = false;
	bool                   atomic             = false;
	bool                   formatted          = false;
	bool                   scalar             = false;
	bool                   live               = false;
	bool                   constant_only      = false;
	uint32_t               indirect_root              = NoIndirectBuffer;
	uint32_t               indirect_mapping_offset    = 0;
	uint32_t               indirect_search_iterations = 0;
	std::vector<uint32_t>  indirect_resources;
	bool                   indirect_write_table  = false;
	uint32_t               indirect_table_offset = 0;
	std::vector<BufferWriteSpan> proven_writes;

	bool operator==(const BufferResource& other) const = default;
};

enum class ImageMipMode { None, Dynamic };

constexpr uint32_t ShaderImageIdentitySwizzle = 0x00000facu;

struct ImageResource {
	static constexpr uint32_t NoIndirectImage = UINT32_MAX;
	static constexpr uint32_t NoImageTable    = UINT32_MAX;

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
	Prospero::BufferFormat        srgb_decode       = Prospero::BufferFormat::kInvalid;
	bool                          read              = false;
	bool                          written           = false;
	bool                          atomic            = false;
	bool                          atomic64          = false;
	bool                          unfiltered_access_only = true;
	bool                          sample_access_only     = true;
	bool                          depth_compare     = false;
	bool                          skip_degamma      = false;
	bool                          cube              = false;
	bool                          r128              = false;
	bool                          heap              = false;
	uint32_t                      indirect_root     = NoIndirectImage;
	uint32_t                      indirect_mapping_offset   = 0;
	uint32_t                      indirect_search_iterations = 0;
	std::vector<uint32_t>         indirect_resources;
	uint32_t                      table                = NoImageTable;
	uint32_t                      table_capacity       = 0;
	uint32_t                      table_mapping_offset = 0;
	uint32_t                      table_entry_mask     = 0;
	uint32_t                      table_descriptor_base = UINT32_MAX;

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
	if (format == Prospero::BufferFormat::k16_16SInt && !image.read && !image.atomic)
		return Prospero::BufferFormat::k16_16UInt;
	return Prospero::BufferFormat::kInvalid;
}

[[nodiscard]] constexpr bool ReadsDepthBits(const ImageResource& image) {
	return image.conversion_format == Prospero::BufferFormat::k16UInt;
}

struct SamplerResource {
	static constexpr uint32_t NoIndirectSampler = UINT32_MAX;

	uint32_t source                = 0;
	uint32_t first_use_pc          = 0;
	uint32_t snapshot_index        = 0;
	uint32_t indirect_root = NoIndirectSampler;
	uint32_t indirect_mapping_offset = 0;
	uint32_t indirect_search_iterations = 0;
	std::vector<uint32_t> indirect_resources;
	bool     force_point_filtering = false;
	bool     depth_compare         = false;
	bool     integer_border        = false;
	bool     gather_lod            = false;

	bool operator==(const SamplerResource& other) const = default;
};

struct SampledResourcePair {
	uint32_t image        = 0;
	uint32_t sampler      = 0;
	uint32_t first_use_pc = 0;

	bool operator==(const SampledResourcePair& other) const = default;
};

enum class TessellationAttribute {
	LocalOutput,
	ControlInput,
	ControlOutput,
	EvaluationInput,
	PatchOutput,
	PatchInput,
#if defined(ANYPS5_KYTY)
	Factor,
	ControlShared
};
constexpr uint32_t NumTessellationAttributes =
    static_cast<uint32_t>(TessellationAttribute::ControlShared) + 1u;
constexpr uint32_t MergedHullRecordBytes = 16u;
#else
	Factor
};
constexpr uint32_t NumTessellationAttributes =
    static_cast<uint32_t>(TessellationAttribute::Factor) + 1u;
#endif

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
	BaseVertex,
	BaseInstance,
	Parameter,
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
	ViewportIndex
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
	UniformBuffers,
	MipStats,
#if defined(ANYPS5_KYTY)
	PackedTargets,
#endif
	Count,
};

inline constexpr uint32_t UniformBufferBytes = 65536u;
inline constexpr uint32_t FlattenedSrtUniformWords = 4096u;
inline constexpr uint32_t UniformBufferBaseSlack = 256u;

static_assert(static_cast<uint32_t>(DescriptorBindingKind::Samplers) == 51u);
#if defined(ANYPS5_KYTY)
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 61u);
#else
static_assert(static_cast<uint32_t>(DescriptorBindingKind::Count) == 60u);
#endif

struct PushData {
	static constexpr uint32_t DwordCount = 32;
	static constexpr uint32_t MeshDrawDwordCount = 6;
	static constexpr uint32_t MeshTaskPayloadDwordCount = 6;
	static constexpr uint32_t NoStart    = UINT32_MAX;
	std::array<uint32_t, DwordCount> dwords {};

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
	[[nodiscard]] static constexpr uint32_t StartFor(uint32_t cursor, uint32_t size) {
		return CanFit(cursor, size) ? cursor : NoStart;
	}
};

static_assert(sizeof(PushData) == 128);
constexpr uint32_t NativePushConstantSize = sizeof(PushData);

[[nodiscard]] constexpr uint32_t NativeBinding(ShaderType stage, DescriptorBindingKind kind) {
	const uint32_t group = stage == ShaderType::Pixel                    ? 1u
	                       : stage == ShaderType::TessellationControl    ? 2u
	                       : stage == ShaderType::TessellationEvaluation ? 3u
	                                                                     : 0u;
	return static_cast<uint32_t>(kind) +
	       group * static_cast<uint32_t>(DescriptorBindingKind::Count);
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

[[nodiscard]] inline std::vector<uint32_t> AllocateImageTableDescriptors(
    const std::vector<ImageResource>& images,
    std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)>& counts) {
	std::vector<uint32_t> bases(images.size(), UINT32_MAX);
	for (size_t i = 0; i < images.size(); i++) {
		const auto& image = images[i];
		if (image.table_capacity == 0u) {
			continue;
		}
		const auto kind = *DescriptorBindingForImage(image);
		for (size_t j = 0; j < i && bases[i] == UINT32_MAX; j++) {
			if (bases[j] != UINT32_MAX && images[j].table == image.table &&
			    DescriptorBindingForImage(images[j]) == kind) {
				if (images[j].table_capacity != image.table_capacity) {
					EXIT("shader binding layout failed: image table %u has two capacities",
					     image.table);
				}
				bases[i] = bases[j];
			}
		}
		if (bases[i] == UINT32_MAX) {
			auto& count = counts[static_cast<size_t>(kind)];
			bases[i]    = count;
			count += image.table_capacity;
		}
	}
	return bases;
}

struct Bindings {
	std::array<uint32_t, static_cast<size_t>(DescriptorBindingKind::Count)> descriptor_counts {};
	uint64_t descriptor_mask = 0;
	uint32_t push_data_start_dword = PushData::NoStart;
	uint32_t dispatch_thread_dword = PushData::NoStart;
	uint32_t ray_query_dword       = PushData::NoStart;
	uint32_t memory_offset_dword = 0;
	uint32_t table_block_dwords  = 0;
	uint32_t mip_stats_count = 0;
	uint32_t stride_count = 0;
	bool     flattened_srt_uniform = false;
	uint32_t flattened_srt_uniform_index = UINT32_MAX;

	[[nodiscard]] uint32_t MemorySizeDword() const {
		return memory_offset_dword +
		       (descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)] + 3u) / 4u;
	}
	[[nodiscard]] uint32_t MipStatsDword() const {
		return MemorySizeDword() +
		       descriptor_counts[static_cast<size_t>(DescriptorBindingKind::Buffers)];
	}
	[[nodiscard]] uint32_t StrideDword() const { return MipStatsDword() + mip_stats_count; }
	[[nodiscard]] uint32_t TableBlockDword() const { return StrideDword() + stride_count; }
	[[nodiscard]] uint32_t ShaderDataDwords() const { return TableBlockDword() + table_block_dwords; }
	[[nodiscard]] bool UsesPushData() const {
		return push_data_start_dword != PushData::NoStart;
	}
	void AdvancePushData(uint32_t& cursor) const {
		if (UsesPushData()) {
			cursor = push_data_start_dword + ShaderDataDwords();
		}
	}

	bool operator==(const Bindings& other) const = default;
};

struct ShaderInfo {
	static constexpr uint32_t NoRayQuerySlot  = 0xffffffffu;
	static constexpr uint32_t UserDataRayQueryHeader = 0xffffffffu;
	static constexpr uint32_t UserDataRayQueryVsharp = 0xfffffffeu;
	static constexpr uint32_t MaxBufferRoots  = 64;
	static constexpr uint32_t MaxImages       = 256;
	static constexpr uint32_t MaxSamplers     = 256;

	std::vector<BufferResource>      buffers;
	std::vector<ImageResource>       images;
	std::vector<SamplerResource>     samplers;
	std::vector<SampledResourcePair> sampled_pairs;
	std::vector<std::array<uint32_t, 2>> address_bases;
	std::vector<std::array<uint32_t, 2>> user_address_bases;
	std::array<uint32_t, 2>          ray_query_tlas {};
	bool                             ray_query_mirror = false;
	std::vector<StageInput>          inputs;
	std::vector<StageOutput>         outputs;
	std::vector<uint32_t>            user_data_registers;
	std::vector<std::array<uint32_t, 2>> dma_base_registers;
	std::vector<std::array<uint32_t, 2>> vsharp_base_registers;
	struct DmaWindow {
		uint32_t base_register = 0;
		int64_t  first         = 0;
		uint64_t last          = 0;

		bool operator==(const DmaWindow& other) const = default;
	};
	struct DmaDescriptorTable {
		uint32_t base_register = 0;
		int64_t  immediate     = 0;
		uint32_t offset_bits   = 0;

		bool operator==(const DmaDescriptorTable& other) const = default;
	};
	struct DmaRecordTable {
		uint32_t buffer = 0;
		uint32_t stride = 0;
		uint32_t offset = 0;

		bool operator==(const DmaRecordTable& other) const = default;
	};
	std::vector<DmaWindow>           dma_windows;
	std::vector<DmaDescriptorTable>  dma_tables;
	std::vector<DmaRecordTable>      dma_record_tables;
	struct SelfSizedBuffer {
		uint32_t base_low_register  = 0;
		uint32_t base_high_register = 0;
		bool     written            = false;

		bool operator==(const SelfSizedBuffer& other) const = default;
	};
	std::vector<SelfSizedBuffer>     self_sized_buffers;
	std::vector<uint32_t>            dma_record_reach;
	std::vector<uint32_t>            ray_query_reach;
	static constexpr uint32_t        RayQuerySrtVsharp = 0x80000000u;
	std::array<uint8_t, 32>          vertex_fetch_components {};
	int32_t                          vertex_offset_sgpr = -1;
	int32_t                          instance_offset_sgpr = -1;
	bool                            has_bitwise_xor               = false;
	bool                            uses_dma                      = false;
	bool                            dma_bounded                   = false;
	bool                            uses_swizzle                  = false;
	bool                            relative_vgprs                = false;
	bool                            uses_lds                      = false;
	bool                            uses_gds                      = false;
	bool                            uses_flattened_srt            = false;
	bool                            uses_image_heap               = false;
	bool                            uses_device_address           = false;
	bool                            uses_dispatch_threads         = false;
	bool                            bvh                           = false;
	bool                            bvh8                          = false;
	bool                            bvh_stack                     = false;
	bool                            ray_query                     = false;
	uint32_t                        ray_query_sites               = 0;
	uint32_t                        ray_query_user_sgpr           = 0;
	uint32_t                        ray_query_header_offset       = 0;
	uint32_t                        ray_query_header_vsharp       = NoRayQuerySlot;
	uint32_t                        ray_query_root                = 0;
	std::array<uint32_t, 4>         ray_query_layout {};
	std::array<uint32_t, 2>         ray_query_header_slots {NoRayQuerySlot, NoRayQuerySlot};
	bool                            subgroup_ballot               = false;
	bool                            subgroup_barrier              = false;
	bool                            subgroup_shuffle              = false;
	bool                            subgroup_arithmetic           = false;
	bool                            subgroup_local_invocation_id  = false;
	bool                            helper_invocation             = false;
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
	bool                            float64                       = false;
	bool                            float16                       = false;

	bool operator==(const ShaderInfo& other) const = default;
};

struct DescriptorSource {
	struct IndirectDescriptor {
		struct SelectorRead {
			struct CountedRows {
				uint32_t source = UINT32_MAX;
				uint32_t stride = 0;

				bool operator==(const CountedRows& other) const = default;
			};
			struct MaskRanks {
				uint32_t source = UINT32_MAX;
				uint32_t stride = 0;
				uint32_t offset = 0;

				bool operator==(const MaskRanks& other) const = default;
			};
			uint32_t source = UINT32_MAX;
			uint32_t stride = 0;
			uint32_t offset = 0;
			uint32_t first_record = 0;
			uint32_t record_step = 1;
			uint32_t maximum = UINT32_MAX;
			bool non_negative = false;
			uint32_t bit_offset = 0;
			uint32_t bit_mask = UINT32_MAX;
			bool global = false;
			bool bitset = false;
			std::optional<CountedRows> rows;
			std::vector<std::pair<uint32_t, uint32_t>> byte_ranges;
			std::optional<uint32_t> excluded;
			std::optional<MaskRanks> ranks;

			bool operator==(const SelectorRead& other) const = default;
		};
		struct PointerTable {
			uint32_t offset = 0;
			uint32_t stride = 0;
			uint32_t count = 1;
			bool indexed = false;
			bool     buffer         = false;
			uint32_t element_stride = 0;
			uint32_t element_offset = 0;
			uint64_t function       = 0;
			std::vector<uint64_t> functions;

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
		uint32_t workgroup_multiplier = 0;
		uint32_t workgroup_shift      = 0;
		uint32_t table_mask      = UINT32_MAX;
		bool     table_array     = false;
		Value    key_count;
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
	Value                 condition;
	std::vector<uint32_t> successors;
	std::vector<uint32_t> sources;
	std::vector<uint32_t> images;
	std::vector<uint32_t> srt_reads;
};

struct TablePlan {
	struct Operand {
		enum class Kind : uint8_t { Immediate, UserData, Slot, ShaderBaseLow, ShaderBaseHigh };
		Kind     kind  = Kind::Immediate;
		uint32_t value = 0;
		bool     operator==(const Operand&) const = default;
	};
	struct Slot {
		Operand low, high, offset;
		int32_t immediate = 0;
		Operand records;
		bool    bounded = false;
		bool    operator==(const Slot&) const = default;
	};
	struct Descriptor {
		std::array<Operand, 8> words {};
		uint32_t               count = 0;
		bool                   operator==(const Descriptor&) const = default;
	};
	std::vector<Slot>                   slots;
	std::vector<std::array<Operand, 4>> buffers;
	std::vector<Descriptor>             images, samplers;
	std::vector<uint8_t>                gpu, cpu;
	bool global_memory = false;
	uint32_t runtime_formats = 0;

	static constexpr uint32_t BufferDwords = 5;
	[[nodiscard]] uint32_t BufferDword(size_t buffer) const {
		return static_cast<uint32_t>(slots.size() + BufferDwords * buffer);
	}
	[[nodiscard]] uint32_t BlockDwords() const { return BufferDword(buffers.size()); }
	bool                   operator==(const TablePlan&) const = default;
};

struct CompiledShaderInfo {
	BdaReadPlan bda_read_plan;
	ShaderType stage              = ShaderType::Unknown;
	uint64_t   shader_hash        = 0;
	uint32_t   wave_size          = 64;
	uint32_t   user_data_base     = 0;
	uint32_t   param_export_mask  = 0;
	bool       has_address_writes = false;
	bool       table_mode         = false;
	TablePlan  table_plan;
	ShaderInfo info;
	Bindings   bindings;
};

struct UniformFillPlan {
	UniformFill          fill;
	std::array<Value, 4> values;
	Value                pattern_size;
	Value                element_count;
};

enum class DescriptorRefresh : uint8_t { None, Buffer, Image };

constexpr uint32_t DescriptorRefreshMask(DescriptorRefresh kind, uint32_t word) {
	switch (kind) {
		case DescriptorRefresh::Buffer: return word == 0 ? 0xffffffffu : (word == 1 ? 0x0000ffffu : 0u);
		case DescriptorRefresh::Image: return word == 5 ? 0x027fff00u : (word == 6 ? 0x000000ffu : 0u);
		default: return 0u;
	}
}

struct ResourcePlan {
	struct SelectorKeys {
		std::vector<uint32_t> values;
		bool                  resolved = false;
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
	std::vector<std::pair<uint64_t, uint64_t>> source_reads;
	bool                                capture_specialization_reads = false;
	bool                                srt_plan_complete          = false;
	bool                                resource_tracking_complete = false;
	ShaderInfo                          info;
	std::vector<uint8_t>                buffer_vector_access;
	std::vector<uint8_t>                buffer_feeds_descriptors;
	UniformFillPlan                     uniform_fill;
	std::vector<DescriptorRefresh>      descriptor_refresh;
	std::vector<uint32_t>               descriptor_word_flat;
	std::unique_ptr<Xbyak::CodeGenerator> srt_code;
	uint32_t                              srt_user_data = 0;
	uint32_t                              srt_captures  = 0;
	std::unique_ptr<Xbyak::CodeGenerator> srt_long_code;
	uint32_t                              srt_long_captures = 0;
	mutable std::vector<uint64_t>           srt_scratch;
	mutable std::vector<uint32_t>           material_keys;
	mutable std::vector<SelectorKeys>       selector_keys;
	struct ThreadScratch {
		std::vector<uint64_t>     srt_scratch;
		std::vector<uint32_t>     material_keys;
		std::vector<SelectorKeys> selector_keys;
	};
#if defined(ANYPS5_KYTY)
	static constexpr uint32_t                             FirstChainScratchSlot = 3;
	static constexpr uint32_t                             ChainScratchSlots     = 16;
	static constexpr uint32_t ThreadScratchSlots = FirstChainScratchSlot + ChainScratchSlots;
#else
	static constexpr uint32_t                             ThreadScratchSlots = 2;
#endif
	mutable std::array<std::unique_ptr<ThreadScratch>, ThreadScratchSlots> thread_scratch;
};

struct PlanScratchRef {
	std::vector<uint64_t>&                   srt_scratch;
	std::vector<uint32_t>&                   material_keys;
	std::vector<ResourcePlan::SelectorKeys>& selector_keys;
};
PlanScratchRef PlanScratch(const ResourcePlan& plan);
void UsePlanScratchSlot(uint32_t slot);

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
	bool                          lane_local     = false;
	uint32_t                      scratch_dwords = 0;
	uint32_t                      relative_vgpr_dwords = 0;
	bool                          dispatcher_fallback = false;
	std::vector<uint32_t>         inlined_call_entries;
	struct ScalarWrite { uint32_t pc; ScalarReg reg; Value value; };
	std::vector<ScalarWrite>      scalar_writes;
	std::vector<ExportInfo>       export_info;
	bool                          has_address_writes = false;
	bool                          shader_info_complete = false;
	Bindings                      bindings;
	bool                          table_mode = false;
	TablePlan                     table_plan;

};

uint32_t StorageBufferElementBits(const Program& program, const MemoryInfo& memory);

std::string ProgramToString(const Program& program);
bool        HasShaderMemoryWrites(const Program& program);

void  ValidateProgram(const Program& program);
bool  EquivalentValue(const ResourcePlan& program, Value left, Value right);
bool  EquivalentSelectorDomain(const ResourcePlan& program,
                               const DescriptorSource::IndirectDescriptor& left,
                               const DescriptorSource::IndirectDescriptor& right);
Value ResolveInvariantPhi(const ResourcePlan& program, Value value);
Value ResolveActiveU32(Value value, Value active);

}

#endif
