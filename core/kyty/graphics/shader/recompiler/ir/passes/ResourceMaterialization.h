#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#if defined(ANYPS5_KYTY)
#include <optional>
#endif

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   scalar_empty                    = false;
		uint32_t               indirect_root                   = BufferResource::NoIndirectBuffer;
		uint32_t               indirect_mapping_offset         = 0;
		uint32_t               indirect_search_iterations      = 0;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		Prospero::BufferFormat        srgb_decode                = Prospero::BufferFormat::kInvalid;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		uint32_t                      table                      = ImageResource::NoImageTable;
		uint32_t                      table_capacity             = 0;
		uint32_t                      table_mapping_offset       = 0;
		uint32_t                      table_entry_mask           = 0;
		bool                          cube                       = false;
		bool                          skip_degamma               = false;
		bool                          fmask                      = false;
		bool                          heap                       = false;
		bool                          operator==(const Image&) const = default;
	};

	struct Sampler {
		uint32_t indirect_root = SamplerResource::NoIndirectSampler;
		uint32_t indirect_mapping_offset = 0;
		uint32_t indirect_search_iterations = 0;
		bool operator==(const Sampler&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;
	std::vector<Sampler> samplers;

	bool operator==(const ResourceSpecialization&) const = default;
};

ResourcePlan ExtractResourcePlan(const Program& program);

struct EncodeFailure {
	enum class Reason : uint8_t { None, ValueType, UnknownInstruction, PhiBlock };

	Reason      reason         = Reason::None;
	uint32_t    type           = 0;
	bool        in_instruction = false;
	ValueOpcode opcode         = ValueOpcode::Void;

	[[nodiscard]] bool Failed() const { return reason != Reason::None; }
};

bool SaveResourcePlan(const ResourcePlan& plan, std::vector<uint8_t>& bytes, EncodeFailure& failure);
bool DecodeResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan);
bool LoadResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan);

void SetComputeIndirectFloors(size_t keys, size_t images);

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

bool ReadsStrideAtRunTime(const ResourcePlan& program, uint32_t index,
                          const ResourceSpecialization::Buffer& buffer);
void ReadStridesAtRunTime(const ResourcePlan& program, ResourceSpecialization& specialization);
#if defined(ANYPS5_KYTY)
bool SharesEmptyBuffers();
void ShareNullBufferVariants(const ResourcePlan& program, ResourceSpecialization& specialization,
                             bool runtime_stride);
uint32_t FormattedReadComponents(const ResourcePlan& program, uint32_t index);
std::optional<uint32_t> EmptyBufferReadBits(const ResourceSpecialization::Buffer& buffer, uint32_t component);
bool SpecializedBuffersValid(const ResourceSpecialization& specialization, bool compute);
bool EmptyBufferServes(const ResourcePlan& program, const ResourceSnapshot& snapshot, uint32_t index,
                       const ResourceSpecialization::Buffer& requested,
                       const ResourceSpecialization::Buffer& candidate);
bool EmptyBuffersServe(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                       const ResourceSpecialization& requested, const ResourceSpecialization& candidate);
bool SameModuleCode(const ResourcePlan& program, const ResourceSpecialization& requested,
                    const ResourceSpecialization& candidate);
bool GenericComputeVariants();
void SetMirrorSize(uint64_t size);
bool MirrorsVsharpBuffers();
uint32_t MirrorElementBytes(const ResourcePlan& program, uint32_t index);
void MirrorBuffers(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                   ResourceSpecialization& specialization);
struct FormedView {
	uint64_t base   = 0;
	uint64_t bytes  = 0;
	uint32_t stride = 0;
	bool     none   = false;
	bool     serves = false;
};
bool       FormsVsharpBuffers();
void       BlockFormedViews(bool blocked);
bool       FormedViewsBlocked();
uint64_t   MirrorSize();
uint32_t   FormedElementBytes(const ResourcePlan& program, uint32_t index);
FormedView FormView(const BufferResource& buffer, uint32_t element, std::span<const uint32_t, 4> words,
                    uint64_t limit);
void       FormBuffers(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                       ResourceSpecialization& specialization, uint64_t formed_now);
bool AlignResources(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                    const ResourceSpecialization& specialization, const ResourceSpecialization& target,
                    ResourceSnapshot* aligned);
bool WidenSpecialization(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                         const ResourceSpecialization&                       specialization,
                         std::span<const ResourceSpecialization* const>      known,
                         ResourceSpecialization&                             widened);
#endif
bool SpecializeDirectResources(const ResourcePlan& program, ResourceSnapshot& snapshot,
                               ResourceSpecialization& specialization);

bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor);

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

bool ResourceSpecializationFits(const Program& program, const ResourceSpecialization& specialization);

uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value);
ImageResource ImageTableHeapProfile(const ImageResource& root, bool skip_degamma);
uint32_t ImageTableHeapKind(const ImageResource& profile, DescriptorValue& value);
bool ImageTableHeapExemplar(const ImageResource& profile, const DescriptorValue& value);
ImageResource ImageTableHeapResource(const ImageResource& profile, const DescriptorValue& value);
ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind);

}

#endif
