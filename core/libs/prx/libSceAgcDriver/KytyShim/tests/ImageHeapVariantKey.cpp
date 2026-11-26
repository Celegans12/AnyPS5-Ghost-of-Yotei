#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler;

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "kyty_shim_image_heap_variant_key: %s\n", message);
    std::exit(1);
}

IR::DescriptorValue MakeTextureDescriptor(Prospero::BufferFormat format) {
    IR::DescriptorValue value {.dword_count = 8u};
    value.dwords[0] = 0x1000u;
    value.dwords[1] = (static_cast<uint32_t>(format) << 20u) | (3u << 30u);
    value.dwords[2] = 15u;
    value.dwords[3] = 0xfacu | (9u << 28u);
    return value;
}

class MockImageHeapSource final : public IR::ImageHeapSource {
public:
    explicit MockImageHeapSource(IR::DescriptorValue exemplar) : m_exemplar(exemplar) {}

    bool Acquire(const IR::ImageHeapRequest& , IR::ImageHeapView& view) override {
        view.slots      = 0x200000000ull;
        view.slot_count = 4u;
        view.kinds      = std::span<const IR::DescriptorValue>(&m_exemplar, 1u);
        return true;
    }

private:
    IR::DescriptorValue m_exemplar;
};

IR::DescriptorSource MakeImmediateSource(std::span<const uint32_t> words) {
    IR::DescriptorSource source;
    source.dword_count = static_cast<uint32_t>(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        source.dwords[i] = IR::Value(words[i]);
    }
    return source;
}

IR::ResourcePlan MakeHeapPlan() {
    IR::ResourcePlan plan;
    plan.stage                      = Libs::Graphics::ShaderType::Compute;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete          = true;

    const std::array<uint32_t, 2> table_pointer {0x00000000u, 0x00000002u};
    const std::array<uint32_t, 4> sampler_words {0x80000012u, 0u, 0u, 0u};

    IR::DescriptorSource indirect;
    indirect.dword_count = 8u;
    indirect.indirect_descriptor = IR::DescriptorSource::IndirectDescriptor {
        .table_maximum = 3u,
        .table_source  = 0u,
        .table_offset  = 0u,
        .table_stride  = 32u,
        .table_scalar  = true,
    };

    plan.descriptor_sources.push_back(MakeImmediateSource(table_pointer));
    plan.descriptor_sources.push_back(std::move(indirect));
    plan.descriptor_sources.push_back(MakeImmediateSource(sampler_words));

    IR::ImageResource image;
    image.resource_class = IR::ImageResourceClass::Sampled;
    image.read           = true;
    image.dimension      = Decoder::ImageDimension::Dim2D;
    image.source         = 1u;
    plan.info.images.push_back(image);

    IR::SamplerResource sampler;
    sampler.source = 2u;
    plan.info.samplers.push_back(sampler);
    plan.info.sampled_pairs.push_back({.image = 0u, .sampler = 0u});
    IR::CompileSrtWalker(plan);
    return plan;
}

IR::ResourcePlan MakeDirectPlan(Prospero::BufferFormat format) {
    IR::ResourcePlan plan;
    plan.stage                      = Libs::Graphics::ShaderType::Compute;
    plan.resource_tracking_complete = true;
    plan.srt_plan_complete          = true;

    const auto descriptor = MakeTextureDescriptor(format);
    const std::array<uint32_t, 4> sampler_words {0x80000012u, 0u, 0u, 0u};
    plan.descriptor_sources.push_back(MakeImmediateSource(descriptor.dwords));
    plan.descriptor_sources.push_back(MakeImmediateSource(sampler_words));

    IR::ImageResource image;
    image.resource_class = IR::ImageResourceClass::Sampled;
    image.read           = true;
    image.dimension      = Decoder::ImageDimension::Dim2D;
    image.source         = 0u;
    plan.info.images.push_back(image);

    IR::SamplerResource sampler;
    sampler.source = 1u;
    plan.info.samplers.push_back(sampler);
    plan.info.sampled_pairs.push_back({.image = 0u, .sampler = 0u});
    IR::CompileSrtWalker(plan);
    return plan;
}

}

int main() {
    const auto heap_plan = MakeHeapPlan();
    const auto srgb_t    = MakeTextureDescriptor(Prospero::BufferFormat::k8Srgb);
    const auto unorm_t   = MakeTextureDescriptor(Prospero::BufferFormat::k8UNorm);

    MockImageHeapSource        heap_srgb(srgb_t);
    MockImageHeapSource        heap_unorm(unorm_t);
    IR::SrtRuntime             runtime_srgb {.image_heap = &heap_srgb};
    IR::SrtRuntime             runtime_unorm {.image_heap = &heap_unorm};
    IR::ResourceSnapshot       snap_srgb, snap_unorm;
    IR::ResourceSpecialization spec_srgb, spec_unorm;

    Require(IR::MaterializeResources(heap_plan, runtime_srgb, snap_srgb, spec_srgb),
            "sRGB heap exemplar failed materialization");
    Require(IR::MaterializeResources(heap_plan, runtime_unorm, snap_unorm, spec_unorm),
            "UNorm heap exemplar failed materialization");
    Require(spec_srgb.images.size() == 2u && spec_unorm.images.size() == 2u,
            "unexpected heap specialization image count");
    Require(spec_srgb == spec_unorm,
            "image heap specialization differed between sRGB and UNorm exemplars of the same kind");
    Require(!spec_srgb.images[1].skip_degamma,
            "image heap child must not record exemplar skip_degamma");

    const auto direct_srgb_plan  = MakeDirectPlan(Prospero::BufferFormat::k8Srgb);
    const auto direct_unorm_plan = MakeDirectPlan(Prospero::BufferFormat::k8UNorm);
    IR::SrtRuntime             direct_runtime {};
    IR::ResourceSnapshot       direct_snap_srgb, direct_snap_unorm;
    IR::ResourceSpecialization direct_spec_srgb, direct_spec_unorm;

    Require(IR::MaterializeResources(direct_srgb_plan, direct_runtime, direct_snap_srgb, direct_spec_srgb),
            "direct sRGB image failed materialization");
    Require(IR::MaterializeResources(direct_unorm_plan, direct_runtime, direct_snap_unorm, direct_spec_unorm),
            "direct UNorm image failed materialization");
    Require(direct_spec_srgb.images[0].skip_degamma && !direct_spec_unorm.images[0].skip_degamma,
            "direct sampled image must preserve skip_degamma");

    std::puts("kyty_shim_image_heap_variant_key: ok");
    return 0;
}
