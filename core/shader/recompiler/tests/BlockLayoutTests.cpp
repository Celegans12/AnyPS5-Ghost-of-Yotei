#include "Optimization/BlockLayout.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace ShaderRecompiler;

namespace {

constexpr std::uint32_t Missing = 0xffffffffu;
constexpr std::uint32_t WholeWord = 0xffffffffu;

int failures = 0;

void Expect(bool condition, const char* text) {
    if (condition) return;
    std::fprintf(stderr, "block layout: %s\n", text);
    ++failures;
}

class Plan {
public:
    explicit Plan(std::size_t values = 512u) {
        plan.valueStorage.Reserve(values);
        plan.srtPlanComplete = true;
        plan.resourceTrackingComplete = true;
        MemoryInfo memory;
        memory.kind = ResourceKind::ScalarAddress;
        plan.memoryInfo.push_back(memory);
        root = value(IrOpcode::GetAddressResource, IrType::AddressResource);
        root->AddArgument(value(IrOpcode::GetUserData, IrType::U32));
        table = value(IrOpcode::GetSrtResource, IrType::SrtResource);
    }

    IrValue* value(IrOpcode opcode, IrType type) {
        return plan.valueStorage.Create(opcode, type, nextId++);
    }

    IrValue* constant(std::uint32_t number) {
        IrValue* result = value(IrOpcode::Void, IrType::U32);
        result->SetImmediateU32(number);
        return result;
    }

    IrValue* flat(std::uint32_t slot) {
        IrValue* result = value(IrOpcode::ReadConst, IrType::U32);
        result->AddArgument(table);
        result->AddArgument(constant(slot));
        return result;
    }

    std::uint32_t read(IrValue* handle) {
        const auto slot = static_cast<std::uint32_t>(plan.srtReads.size());
        IrValue* raw = value(IrOpcode::LoadAddressU32, IrType::U32);
        raw->SetFlags(MemoryFlags{0u, 0u});
        raw->AddArgument(handle);
        raw->AddArgument(constant(slot * 4u));
        plan.srtReads.push_back({raw, slot});
        return slot;
    }

    std::uint32_t word() {
        return read(root);
    }

    void words(std::uint32_t count) {
        for (std::uint32_t index = 0; index < count; ++index) word();
    }

    std::uint32_t chased(std::uint32_t pointer) {
        IrValue* handle = value(IrOpcode::GetAddressResource, IrType::AddressResource);
        handle->AddArgument(flat(pointer));
        return read(handle);
    }

    IrValue* user(std::uint32_t index) {
        IrValue* reg = value(IrOpcode::Void, IrType::ScalarReg);
        reg->SetRegister(GuestRegister{RegisterBank::UserData, index});
        IrValue* result = value(IrOpcode::GetUserData, IrType::U32);
        result->AddArgument(reg);
        return result;
    }

    std::uint32_t sourceOf(std::initializer_list<IrValue*> values) {
        DescriptorSource entry;
        for (IrValue* dword : values) entry.dwords[entry.dwordCount++] = dword;
        plan.descriptorSources.push_back(entry);
        return static_cast<std::uint32_t>(plan.descriptorSources.size() - 1u);
    }

    std::uint32_t userSource(std::uint32_t first, std::uint32_t count) {
        DescriptorSource entry;
        for (std::uint32_t index = 0; index < count; ++index) entry.dwords[entry.dwordCount++] = user(first + index);
        plan.descriptorSources.push_back(entry);
        return static_cast<std::uint32_t>(plan.descriptorSources.size() - 1u);
    }

    std::uint32_t source(std::initializer_list<std::uint32_t> slots) {
        DescriptorSource entry;
        for (const auto slot : slots) entry.dwords[entry.dwordCount++] = slot == Missing ? value(IrOpcode::GetUserData, IrType::U32) : flat(slot);
        plan.descriptorSources.push_back(entry);
        return static_cast<std::uint32_t>(plan.descriptorSources.size() - 1u);
    }

    BufferResource& buffer(std::uint32_t descriptor) {
        BufferResource resource;
        resource.source = descriptor;
        resource.read = true;
        plan.info.buffers.push_back(resource);
        return plan.info.buffers.back();
    }

    ImageResource& image(std::uint32_t descriptor, std::size_t modes) {
        ImageResource resource;
        resource.source = descriptor;
        resource.read = true;
        plan.info.images.push_back(resource);
        plan.info.runtimeImageModes.emplace_back(modes, resource);
        return plan.info.images.back();
    }

    SamplerResource& sampler(std::uint32_t descriptor) {
        SamplerResource resource;
        resource.source = descriptor;
        plan.info.samplers.push_back(resource);
        return plan.info.samplers.back();
    }

    IrResourcePlan plan;

private:
    IrValue* root = nullptr;
    IrValue* table = nullptr;
    std::uint32_t nextId = 0;
};

bool Same(const BlockLayout& left, const BlockLayout& right) {
    if (left.abiVersion != right.abiVersion || left.words != right.words || left.refusal != right.refusal) return false;
    if (left.buffers.size() != right.buffers.size() || left.variantKey.size() != right.variantKey.size() || left.imageKey.size() != right.imageKey.size()) return false;
    if (left.userVariantKey.size() != right.userVariantKey.size() || left.userImageKey.size() != right.userImageKey.size()) return false;
    for (std::size_t index = 0; index < left.userVariantKey.size(); ++index) {
        if (left.userVariantKey[index].word != right.userVariantKey[index].word || left.userVariantKey[index].mask != right.userVariantKey[index].mask) return false;
    }
    for (std::size_t index = 0; index < left.userImageKey.size(); ++index) {
        if (left.userImageKey[index].word != right.userImageKey[index].word || left.userImageKey[index].mask != right.userImageKey[index].mask) return false;
    }
    for (std::size_t index = 0; index < left.buffers.size(); ++index) {
        const auto& a = left.buffers[index];
        const auto& b = right.buffers[index];
        if (a.resource != b.resource || a.descriptorWords != b.descriptorWords || a.formatFixed != b.formatFixed || a.typedAlignment != b.typedAlignment) return false;
    }
    for (std::size_t index = 0; index < left.variantKey.size(); ++index) {
        if (left.variantKey[index].word != right.variantKey[index].word || left.variantKey[index].mask != right.variantKey[index].mask) return false;
    }
    for (std::size_t index = 0; index < left.imageKey.size(); ++index) {
        if (left.imageKey[index].word != right.imageKey[index].word || left.imageKey[index].mask != right.imageKey[index].mask) return false;
    }
    return true;
}

BlockRefusal RefusalOf(const Plan& plan) {
    const auto layout = BuildBlockLayout(plan.plan);
    if (layout.Eligible() != (layout.refusal == BlockRefusal::None)) Expect(false, "Eligible disagrees with the refusal");
    return layout.refusal;
}

void FillTwoBuffers(Plan& plan) {
    plan.words(10u);
    plan.buffer(plan.source({4u, 5u, 6u, 7u}));
    plan.buffer(plan.source({9u, 2u, Missing, 0u})).typedAlignment = 4u;
}

void CheckWordsOnly() {
    Plan plan;
    plan.words(5u);
    const auto layout = BuildBlockLayout(plan.plan);
    Expect(layout.Eligible(), "a plan of plain words is refused");
    Expect(layout.abiVersion == BlockAbi::Version, "the layout does not carry the block ABI version");
    Expect(layout.words == 5u, "the word count is not the number of flat slots");
    Expect(layout.buffers.empty() && layout.variantKey.empty() && layout.imageKey.empty(), "a plan of plain words lists resources");
    Expect(layout.SectionBytes() == 32u, "five words do not round up to a 32-byte section");
    Plan empty;
    const auto none = BuildBlockLayout(empty.plan);
    Expect(none.Eligible() && none.words == 0u && none.SectionBytes() == 0u, "an empty plan does not give an empty section");
}

void CheckReadBuffers() {
    Plan plan;
    FillTwoBuffers(plan);
    const auto layout = BuildBlockLayout(plan.plan);
    Expect(layout.Eligible(), "a plan of read buffers is refused");
    Expect(layout.words == 10u && layout.buffers.size() == 2u, "the layout does not list ten words and two buffers");
    if (layout.buffers.size() != 2u) return;
    const std::array<std::uint32_t, 4> first{4u, 5u, 6u, 7u};
    const std::array<std::uint32_t, 4> second{9u, 2u, Missing, 0u};
    Expect(layout.buffers[0].resource == 0u && layout.buffers[1].resource == 1u, "buffer entries do not follow the shader's buffer order");
    Expect(layout.buffers[0].descriptorWords == first, "the first buffer does not name its four V# words");
    Expect(layout.buffers[1].descriptorWords == second, "a V# word the plan does not read is not marked missing, or a read one is misplaced");
    Expect(!layout.buffers[0].formatFixed && !layout.buffers[1].formatFixed, "a buffer without a fixed format is marked fixed");
    Expect(layout.buffers[0].typedAlignment == 1u && layout.buffers[1].typedAlignment == 4u, "the typed alignment is not carried over");
    Expect(layout.variantKey.empty() && layout.imageKey.empty(), "buffers without a fixed format produce key words");
    Expect(layout.SectionBytes() == 80u, "ten words and two entries do not round up to an 80-byte section");
    Expect(layout.SectionBytes() == BlockAbi::SectionBytes(10u, 2u), "SectionBytes disagrees with the block ABI");
}

void CheckFlatOffsets() {
    Plan plan;
    plan.words(4u);
    plan.plan.srtReads[0].flatOffset = 3u;
    plan.plan.srtReads[3].flatOffset = 0u;
    plan.buffer(plan.source({0u, 1u, 2u, 3u}));
    const auto layout = BuildBlockLayout(plan.plan);
    const std::array<std::uint32_t, 4> expected{3u, 1u, 2u, 0u};
    Expect(layout.Eligible() && layout.buffers.size() == 1u && layout.buffers[0].descriptorWords == expected, "V# words are not located by the flat offset of their slot");
}

void CheckRefusals() {
    {
        Plan plan;
        FillTwoBuffers(plan);
        Expect(RefusalOf(plan) == BlockRefusal::None, "the reference plan of the refusal cases is refused");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        MemoryInfo memory;
        memory.kind = ResourceKind::Buffer;
        memory.gpuDescriptor = true;
        plan.plan.memoryInfo.push_back(memory);
        Expect(RefusalOf(plan) == BlockRefusal::GpuDescriptor, "a descriptor read on the GPU is not refused as GpuDescriptor");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.descriptorSources[0].tableColumn = TableColumn{};
        Expect(RefusalOf(plan) == BlockRefusal::ImageTable, "a table column source is not refused as ImageTable");
    }
    {
        Plan plan;
        plan.words(8u);
        plan.image(plan.source({0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}), 1u).table = 0u;
        Expect(RefusalOf(plan) == BlockRefusal::ImageTable, "an image read through a table is not refused as ImageTable");
    }
    {
        Plan plan;
        plan.words(4u);
        plan.sampler(plan.source({0u, 1u, 2u, 3u})).table = 0u;
        Expect(RefusalOf(plan) == BlockRefusal::ImageTable, "a sampler read through a table is not refused as ImageTable");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.info.buffers[1].written = true;
        Expect(RefusalOf(plan) == BlockRefusal::WrittenBuffer, "a written buffer is not refused as WrittenBuffer");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.info.buffers[1].atomic = true;
        Expect(RefusalOf(plan) == BlockRefusal::AtomicBuffer, "an atomic buffer is not refused as AtomicBuffer");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.uniformFill.fill.kind = UniformFillKind::Buffer;
        Expect(RefusalOf(plan) == BlockRefusal::UniformFill, "a uniform fill is not refused as UniformFill");
    }
    {
        Plan plan;
        plan.words(2u);
        const auto chased = plan.chased(0u);
        Expect(RefusalOf(plan) == BlockRefusal::UnguardedPoison, "a word read through a loaded pointer without a guard is not refused as UnguardedPoison");
        plan.plan.guardedSrtSlots.push_back(chased);
        Expect(RefusalOf(plan) == BlockRefusal::None, "a word read through a loaded pointer is refused although it is guarded");
    }
    {
        Plan plan(8192u);
        plan.words(BlockAbi::MaxWords);
        Expect(RefusalOf(plan) == BlockRefusal::None, "a plan of exactly MaxWords words is refused");
        plan.word();
        Expect(RefusalOf(plan) == BlockRefusal::TooManyWords, "a plan of more than MaxWords words is not refused as TooManyWords");
    }
    {
        Plan plan(1024u);
        const auto descriptor = plan.source({Missing, Missing, Missing, Missing});
        for (std::uint32_t index = 0; index < BlockAbi::MaxBuffers; ++index) plan.buffer(descriptor);
        Expect(RefusalOf(plan) == BlockRefusal::None, "a plan of exactly MaxBuffers buffers is refused");
        plan.buffer(descriptor);
        Expect(RefusalOf(plan) == BlockRefusal::TooManyBuffers, "a plan of more than MaxBuffers buffers is not refused as TooManyBuffers");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.srtPlanComplete = false;
        Expect(RefusalOf(plan) == BlockRefusal::SrtPlanIncomplete, "a plan without a completed SRT plan is not refused as SrtPlanIncomplete");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.resourceTrackingComplete = false;
        Expect(RefusalOf(plan) == BlockRefusal::ResourceTrackingIncomplete, "a plan without completed resource tracking is not refused as ResourceTrackingIncomplete");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.info.buffers[1].source = 7u;
        Expect(RefusalOf(plan) == BlockRefusal::BufferSourceMissing, "a buffer without a descriptor source is not refused as BufferSourceMissing");
    }
    {
        Plan plan;
        FillTwoBuffers(plan);
        plan.plan.srtReads[1].flatOffset = 0u;
        Expect(RefusalOf(plan) == BlockRefusal::WordShared, "two slots sharing a flat offset are not refused as WordShared");
    }
    {
        Plan plan;
        plan.words(4u);
        plan.buffer(plan.source({0u, 1u, 2u, Missing})).descriptorFormatted = true;
        Expect(RefusalOf(plan) == BlockRefusal::BufferFormatWordMissing, "a fixed-format buffer whose format word is outside the block is not refused as BufferFormatWordMissing");
    }
    {
        Plan plan;
        plan.words(8u);
        plan.image(plan.source({0u, 1u, 2u, Missing, 4u, 5u, 6u, 7u}), 2u);
        Expect(RefusalOf(plan) == BlockRefusal::ImageWordMissing, "a multi-mode image with a T# word outside the block is not refused as ImageWordMissing");
    }
}

void CheckRefusalNames() {
    const auto count = static_cast<std::uint32_t>(BlockRefusal::Count);
    std::vector<std::string_view> names;
    for (std::uint32_t index = 0; index < count; ++index) names.push_back(BlockRefusalName(static_cast<BlockRefusal>(index)));
    for (std::size_t index = 0; index < names.size(); ++index) {
        Expect(!names[index].empty(), "a refusal has no name");
        for (std::size_t other = 0; other < index; ++other) Expect(names[index] != names[other], "two refusals share a name");
    }
    Expect(BlockRefusalName(BlockRefusal::None) == "None" && BlockRefusalName(BlockRefusal::SamplerSourceMissing) == "SamplerSourceMissing", "refusal names do not match their enumerators");
    Expect(!BlockRefusalName(BlockRefusal::Count).empty(), "an out-of-range refusal has no name");
}

void CheckVariantKey() {
    Plan plan;
    plan.words(12u);
    plan.buffer(plan.source({0u, 1u, 2u, 3u})).descriptorFormatted = true;
    plan.buffer(plan.source({4u, 5u, 6u, 7u}));
    const auto layout = BuildBlockLayout(plan.plan);
    Expect(layout.Eligible(), "a plan with a fixed-format buffer is refused");
    Expect(layout.buffers.size() == 2u && layout.buffers[0].formatFixed && !layout.buffers[1].formatFixed, "the fixed format flag does not follow descriptorFormatted");
    Expect(layout.variantKey.size() == 1u && layout.variantKey[0].word == 3u, "the variant key does not name the format word of the fixed-format buffer alone");
    if (layout.variantKey.size() != 1u) return;
    Expect((layout.variantKey[0].mask & (0x7fu << 12u)) == (0x7fu << 12u), "the variant mask does not cover the buffer format bits");
    std::vector<std::uint32_t> words(12u, 0u);
    for (std::uint32_t index = 0; index < words.size(); ++index) words[index] = 0x01010101u * (index + 1u);
    const auto base = BlockVariantKey(layout, words);
    Expect(BlockVariantKey(layout, words) == base, "the variant key of the same words changes");
    for (std::uint32_t bit = 12u; bit < 19u; ++bit) {
        auto changed = words;
        changed[3] ^= 1u << bit;
        Expect(BlockVariantKey(layout, changed) != base, "the variant key ignores a format bit of a fixed-format buffer");
    }
    for (const std::uint32_t outside : {0u, 1u, 2u, 4u, 5u, 6u, 7u, 8u, 11u}) {
        auto changed = words;
        changed[outside] ^= 1u << 12u;
        Expect(BlockVariantKey(layout, changed) == base, "the variant key follows a word outside the key");
    }
    BlockLayout narrow;
    narrow.words = 12u;
    narrow.variantKey.push_back({3u, 0x7fu << 12u});
    narrow.variantKey.push_back({9u, 0x000000ffu});
    const auto narrowBase = BlockVariantKey(narrow, words);
    auto inside = words;
    inside[3] ^= 1u << 14u;
    Expect(BlockVariantKey(narrow, inside) != narrowBase, "the variant key ignores a bit inside the first mask");
    inside = words;
    inside[9] ^= 1u << 7u;
    Expect(BlockVariantKey(narrow, inside) != narrowBase, "the variant key ignores a bit inside the second mask");
    for (const std::uint32_t bit : {0u, 11u, 19u, 31u}) {
        auto changed = words;
        changed[3] ^= 1u << bit;
        Expect(BlockVariantKey(narrow, changed) == narrowBase, "the variant key follows a bit outside the mask");
    }
    auto changed = words;
    changed[9] ^= 1u << 8u;
    Expect(BlockVariantKey(narrow, changed) == narrowBase, "the variant key follows a bit outside the second mask");
    BlockLayout none;
    Expect(BlockVariantKey(none, words) == BlockVariantKey(none, {}), "a layout without variant bits does not give one constant key");
    bool thrown = false;
    try {
        static_cast<void>(BlockVariantKey(narrow, std::span<const std::uint32_t>(words).first(9u)));
    } catch (const std::out_of_range&) {
        thrown = true;
    }
    Expect(thrown, "a variant word outside the given words is read");
}

void CheckImageKeys() {
    Plan plan;
    plan.words(24u);
    plan.image(plan.source({0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}), 1u);
    plan.image(plan.source({8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}), 3u);
    plan.sampler(plan.source({16u, 17u, 18u, 19u}));
    const auto layout = BuildBlockLayout(plan.plan);
    Expect(layout.Eligible(), "a plan of direct images and samplers is refused");
    Expect(layout.imageKey.size() == 20u, "the image key does not list every T# and S# word of the plan");
    for (std::size_t index = 0; index < layout.imageKey.size(); ++index) Expect(layout.imageKey[index].word == index && layout.imageKey[index].mask == WholeWord, "an image key word is misplaced or masked");
    Expect(layout.variantKey.size() == 8u, "the variant key does not list the T# words of the multi-mode image alone");
    for (std::size_t index = 0; index < layout.variantKey.size(); ++index) Expect(layout.variantKey[index].word == index + 8u && layout.variantKey[index].mask != 0u, "a multi-mode image word is missing from the variant key");
    std::vector<std::uint32_t> words(24u, 0x55aa55aau);
    const auto base = BlockVariantKey(layout, words);
    auto changed = words;
    changed[9] ^= 1u << 20u;
    Expect(BlockVariantKey(layout, changed) != base, "the variant key ignores the format of a multi-mode image");
    changed = words;
    changed[1] ^= 1u << 20u;
    Expect(BlockVariantKey(layout, changed) == base, "the variant key follows a single-mode image");
    changed = words;
    changed[17] ^= 1u;
    Expect(BlockVariantKey(layout, changed) == base, "the variant key follows a sampler word");
}

void CheckUserDataImages() {
    {
        Plan plan;
        plan.words(6u);
        plan.buffer(plan.source({0u, 1u, 2u, 3u}));
        plan.image(plan.userSource(4u, 8u), 3u);
        plan.image(plan.userSource(12u, 8u), 1u);
        plan.sampler(plan.userSource(20u, 4u));
        const auto layout = BuildBlockLayout(plan.plan);
        Expect(layout.Eligible(), "a plan whose T# and S# words are user data words is refused");
        Expect(layout.words == 6u && layout.buffers.size() == 1u, "user data words of images change the block words or buffers");
        Expect(layout.variantKey.empty() && layout.imageKey.empty(), "user data words of images are listed as block words");
        Expect(layout.userVariantKey.size() == 8u, "the user variant key does not list the T# words of the multi-mode image alone");
        for (std::size_t index = 0; index < layout.userVariantKey.size(); ++index) Expect(layout.userVariantKey[index].word == index + 4u && layout.userVariantKey[index].mask == WholeWord, "a user variant word is misplaced or masked");
        Expect(layout.userImageKey.size() == 20u, "the user image key does not list every T# and S# word held in user data");
        for (std::size_t index = 0; index < layout.userImageKey.size(); ++index) Expect(layout.userImageKey[index].word == index + 4u && layout.userImageKey[index].mask == WholeWord, "a user image key word is misplaced or masked");
        std::vector<std::uint32_t> words(6u, 0x11111111u);
        std::vector<std::uint32_t> userData(64u, 0x22222222u);
        const auto base = BlockVariantKey(layout, words, userData);
        Expect(BlockVariantKey(layout, words, userData) == base, "the variant key of the same user data changes");
        for (std::uint32_t word = 4u; word < 12u; ++word) {
            auto changed = userData;
            changed[word] ^= 1u << 20u;
            Expect(BlockVariantKey(layout, words, changed) != base, "the variant key ignores a T# word of a multi-mode image held in user data");
        }
        for (const std::uint32_t outside : {0u, 3u, 12u, 19u, 20u, 23u, 63u}) {
            auto changed = userData;
            changed[outside] ^= 1u << 20u;
            Expect(BlockVariantKey(layout, words, changed) == base, "the variant key follows a user data word outside the key");
        }
        auto changedWords = words;
        changedWords[3] ^= 1u;
        Expect(BlockVariantKey(layout, changedWords, userData) == base, "the variant key follows a block word outside the key");
        bool refused = false;
        try {
            static_cast<void>(BlockVariantKey(layout, words));
        } catch (const std::logic_error&) {
            refused = true;
        }
        Expect(refused, "a layout keyed on user data gives a variant key without the user data");
        bool thrown = false;
        try {
            static_cast<void>(BlockVariantKey(layout, words, std::span<const std::uint32_t>(userData).first(11u)));
        } catch (const std::out_of_range&) {
            thrown = true;
        }
        Expect(thrown, "a variant word outside the given user data is read");
    }
    {
        Plan plan;
        plan.words(8u);
        plan.image(plan.sourceOf({plan.flat(0u), plan.user(9u), plan.constant(7u), plan.flat(3u), plan.user(2u), plan.flat(5u), plan.constant(0u), plan.flat(7u)}), 2u);
        const auto layout = BuildBlockLayout(plan.plan);
        Expect(layout.Eligible(), "a multi-mode image whose T# mixes block words, user data words and constants is refused");
        Expect(layout.variantKey.size() == 4u && layout.imageKey.size() == 4u, "the block words of a mixed T# are not all keyed");
        Expect(layout.userVariantKey.size() == 2u && layout.userImageKey.size() == 2u, "the user data words of a mixed T# are not all keyed");
        if (layout.variantKey.size() == 4u) Expect(layout.variantKey[0].word == 0u && layout.variantKey[1].word == 3u && layout.variantKey[2].word == 5u && layout.variantKey[3].word == 7u, "the block words of a mixed T# are misplaced");
        if (layout.userVariantKey.size() == 2u) Expect(layout.userVariantKey[0].word == 9u && layout.userVariantKey[1].word == 2u, "the user data words of a mixed T# are misplaced");
        std::vector<std::uint32_t> words(8u, 1u);
        std::vector<std::uint32_t> userData(16u, 2u);
        const auto base = BlockVariantKey(layout, words, userData);
        auto changed = userData;
        changed[2] ^= 4u;
        Expect(BlockVariantKey(layout, words, changed) != base, "the variant key ignores the user data half of a mixed T#");
        auto changedWords = words;
        changedWords[5] ^= 4u;
        Expect(BlockVariantKey(layout, changedWords, userData) != base, "the variant key ignores the block half of a mixed T#");
    }
    {
        Plan plan;
        plan.words(2u);
        plan.image(plan.userSource(0u, 8u), 1u);
        const auto layout = BuildBlockLayout(plan.plan);
        Expect(layout.Eligible() && layout.userVariantKey.empty() && layout.userImageKey.size() == 8u, "a single-mode image held in user data is refused or keyed as a variant");
        const std::vector<std::uint32_t> words(2u, 5u);
        Expect(BlockVariantKey(layout, words) == BlockVariantKey(layout, words, {}), "a layout without user variant words gives two keys");
    }
    {
        Plan plan;
        plan.words(2u);
        plan.plan.userDataCount = 8u;
        plan.image(plan.userSource(4u, 8u), 2u);
        Expect(RefusalOf(plan) == BlockRefusal::ImageWordMissing, "a multi-mode image with a T# word past the user data words is not refused as ImageWordMissing");
    }
    {
        Plan plan;
        plan.words(2u);
        plan.plan.userDataBase = 16u;
        plan.image(plan.userSource(16u, 8u), 2u);
        const auto layout = BuildBlockLayout(plan.plan);
        Expect(layout.Eligible() && layout.userVariantKey.size() == 8u && layout.userVariantKey[0].word == 0u && layout.userVariantKey[7].word == 7u, "user data words are not counted from the stage's first user data register");
        Plan below;
        below.words(2u);
        below.plan.userDataBase = 16u;
        below.image(below.userSource(12u, 8u), 2u);
        Expect(RefusalOf(below) == BlockRefusal::ImageWordMissing, "a T# word before the stage's user data words is not refused as ImageWordMissing");
    }
    {
        Plan plan;
        plan.words(4u);
        IrValue* sum = plan.value(IrOpcode::BitwiseOr32, IrType::U32);
        sum->AddArgument(plan.flat(0u));
        sum->AddArgument(plan.user(1u));
        plan.image(plan.sourceOf({sum, plan.flat(1u), plan.flat(2u), plan.flat(3u), plan.user(4u), plan.user(5u), plan.user(6u), plan.user(7u)}), 2u);
        Expect(RefusalOf(plan) == BlockRefusal::ImageWordMissing, "a multi-mode image with a computed T# word is not refused as ImageWordMissing");
    }
}

void CheckSamePlanSameLayout() {
    Plan plan;
    plan.words(30u);
    plan.buffer(plan.source({0u, 1u, 2u, 3u})).descriptorFormatted = true;
    plan.buffer(plan.source({4u, Missing, 6u, 7u}));
    plan.image(plan.source({8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}), 2u);
    plan.sampler(plan.source({16u, 17u, 18u, 19u}));
    const auto first = BuildBlockLayout(plan.plan);
    const auto second = BuildBlockLayout(plan.plan);
    Expect(Same(first, second), "the same plan gives two layouts");
    Plan twin;
    twin.words(30u);
    twin.buffer(twin.source({0u, 1u, 2u, 3u})).descriptorFormatted = true;
    twin.buffer(twin.source({4u, Missing, 6u, 7u}));
    twin.image(twin.source({8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}), 2u);
    twin.sampler(twin.source({16u, 17u, 18u, 19u}));
    Expect(Same(first, BuildBlockLayout(twin.plan)), "two equal plans give two layouts");
    twin.plan.info.buffers[1].written = true;
    const auto refused = BuildBlockLayout(twin.plan);
    Expect(!Same(first, refused) && Same(refused, BuildBlockLayout(twin.plan)), "a refused plan does not give one stable layout");
    Expect(refused.words == first.words && refused.buffers.size() == first.buffers.size(), "a refused layout loses the plan's counts");
}

}

int main() {
    try {
        CheckWordsOnly();
        CheckReadBuffers();
        CheckFlatOffsets();
        CheckRefusals();
        CheckRefusalNames();
        CheckVariantKey();
        CheckImageKeys();
        CheckUserDataImages();
        CheckSamePlanSameLayout();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "block layout: unexpected exception: %s\n", error.what());
        return 1;
    }
    if (failures != 0) std::fprintf(stderr, "block layout: %d check(s) failed\n", failures);
    return failures == 0 ? 0 : 1;
}
