#include "BdaAbi.hpp"
#include "BlockAbi.hpp"
#include "PipelineSpecialization.hpp"
#include "Recompiler.hpp"
#ifdef ANYPS5_BLOCK_TEST_VALIDATOR
#include <spirv-tools/libspirv.hpp>
#endif
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <set>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t NoWord = 0xffffffffu;
constexpr std::uint32_t TableWords = 5u;

int failures = 0;

void Expect(bool condition, const char* text) {
    if (condition) return;
    std::fprintf(stderr, "block end to end: %s\n", text);
    ++failures;
}

SpirvTarget Target() {
    static constexpr std::array<std::uint32_t, 5> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityClipDistance, spv::CapabilityInt64Atomics,
    };
    static constexpr std::array<std::string_view, 4> extensions{
        "SPV_EXT_descriptor_indexing", "SPV_KHR_8bit_storage", "SPV_KHR_float_controls", "SPV_KHR_physical_storage_buffer",
    };
    return {0x00401000u, 0x00010300u, 32, BdaAbi::Version, capabilities, extensions, false,
            {1024, 1024, 64}, 1024, 32768, {}, {}};
}

struct Fixture {
    std::vector<std::uint32_t> code{
        0xf4080200u, 0xfa000000u,
        0xf4000300u, 0xfa000010u,
        0xbf8cc07fu,
        0xe0000000u, 0x80020a00u,
        0xbf8c3f70u,
        0x7e1c020cu,
        0xf800180fu, 0x0e0c0b0au,
        0xbf810000u};
    alignas(16) std::array<std::uint32_t, 64> data{};
    alignas(16) std::array<std::uint32_t, TableWords> table{};
    std::array<std::uint32_t, 2> userData{};
    std::array<MemoryRegion, 3> memory{};

    Fixture() {
        for (std::uint32_t index = 0; index < data.size(); ++index) data[index] = 0x3f000000u + index;
        const auto dataAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data.data()));
        const auto tableAddress = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(table.data()));
        table = {static_cast<std::uint32_t>(dataAddress), static_cast<std::uint32_t>((dataAddress >> 32u) & 0xffffu), static_cast<std::uint32_t>(sizeof(data)), 0x31016facu, 0x3f800000u};
        userData = {static_cast<std::uint32_t>(tableAddress), static_cast<std::uint32_t>(tableAddress >> 32u)};
        memory = {{{static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(code.data())), std::as_bytes(std::span<const std::uint32_t>(code))},
                   {tableAddress, std::as_bytes(std::span<const std::uint32_t>(table))},
                   {dataAddress, std::as_bytes(std::span<const std::uint32_t>(data))}}};
    }

    [[nodiscard]] RecompileRequest Request(bool blockAccess, std::uint32_t pushBytes) const {
        RecompileRequest request{};
        request.shader = {ShaderStage::Fragment, reinterpret_cast<std::uintptr_t>(code.data()), std::span<const std::uint32_t>(code), 0, {}};
        request.target = Target();
        request.target.blockAccess = blockAccess;
        request.layout = {0, 0, 0, pushBytes};
        request.context.waveSize = 32;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.pixel = ShaderPixelStageInfo{};
        request.context.pixel->wave32 = true;
        request.context.pixel->targetOutputMode[0] = 9u;
        request.context.pixel->targetExportMapping.fill(0xe4u);
        request.context.memory = memory;
        request.useCache = false;
        return request;
    }
};

struct Shape {
    bool physical = false;
    bool pushConstants = false;
    std::set<std::uint32_t> bindings;
    std::size_t physicalLoads = 0;
    std::size_t arrayLengths = 0;
};

Shape Describe(std::span<const std::uint32_t> words) {
    Shape shape;
    if (words.size() < 5u || words[0] != spv::MagicNumber) return shape;
    std::set<std::uint32_t> physicalPointers;
    std::set<std::uint32_t> physicalValues;
    for (std::size_t offset = 5u; offset < words.size();) {
        const std::size_t count = words[offset] >> spv::WordCountShift;
        if (count == 0u || count > words.size() - offset) break;
        const auto opcode = words[offset] & spv::OpCodeMask;
        if (opcode == spv::OpMemoryModel && words[offset + 1u] == spv::AddressingModelPhysicalStorageBuffer64) shape.physical = true;
        if (opcode == spv::OpDecorate && count == 4u && words[offset + 2u] == spv::DecorationBinding) shape.bindings.insert(words[offset + 3u]);
        if (opcode == spv::OpTypePointer && words[offset + 2u] == spv::StorageClassPhysicalStorageBuffer) physicalPointers.insert(words[offset + 1u]);
        if (opcode == spv::OpTypePointer && words[offset + 2u] == spv::StorageClassPushConstant) shape.pushConstants = true;
        if ((opcode == spv::OpConvertUToPtr || opcode == spv::OpAccessChain || opcode == spv::OpInBoundsAccessChain) && physicalPointers.contains(words[offset + 1u])) physicalValues.insert(words[offset + 2u]);
        if (opcode == spv::OpLoad && physicalValues.contains(words[offset + 3u])) ++shape.physicalLoads;
        if (opcode == spv::OpArrayLength) ++shape.arrayLengths;
        offset += count;
    }
    return shape;
}

std::uint64_t Digest(std::span<const std::uint32_t> words) {
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (const auto word : words) hash = (hash ^ word) * 0x100000001b3ull;
    return hash;
}

bool Valid(std::span<const std::uint32_t> words) {
#ifdef ANYPS5_BLOCK_TEST_VALIDATOR
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_1);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        std::fprintf(stderr, "block end to end: validator at word %lu: %s\n", static_cast<unsigned long>(position.index), message);
    });
    return tools.Validate(words.data(), words.size());
#else
    return words.size() >= 5u && words[0] == spv::MagicNumber;
#endif
}

bool Specialized(const RecompileResult& result, std::uint32_t id) {
    return std::ranges::any_of(result.specialization, [id](const PipelineSpecializationConstant& constant) { return constant.id == id; });
}

std::size_t Roles(const RecompileResult& result, DescriptorRole role) {
    return static_cast<std::size_t>(std::ranges::count_if(result.bindings, [role](const DescriptorBinding& binding) { return binding.role == role; }));
}

void Run() {
    const Fixture fixture;
    const auto classic = Recompile(fixture.Request(false, 128u));
    const auto classicWords = classic.spirv.Copy();
    const auto classicShape = Describe(classicWords);
    std::printf("block end to end: classic module %zu words, digest %016llx\n", classicWords.size(), static_cast<unsigned long long>(Digest(classicWords)));
    Expect(!classic.blockModule, "a request without block access produced a block module");
    Expect(classic.blockAddressDword == NoWord, "a classic result names a block address word");
    Expect(Valid(classicWords), "the classic module is rejected by the validator");
    Expect(!classicShape.physical && classicShape.physicalLoads == 0u, "the classic module reads through physical pointers");
    Expect(classicShape.arrayLengths != 0u, "the classic module does not bound its buffer by its descriptor length");
    Expect(Roles(classic, DescriptorRole::GuestBuffers) == 1u && Roles(classic, DescriptorRole::FlattenedSrt) == 1u, "the classic result lacks its buffer or flattened SRT binding");

    const auto block = Recompile(fixture.Request(true, 128u));
    const auto blockWords = block.spirv.Copy();
    const auto blockShape = Describe(blockWords);
    std::printf("block end to end: block module %zu words, digest %016llx\n", blockWords.size(), static_cast<unsigned long long>(Digest(blockWords)));
    Expect(block.blockModule, "a request with block access on an eligible plan produced a classic module");
    Expect(block.blockLayout.Eligible() && block.blockLayout.abiVersion == BlockAbi::Version, "the block layout of an eligible plan is refused");
    Expect(block.blockLayout.words == TableWords, "the block layout does not hold one word per flat slot");
    Expect(block.blockLayout.buffers.size() == 1u, "the block layout does not hold the buffer of the shader");
    if (block.blockLayout.buffers.size() == 1u) {
        const auto& buffer = block.blockLayout.buffers.front();
        Expect(buffer.resource == 0u, "the block buffer does not name its resource");
        Expect(std::ranges::none_of(buffer.descriptorWords, [](std::uint32_t word) { return word == NoWord || word >= TableWords; }), "a word of the V# read through the SRT is not in the block");
    }
    Expect(Valid(blockWords), "the block module is rejected by the validator");
    Expect(blockShape.physical && blockShape.pushConstants, "the block module does not read a physical address from its push constants");
    Expect(blockShape.physicalLoads >= TableWords + BlockAbi::BufferEntryWords, "the block module does not load its words and its buffer entry through the block");
    Expect(blockShape.arrayLengths == 0u, "the block module still bounds a buffer by a descriptor length");
    Expect(std::ranges::includes(classicShape.bindings, blockShape.bindings) && classicShape.bindings.size() == blockShape.bindings.size() + 2u, "the block module does not drop exactly the buffer and flattened SRT bindings");
    Expect(Roles(block, DescriptorRole::GuestBuffers) == 1u && Roles(block, DescriptorRole::FlattenedSrt) == 1u, "the block result does not keep its buffer and flattened SRT bindings as metadata");
    Expect(block.bindings.size() == classic.bindings.size(), "the block result does not list the bindings of the classic result");
    Expect(!Specialized(block, PipelineSpecialization::BufferBase) && !Specialized(block, PipelineSpecialization::BufferBase + 2u) && !Specialized(block, PipelineSpecialization::BufferBase + 3u), "the block result still specializes the address, tail or misalignment of its buffer");
    Expect(block.shaderDataUserWords == classic.shaderDataUserWords && block.memoryOffsetDword == classic.memoryOffsetDword && block.shaderDataDwords == classic.shaderDataDwords, "the block result moves the shader data words");
    Expect(block.pushConstants.size() == classic.pushConstants.size() + BlockAbi::PushAddressDwords * sizeof(std::uint32_t), "the block result does not add the two address words to its push constants");
    Expect(block.blockAddressDword * sizeof(std::uint32_t) == classic.pushConstants.size(), "the block address words do not follow the pushed shader data");
    Expect(block.pushConstants.size() >= classic.pushConstants.size() && std::equal(classic.pushConstants.begin(), classic.pushConstants.end(), block.pushConstants.begin()), "the block result changes the pushed shader data");
    Expect(std::all_of(block.pushConstants.begin() + static_cast<std::ptrdiff_t>(std::min(classic.pushConstants.size(), block.pushConstants.size())), block.pushConstants.end(), [](std::byte value) { return value == std::byte{0}; }), "the reserved block address words are not zero");

    const auto again = Recompile(fixture.Request(false, 128u));
    Expect(std::ranges::equal(again.spirv.Words(), classicWords), "a classic module compiled after a block module differs from the one compiled before");
    Expect(again.pushConstants == classic.pushConstants, "a classic result compiled after a block module differs from the one compiled before");

    const auto narrowClassic = Recompile(fixture.Request(false, 4u));
    const auto narrow = Recompile(fixture.Request(true, 4u));
    Expect(!narrow.blockModule && narrow.blockAddressDword == NoWord, "a push range too small for the block address still produced a block module");
    Expect(std::ranges::equal(narrow.spirv.Words(), narrowClassic.spirv.Words()), "a refused block request does not produce the classic module");
    Expect(narrow.pushConstants == narrowClassic.pushConstants, "a refused block request changes the push constants");
}

}

int main() {
    try {
        Run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "block end to end: %s\n", error.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::puts("Block end to end tests passed");
    return 0;
}
