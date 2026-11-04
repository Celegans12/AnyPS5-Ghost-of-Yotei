#include "BdaShader.hpp"
#include "BdaAbi.hpp"
#include "IntermediateRepresentation/IrMetadata/ShaderStage.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <array>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string Hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}

using namespace AgcDriver::Graphics;

class Pipeline {
public:
    Pipeline(const Context& context, std::span<const std::uint32_t> code, const std::array<Buffer*, 3>& buffers) : context(context) {
        try {
            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            for (std::uint32_t i = 0; i < bindings.size(); ++i) bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setInfo.bindingCount = bindings.size();
            setInfo.pBindings = bindings.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
            VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            poolInfo.maxSets = 1;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &size;
            Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
            VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, pool, 1, &setLayout};
            Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &set), "vkAllocateDescriptorSets");
            for (std::uint32_t i = 0; i < buffers.size(); ++i) {
                const VkDescriptorBufferInfo buffer{buffers[i]->Handle(), 0, buffers[i]->Bytes().size()};
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = set;
                write.dstBinding = i;
                write.descriptorCount = 1;
                write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write.pBufferInfo = &buffer;
                context.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(context.device, 1, &write, 0, nullptr);
            }
            VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
            layoutInfo.setLayoutCount = 1;
            layoutInfo.pSetLayouts = &setLayout;
            Check(context.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(context.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
            VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            moduleInfo.codeSize = code.size_bytes();
            moduleInfo.pCode = code.data();
            Check(context.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(context.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            pipelineInfo.layout = layout;
            Check(context.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
        } catch (...) { release(); throw; }
    }

    ~Pipeline() { release(); }

    void Run(std::uint32_t groups) {
        CommandBatch batch(context);
        const auto commands = batch.Handle();
        VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &upload, 0, nullptr, 0, nullptr);
        context.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        context.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        context.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, groups, 1, 1);
        VkMemoryBarrier download{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        context.Function<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &download, 0, nullptr, 0, nullptr);
        batch.SubmitAndWait();
    }

private:
    void release() noexcept {
        if (pipeline) context.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(context.device, pipeline, nullptr);
        if (module) context.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(context.device, module, nullptr);
        if (layout) context.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(context.device, layout, nullptr);
        if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
        if (setLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, setLayout, nullptr);
    }

    const Context& context;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

}

static void RunWrittenRangeDecodeTests() {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::vector<std::uint32_t> words(Abi::FaultBufferBytes / sizeof(std::uint32_t));
    std::vector<std::pair<std::uint64_t, std::size_t>> marked;
    const auto decode = [&] {
        marked.clear();
        return Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t address, std::size_t bytes) { marked.emplace_back(address, bytes); });
    };
    const auto claim = [&](std::uint32_t index, std::uint32_t slot, std::uint64_t address, std::uint32_t mask) {
        words[Abi::WrittenListWord + index] = slot;
        words[Abi::WrittenSlotsWord + 2 * slot] = Abi::WrittenChunkKey(address);
        words[Abi::WrittenSlotsWord + 2 * slot + 1] = mask;
        words[Abi::WrittenCountWord] = std::max(words[Abi::WrittenCountWord], index + 1);
    };
    constexpr std::uint64_t page = std::uint64_t{1} << Abi::WrittenPageShift;
    Require(decode() && marked.empty(), "an empty written-page record marked pages");
    constexpr std::uint64_t top = 0x7ffffffe0000ULL;
    claim(0, 5, top, 0b1011u);
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{top, 2 * page}, {top + 3 * page, page}}, "written pages near the top of the address space were not decoded as written");
    std::fill(words.begin(), words.end(), 0u);
    constexpr std::uint64_t high = 0x200000001000ULL;
    claim(0, Abi::WrittenPageSlots - 1, high, 1u << ((high >> Abi::WrittenPageShift) & (Abi::WrittenChunkPages - 1u)));
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{high, page}}, "a written page above 16 TiB was decoded at another address");
    std::fill(words.begin(), words.end(), 0u);
    constexpr std::uint64_t chunk = 0x3e0000000ULL;
    claim(0, 7, chunk, 0xffffffffu);
    claim(1, 8, chunk + (std::uint64_t{1} << Abi::WrittenChunkShift), 1u);
    claim(2, 9, chunk - (std::uint64_t{1} << Abi::WrittenChunkShift), 0x80000000u);
    Require(decode() && marked == std::vector<std::pair<std::uint64_t, std::size_t>>{{chunk, 32 * page}, {chunk + 32 * page, page}, {chunk - page, page}}, "whole, first-page and last-page chunks were not decoded exactly");
    words[Abi::WrittenListWord + 1] = Abi::WrittenPageSlots;
    Require(!decode(), "a claim outside the slots was accepted");
    std::fill(words.begin(), words.end(), 0u);
    words[Abi::WrittenCountWord] = 1;
    Require(!decode(), "a claim of an empty slot was accepted");
    words[Abi::WrittenCountWord] = Abi::WrittenPageSlots + 1;
    Require(!decode(), "more claims than slots were accepted");
}

static void RunBdaDwordWriteTests(const Context& context) {
    RunWrittenRangeDecodeTests();
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340f80ULL;
    constexpr std::size_t firstBytes = 160;
    Buffer first(context, firstBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer readOnly(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    const std::array<Abi::Range, 3> ranges{{
        {guest, guest + firstBytes, first.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + firstBytes, guest + firstBytes + 16, second.DeviceAddress(), Abi::Read | Abi::Write, 0},
        {guest + firstBytes + 16, guest + firstBytes + 32, readOnly.DeviceAddress(), Abi::Read, 0},
    }};
    Buffer table(context, sizeof(Abi::Header) + sizeof(ranges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const std::array<std::uint32_t, 4> values{0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u};
    const auto bytesAt = [&](std::uint64_t address) -> std::byte* {
        for (std::size_t index = 0; index < ranges.size(); ++index) {
            if (address < ranges[index].begin || address >= ranges[index].end) continue;
            auto& buffer = index == 0 ? first : index == 1 ? second : readOnly;
            return buffer.Bytes().data() + (address - ranges[index].begin);
        }
        return nullptr;
    };
    const auto notedBytes = [&](std::uint64_t first, std::uint64_t last) {
        std::uint64_t covered = 0, total = 0;
        const auto* faultWords = reinterpret_cast<const std::uint32_t*>(fault.Bytes().data());
        Require(Abi::ForEachWrittenRange(faultWords, [&](std::uint64_t begin, std::size_t bytes) {
            total += bytes;
            for (const auto address : {first, last}) covered += address >= begin && address < begin + bytes ? 1u : 0u;
        }), "a BDA dword write left an invalid written-page record");
        return std::pair{covered, total};
    };
    const auto write = [&](std::uint64_t address, std::uint32_t dwords, std::uint32_t storedBytes, Abi::FaultReason reason, std::uint32_t groups = 1) {
        const Abi::Header header{Abi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        for (auto* buffer : {&first, &second, &readOnly}) std::memset(buffer->Bytes().data(), 0xa5, buffer->Bytes().size());
        Pipeline pipeline(context, MakeBdaDwordWriteTestShader(address, dwords, values.data()), {&table, &fault, &output});
        pipeline.Run(groups);
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        for (std::uint32_t byte = 0; byte < dwords * 4u; ++byte) {
            const auto* stored = bytesAt(address + byte);
            const auto expected = byte < storedBytes ? static_cast<std::byte>(values[byte / 4u] >> ((byte % 4u) * 8u)) : std::byte{0xa5};
            Require(stored != nullptr && *stored == expected, "BDA dword write stored incorrect bytes at offset " + std::to_string(byte) + " of guest + " + std::to_string(address - guest));
        }
        Require(*bytesAt(address - 1) == std::byte{0xa5}, "BDA dword write touched the byte before it");
        if (address + dwords * 4u < guest + firstBytes + 32) Require(*bytesAt(address + dwords * 4u) == std::byte{0xa5}, "BDA dword write touched the byte after it");
        if (static_cast<std::uint32_t>(reason) == 0) {
            Require(report.state == Abi::FaultState::Empty, "a mapped BDA dword write published a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == 0x1234, "a BDA dword write did not publish its fault");
        }
        const auto [covered, total] = notedBytes(address, address + (storedBytes != 0 ? storedBytes - 1u : 0u));
        if (storedBytes == 0) {
            Require(total == 0, "a BDA dword write that stored nothing noted pages");
        } else {
            const auto pages = ((address + storedBytes - 1u) >> Abi::WrittenPageShift) - (address >> Abi::WrittenPageShift) + 1u;
            Require(covered == 2 && total == pages << Abi::WrittenPageShift, "a BDA dword write did not note exactly its written pages");
            const auto chunks = ((address + storedBytes - 1u) >> Abi::WrittenChunkShift) - (address >> Abi::WrittenChunkShift) + 1u;
            std::uint32_t claims = 0;
            std::memcpy(&claims, fault.Bytes().data() + Abi::WrittenCountWord * sizeof(std::uint32_t), sizeof(claims));
            Require(claims == chunks, "a BDA dword write claimed a written-page slot more than once per chunk");
        }
    };
    write(guest + 16, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 20, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 8, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 12, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 32, 3, 12, static_cast<Abi::FaultReason>(0));
    write(guest + 0x78, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0));
    write(guest + 41, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes - 8, 4, 16, static_cast<Abi::FaultReason>(0));
    write(guest + firstBytes + 12, 4, 4, Abi::FaultReason::Permission);
    write(guest + 0x7c, 2, 8, static_cast<Abi::FaultReason>(0), 64);
}

static void RunBdaRangeMemoTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    constexpr std::uint64_t highest = 0xffffffffffffff00ULL;
    const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer first(context, 3, usage);
    Buffer second(context, 2, usage);
    Buffer words(context, 16, usage);
    Buffer shifted(context, 20, usage);
    Buffer large(context, 1024, usage);
    Buffer overlap(context, 0x200, usage);
    Buffer above(context, 16, usage);
    Buffer spread(context, 0x40 * 20, usage);
    std::uint32_t seed = 0x17;
    for (auto* buffer : {&first, &second, &words, &shifted, &large, &overlap, &above, &spread}) {
        for (std::size_t index = 0; index < buffer->Bytes().size(); ++index) buffer->Bytes()[index] = std::byte(static_cast<std::uint8_t>(seed + index * 13u));
        seed += 0x29;
    }
    std::vector<Abi::Range> mixed{{guest, guest + 3, first.DeviceAddress(), Abi::Read, 0},
        {guest + 3, guest + 5, second.DeviceAddress(), Abi::Read, 0},
        {guest + 0x100, guest + 0x110, words.DeviceAddress(), Abi::Read, 0},
        {guest + 0x200, guest + 0x214, shifted.DeviceAddress() + 1u, Abi::Read, 0},
        {guest + 0x300, guest + 0x310, words.DeviceAddress(), Abi::Write, 0},
        {guest + 0x400, guest + 0x800, large.DeviceAddress(), Abi::Read, 0},
        {guest + 0x700, guest + 0x900, overlap.DeviceAddress(), Abi::Read, 0},
        {highest, highest + 0x10, above.DeviceAddress(), Abi::Read, 0}};
    std::vector<Abi::Range> indexed;
    for (std::uint64_t index = 0; index < 20u; ++index) indexed.push_back({guest + index * 0x80u, guest + index * 0x80u + 0x40u, spread.DeviceAddress() + index * 0x40u, Abi::Read, 0});
    Require(Abi::RangesOrdered(mixed.data(), mixed.size()) && Abi::RangesOrdered(indexed.data(), indexed.size()), "the BDA range memo test tables do not ascend by begin");
    const auto index = Abi::BuildRangeIndex(indexed.data(), indexed.size());
    Require(!index.empty(), "the BDA range memo test's indexed table has no index");
    Buffer mixedTable(context, sizeof(Abi::Header) + mixed.size() * sizeof(Abi::Range), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer indexedTable(context, sizeof(Abi::Header) + indexed.size() * sizeof(Abi::Range) + index.size() * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const std::vector<BdaMemoRead> reads{{32, 1}, {32, 4}, {32, 1}, {8, 1}, {32, 2}, {32, 1}, {32, 4}, {16, 1}, {32, 1}, {32, 3}, {32, 1}, {8, 1}};
    Buffer output(context, (BdaAnchoredReadInputWord + 2u + reads.size()) * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    struct Outcome {
        Abi::Fault fault{};
        std::vector<std::uint32_t> values;
    };
    const auto run = [&](Pipeline& pipeline, bool withIndex, std::uint64_t base, const std::vector<std::uint32_t>& offsets) {
        const auto& ranges = withIndex ? indexed : mixed;
        Buffer& table = withIndex ? indexedTable : mixedTable;
        const Abi::Header header{Abi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(Abi::Range), withIndex ? Abi::IndexedTable : 0u};
        std::memset(table.Bytes().data(), 0, table.Bytes().size());
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), ranges.size() * sizeof(Abi::Range));
        if (withIndex) std::memcpy(table.Bytes().data() + sizeof(header) + ranges.size() * sizeof(Abi::Range), index.data(), index.size() * sizeof(std::uint32_t));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0xa5, BdaAnchoredReadInputWord * sizeof(std::uint32_t));
        std::vector<std::uint32_t> inputs{static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(base >> 32u)};
        inputs.insert(inputs.end(), offsets.begin(), offsets.end());
        inputs.resize(2u + reads.size(), offsets.back());
        std::memcpy(output.Bytes().data() + BdaAnchoredReadInputWord * sizeof(std::uint32_t), inputs.data(), inputs.size() * sizeof(std::uint32_t));
        pipeline.Run(1);
        Outcome outcome;
        std::memcpy(&outcome.fault, fault.Bytes().data(), sizeof(outcome.fault));
        outcome.values.resize(reads.size() * 4u);
        std::memcpy(outcome.values.data(), output.Bytes().data(), outcome.values.size() * sizeof(std::uint32_t));
        return outcome;
    };
    struct Scenario {
        const char* name;
        bool indexedTable;
        std::uint64_t base;
        std::vector<std::uint32_t> offsets;
    };
    const std::vector<Scenario> scenarios{
        {"hits in one range", false, guest + 0x400, {0x0, 0x10, 0x4, 0xb, 0x100, 0x2fc, 0x2f0, 0x2fe, 0x8, 0x20, 0x2fc, 0x2ff}},
        {"the bound before an overlapping range", false, guest + 0x400, {0x0, 0x2f0, 0x2fc, 0x2ff, 0x2f8, 0x300, 0x2f8, 0x304, 0x2fc, 0x4f4, 0x4fc, 0x4ff}},
        {"range ends and holes", false, guest, {0x100, 0x10c, 0x110, 0x0, 0x3, 0x5, 0x104, 0x200, 0x10f, 0x20c, 0x10d, 0x213}},
        {"crossing ranges after a hit", false, guest, {0x0, 0x0, 0x1, 0x2, 0x1, 0x2, 0x0, 0x2, 0x3, 0x2, 0x4, 0x5}},
        {"a range without read permission", false, guest, {0x400, 0x300, 0x404, 0x30c, 0x100, 0x400, 0x104, 0x408, 0x300, 0x40c, 0x108, 0x303}},
        {"alternating ranges", false, guest, {0x400, 0x100, 0x404, 0x104, 0x408, 0x200, 0x40c, 0x108, 0x700, 0x10, 0x104, 0x710}},
        {"the top of the address space", false, highest, {0x0, 0x0, 0xc, 0xf, 0x8, 0x10, 0x4, 0xe, 0xfffffff0u, 0x4, 0xc, 0xff}},
        {"an indexed table", true, guest, {0x80, 0x84, 0xc0, 0xbf, 0x100, 0x13c, 0x980, 0x9b0, 0x84, 0x9fc, 0x9c0, 0x0}},
        {"an indexed table, misses", true, guest, {0x0, 0x40, 0x3c, 0x7c, 0x80, 0xa00, 0x3c, 0x980, 0x9bc, 0x9c0, 0x83, 0x1}},
    };
    const auto dword = [](Buffer& buffer, std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, buffer.Bytes().data() + offset, sizeof(value));
        return value;
    };
    const auto buffers = [&](bool withIndex) { return std::array<Buffer*, 3>{withIndex ? &indexedTable : &mixedTable, &fault, &output}; };
    for (const bool stops : {false, true}) {
        const auto plainCode = MakeBdaMemoReadTestShader(reads, stops, 0u);
        Pipeline plain(context, plainCode, buffers(false));
        Pipeline plainIndexed(context, plainCode, buffers(true));
        const auto clean = run(plain, false, guest + 0x400, scenarios[0].offsets);
        const auto cleanIndexed = run(plainIndexed, true, guest, scenarios[7].offsets);
        Require(clean.fault.state == Abi::FaultState::Empty && clean.values[0] == dword(large, 0) && cleanIndexed.values[0] == dword(spread, 0x40) && cleanIndexed.fault.reason == Abi::FaultReason::Unmapped && cleanIndexed.fault.address == guest + 0xc0, "the BDA range memo test's tables do not map its reads");
        for (std::uint32_t entries = 1; entries <= 4u; ++entries) {
            const auto code = MakeBdaMemoReadTestShader(reads, stops, entries);
            Require(code != plainCode, "the BDA range memo did not change the module");
            Pipeline memo(context, code, buffers(false));
            Pipeline memoIndexed(context, code, buffers(true));
            for (const auto& scenario : scenarios) {
                const auto expected = run(scenario.indexedTable ? plainIndexed : plain, scenario.indexedTable, scenario.base, scenario.offsets);
                const auto actual = run(scenario.indexedTable ? memoIndexed : memo, scenario.indexedTable, scenario.base, scenario.offsets);
                Require(std::memcmp(&expected.fault, &actual.fault, sizeof(expected.fault)) == 0 && expected.values == actual.values, std::string("a read through the BDA range memo (") + std::to_string(entries) + " entries, " + (stops ? "stopping" : "continuing") + ") differs from the lookup: " + scenario.name);
            }
        }
    }
    Pipeline memo(context, MakeBdaMemoReadTestShader(reads, false, 2u), buffers(false));
    const auto bounded = run(memo, false, guest + 0x400, {0x0, 0x2f0, 0x2fc, 0x2ff, 0x2f8, 0x300, 0x2f8, 0x304, 0x2fc, 0x4f4, 0x4fc, 0x4ff});
    Require(bounded.fault.state == Abi::FaultState::Empty && bounded.values[0] == dword(large, 0) && bounded.values[8] == dword(large, 0x2fc) && bounded.values[16] == dword(large, 0x2f8) && bounded.values[17] == dword(large, 0x2fc) && bounded.values[20] == dword(overlap, 0) && bounded.values[24] == dword(large, 0x2f8) && bounded.values[27] == dword(large, 0x304) && bounded.values[28] == (dword(overlap, 4) & 0xffffu), "a read through the BDA range memo past the next range's begin did not take the next range: fault " + std::to_string(static_cast<std::uint32_t>(bounded.fault.state)) + " values " + Hex(bounded.values[0]) + " " + Hex(bounded.values[8]) + " " + Hex(bounded.values[16]) + " " + Hex(bounded.values[17]) + " " + Hex(bounded.values[20]) + " " + Hex(bounded.values[24]) + " " + Hex(bounded.values[27]) + " " + Hex(bounded.values[28]) + " want " + Hex(dword(large, 0)) + " " + Hex(dword(large, 0x2fc)) + " " + Hex(dword(large, 0x2f8)) + " " + Hex(dword(overlap, 0)) + " " + Hex(dword(large, 0x304)) + " " + Hex(dword(overlap, 4)));
    const auto hole = run(memo, false, guest, {0x100, 0x10c, 0x110, 0x0, 0x3, 0x5, 0x104, 0x200, 0x10f, 0x20c, 0x10d, 0x213});
    Require(hole.fault.state == Abi::FaultState::Ready && hole.fault.reason == Abi::FaultReason::Unmapped && hole.fault.address == guest + 0x110 && hole.fault.instruction == 0x1001u && hole.values[0] == dword(words, 0) && hole.values[4] == dword(words, 0xc), "a read through the BDA range memo into a hole after a hit did not fault");
}

static void RunBdaAnchorTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    constexpr std::uint64_t limit = 0xffffffff00000000ULL;
    constexpr std::uint64_t highest = 0xffffffffffffff00ULL;
    constexpr std::uint32_t lanes = 64;
    const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    Buffer first(context, 3, usage);
    Buffer second(context, 2, usage);
    Buffer words(context, 16, usage);
    Buffer shifted(context, 20, usage);
    Buffer large(context, 1024, usage);
    Buffer top(context, 16, usage);
    Buffer above(context, 16, usage);
    Buffer overlap(context, 0x200, usage);
    std::uint32_t seed = 0x11;
    for (auto* buffer : {&first, &second, &words, &shifted, &large, &top, &above, &overlap}) {
        for (std::size_t index = 0; index < buffer->Bytes().size(); ++index) buffer->Bytes()[index] = std::byte(static_cast<std::uint8_t>(seed + index * 7u));
        seed += 0x33;
    }
    for (std::uint32_t index = 0; index < 16u; ++index) words.Bytes()[index] = std::byte(static_cast<std::uint8_t>(0xa0u + index));
    for (std::uint32_t index = 0; index < 20u; ++index) shifted.Bytes()[index] = std::byte(static_cast<std::uint8_t>(0xc0u + index));
    std::vector<Abi::Range> ranges{{guest, guest + 3, first.DeviceAddress(), Abi::Read, 0},
        {guest + 3, guest + 5, second.DeviceAddress(), Abi::Read, 0},
        {guest + 0x100, guest + 0x110, words.DeviceAddress(), Abi::Read, 0},
        {guest + 0x200, guest + 0x210, shifted.DeviceAddress() + 1u, Abi::Read, 0},
        {guest + 0x400, guest + 0x800, large.DeviceAddress(), Abi::Read, 0},
        {guest + 0x700, guest + 0x900, overlap.DeviceAddress(), Abi::Read, 0},
        {limit - 0x10, limit, top.DeviceAddress(), Abi::Read, 0},
        {highest, highest + 0x10, above.DeviceAddress(), Abi::Read, 0}};
    Require(Abi::RangesOrdered(ranges.data(), ranges.size()), "the BDA anchor test table does not ascend by begin");
    Buffer table(context, sizeof(Abi::Header) + ranges.size() * sizeof(Abi::Range), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, (BdaAnchoredReadInputWord + 3u + lanes) * sizeof(std::uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    struct Outcome {
        Abi::Fault fault{};
        std::array<std::uint32_t, BdaAnchoredReadInputWord> values{};
    };
    std::uint32_t count = static_cast<std::uint32_t>(ranges.size());
    std::uint32_t permissions = Abi::Read;
    const auto run = [&](Pipeline& pipeline, std::uint32_t read, std::uint64_t anchor, const std::vector<std::uint32_t>& lows) {
        auto copy = ranges;
        if (copy.size() > 2u) copy[2].permissions = permissions;
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), copy.data(), copy.size() * sizeof(Abi::Range));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0xa5, BdaAnchoredReadInputWord * sizeof(std::uint32_t));
        std::vector<std::uint32_t> inputs{static_cast<std::uint32_t>(anchor), static_cast<std::uint32_t>(anchor >> 32u), read};
        inputs.insert(inputs.end(), lows.begin(), lows.end());
        std::memcpy(output.Bytes().data() + BdaAnchoredReadInputWord * sizeof(std::uint32_t), inputs.data(), inputs.size() * sizeof(std::uint32_t));
        pipeline.Run(1);
        Outcome outcome;
        std::memcpy(&outcome.fault, fault.Bytes().data(), sizeof(outcome.fault));
        std::memcpy(outcome.values.data(), output.Bytes().data(), sizeof(outcome.values));
        return outcome;
    };
    struct Programs {
        Programs(const Context& context, const BdaAnchoredReadProgram& reads, const std::array<Buffer*, 3>& buffers) : reads(reads), plain(context, MakeBdaAnchoredReadTestShader(reads, false), buffers), anchored(context, MakeBdaAnchoredReadTestShader(reads, true), buffers) {}
        BdaAnchoredReadProgram reads;
        Pipeline plain;
        Pipeline anchored;
    };
    const auto build = [&](const BdaAnchoredReadProgram& reads) { return std::make_unique<Programs>(context, reads, std::array<Buffer*, 3>{&table, &fault, &output}); };
    const auto compare = [&](Programs& programs, std::initializer_list<std::uint64_t> anchors, const std::vector<std::uint32_t>& lows) {
        std::vector<std::vector<std::uint32_t>> batches;
        if (programs.reads.lanes == 1u) {
            for (const auto low : lows) batches.push_back({low});
        } else {
            batches.push_back(lows);
            batches.back().resize(programs.reads.lanes, lows.back());
        }
        for (std::uint32_t index = 0; index < programs.reads.reads.size(); ++index) {
            const auto& read = programs.reads.reads[index];
            for (const auto anchor : anchors) {
                for (const auto& batch : batches) {
                    const auto expected = run(programs.plain, index, anchor, batch);
                    const auto actual = run(programs.anchored, index, anchor, batch);
                    const bool faultsMatch = programs.reads.lanes == 1u ? std::memcmp(&expected.fault, &actual.fault, sizeof(expected.fault)) == 0 : expected.fault.state == actual.fault.state;
                    Require(faultsMatch && expected.values == actual.values, "an anchored BDA read differs from the same read through the lookups at anchor " + Hex(anchor) + " + " + std::to_string(batch[0]) + " + (" + std::to_string(read.immediate) + "), " + std::to_string(read.bits) + " bits, " + std::to_string(read.dwords) + " dwords");
                }
            }
        }
    };
    std::vector<BdaAnchoredRead> reads;
    for (const std::int32_t immediate : {0, 4, -1, -8, 0x7ffffff0, std::numeric_limits<std::int32_t>::min()}) {
        for (const auto& [bits, dwords, used] : std::initializer_list<std::array<std::uint32_t, 3>>{{8, 1, 1}, {16, 1, 1}, {32, 1, 1}, {32, 4, 15}, {32, 4, 10}, {32, 3, 4}}) reads.push_back({immediate, bits, dwords, used});
    }
    const auto anchors = {guest + 0x100, guest + 0x108, guest, guest + 3, guest + 0x200, guest - 1, guest + 0x110, guest + 0x400, guest + 0x6f0, guest + 0x700, guest + 0x800, limit - 0x10, limit - 4, highest, highest + 8, std::numeric_limits<std::uint64_t>::max() - 2};
    std::vector<std::uint32_t> lows{0u, 1u, 2u, 3u, 4u, 8u, 0xcu, 0xdu, 0xeu, 0xfu, 0x10u, 0x14u, 0x100u, 0x108u, 0x10cu, 0x1fcu, 0x200u, 0x20eu, 0x3f0u, 0x3fcu, 0x400u, 0x7fffu, 0x7fffffffu, 0x80000000u, 0xfffffff0u, 0xfffffffcu, 0xffffffffu};
    for (std::uint32_t lane = 0; lane < 32u; ++lane) lows.push_back(lane * 0x24u);
    std::vector<std::uint32_t> strided;
    for (std::uint32_t lane = 0; lane < lanes; ++lane) strided.push_back(lane * 16u);
    auto halfword = build({{{-1, 16, 1, 1}}, false, false, false, lanes});
    compare(*halfword, {guest + 0x100}, {1u});
    compare(*halfword, {guest + 0x100}, {0u, 1u});
    auto continuing = build({reads, false, false, false, lanes});
    compare(*continuing, anchors, lows);
    compare(*continuing, {guest + 0x400}, strided);
    auto elected = build({reads, false, true, false, lanes});
    compare(*elected, anchors, lows);
    compare(*elected, {guest + 0x400}, strided);
    auto coherent = build({{reads[0], reads[2], reads[9], reads[16], reads[22]}, false, false, true, lanes});
    compare(*coherent, anchors, lows);
    auto stopping = build({reads, true, true, false, 1});
    compare(*stopping, anchors, {0u, 0x80000000u, 0xfffffffcu});
    const auto check = [&](std::uint32_t read, std::uint64_t anchor, std::uint32_t low) { return run(stopping->anchored, read, anchor, {low}); };
    auto direct = std::make_unique<Pipeline>(context, MakeBdaAnchoredReadTestShader({reads, true, false, false, 1, true}, true), std::array<Buffer*, 3>{&table, &fault, &output});
    const auto dword = [](Buffer& buffer, std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, buffer.Bytes().data() + offset, sizeof(value));
        return value;
    };
    const auto loads = [&](std::uint32_t read, std::uint64_t anchor, std::uint32_t low, std::initializer_list<std::uint32_t> values) {
        const auto outcome = run(*direct, read, anchor, {low});
        Require(outcome.fault.state == Abi::FaultState::Empty && std::equal(values.begin(), values.end(), outcome.values.begin()), "a read the anchor holds did not load directly at anchor " + Hex(anchor) + " + " + std::to_string(low) + ", read " + std::to_string(read));
    };
    const auto misses = [&](std::uint32_t read, std::uint64_t anchor, std::uint32_t low) {
        const auto outcome = run(*direct, read, anchor, {low});
        Require(outcome.fault.state == Abi::FaultState::Empty && outcome.values[0] == BdaAnchoredReadSentinel, "a read the anchor does not hold did not take the lookup path at anchor " + Hex(anchor) + " + " + std::to_string(low) + ", read " + std::to_string(read));
    };
    loads(2, guest + 0x100, 0, {dword(words, 0)});
    loads(2, guest + 0x100, 0xc, {dword(words, 0xc)});
    loads(3, guest + 0x100, 0, {dword(words, 0), dword(words, 4), dword(words, 8), dword(words, 0xc)});
    loads(0, guest + 0x200, 0xd, {0xceu});
    loads(0, guest + 3, 1, {static_cast<std::uint32_t>(second.Bytes()[1])});
    loads(1, guest, 1, {static_cast<std::uint32_t>(first.Bytes()[1]) | static_cast<std::uint32_t>(first.Bytes()[2]) << 8u});
    loads(2, guest + 0x400, 0x2fc, {dword(large, 0x2fc)});
    loads(2, guest + 0x700, 0x1fc, {dword(overlap, 0x1fc)});
    loads(2, limit - 0x10, 0xc, {dword(top, 0xc)});
    misses(2, guest + 0x100, 0xd);
    misses(3, guest + 0x100, 4);
    misses(2, guest + 0x200, 0);
    misses(1, guest + 2, 0);
    misses(0, guest + 3, 2);
    misses(2, guest + 0x400, 0x300);
    misses(2, guest + 0x400, 0x2fd);
    misses(2, limit - 0x10, 0xd);
    misses(2, highest, 0);
    misses(2, guest - 1, 1);
    const auto wide = check(3, guest + 0x100, 0);
    Require(wide.fault.state == Abi::FaultState::Empty && wide.values[0] == 0xa3a2a1a0u && wide.values[1] == 0xa7a6a5a4u && wide.values[2] == 0xabaaa9a8u && wide.values[3] == 0xafaeadacu, "an anchored span read returned wrong data");
    const auto some = check(4, guest + 0x100, 0);
    Require(some.fault.state == Abi::FaultState::Empty && some.values[0] == 0u && some.values[1] == 0xa7a6a5a4u && some.values[2] == 0u && some.values[3] == 0xafaeadacu, "an anchored read of some dwords returned wrong data");
    const auto odd = check(2, guest + 0x200, 0);
    Require(odd.fault.state == Abi::FaultState::Empty && odd.values[0] == 0xc4c3c2c1u, "an anchored read of an odd device address returned wrong data");
    const auto byte = check(0, guest + 0x200, 0xd);
    Require(byte.fault.state == Abi::FaultState::Empty && byte.values[0] == 0xceu, "an anchored byte read returned wrong data");
    const auto below = check(20, guest + 0x108, 4);
    Require(below.fault.state == Abi::FaultState::Empty && below.values[0] == 0xa7a6a5a4u, "an anchored read below its base returned wrong data");
    const auto unmapped = check(2, guest + 0x100, 0x10);
    Require(unmapped.fault.state == Abi::FaultState::Ready && unmapped.fault.reason == Abi::FaultReason::Unmapped && unmapped.fault.address == guest + 0x110 && unmapped.fault.instruction == 0x1234 && unmapped.values[0] == 0xa5a5a5a5u, "an anchored read past its range did not fault");
    permissions = 0;
    misses(2, guest + 0x100, 0);
    compare(*stopping, {guest + 0x100}, {0u, 4u});
    compare(*elected, {guest + 0x100}, lows);
    permissions = Abi::Read;
    count = 3;
    misses(2, guest + 0x100, 0);
    misses(0, guest, 0);
    compare(*stopping, {guest + 0x100, guest}, {0u, 4u});
    compare(*elected, {guest + 0x100, guest}, lows);
    ranges = {{0, limit, large.DeviceAddress(), Abi::Read, 0}};
    count = 1;
    compare(*stopping, {limit - 1}, {0xffffffffu});
    ranges = {{0x100000, 0x100010, top.DeviceAddress(), Abi::Read, 0}};
    for (std::uint64_t index = 1; index < 8u; ++index) ranges.push_back({0x10000000 + index * 0x1000, 0x10000010 + index * 0x1000, above.DeviceAddress(), Abi::Read, 0});
    count = static_cast<std::uint32_t>(ranges.size());
    std::vector<std::uint32_t> mixed(lanes, 0x100u);
    mixed[26] = 0x1u;
    mixed[31] = 0x110u;
    mixed[40] = 0x10cu;
    const auto divergent = run(continuing->plain, 2, 0xfff00, mixed);
    Require(divergent.fault.state == Abi::FaultState::Ready && divergent.fault.reason == Abi::FaultReason::Unmapped, "a subgroup reading below the first range, past a range's end and inside it did not record the unmapped read");
    for (std::uint32_t lane = 0; lane < lanes; ++lane) {
        const auto expected = lane == 26u || lane == 31u ? 0u : dword(top, lane == 40u ? 0xcu : 0u);
        Require(divergent.values[lane * 4u] == expected, "a subgroup reading below the first range, past a range's end and inside it returned a wrong value in lane " + std::to_string(lane));
    }
    compare(*continuing, {0xfff00}, mixed);
}

void RunBdaExecutionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer first(context, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    first.Bytes()[0] = std::byte{0x11};
    first.Bytes()[1] = std::byte{0x22};
    first.Bytes()[2] = std::byte{0x33};
    second.Bytes()[0] = std::byte{0x44};
    second.Bytes()[1] = std::byte{0x55};
    constexpr std::uint64_t guest = 0x7fff12340001ULL;
    std::array<Abi::Range, 2> ranges{{{guest, guest + 3, first.DeviceAddress(), Abi::Read, 0}, {guest + 3, guest + 5, second.DeviceAddress(), Abi::Read, 0}}};
    Buffer table(context, sizeof(Abi::Header) + sizeof(ranges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, sizeof(Abi::Fault), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto run = [&](std::uint64_t address, std::uint32_t bits, std::uint32_t expected, Abi::FaultReason reason, std::uint32_t count = 2, std::uint32_t groups = 1, std::int64_t offset = 0) {
        const Abi::Header header{Abi::Version, count, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::uint32_t sentinel = 0xdeadbeef;
        std::memcpy(output.Bytes().data(), &sentinel, sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaTestShader(address, bits, offset), {&table, &fault, &output});
        pipeline.Run(groups);
        Abi::Fault report{};
        std::uint32_t result = 0;
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(&result, output.Bytes().data(), sizeof(result));
        if (static_cast<std::uint32_t>(reason) == 0) {
            Require(report.state == Abi::FaultState::Empty && result == expected, "BDA GPU read produced incorrect data or a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.instruction == 0x1234, "BDA GPU fault was not published correctly");
            Require(result == sentinel, "faulting BDA shader continued to output a substitute value");
        }
    };
    run(guest, 8, 0x11, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 16, 0x3322, static_cast<Abi::FaultReason>(0));
    run(guest + 1, 32, 0x55443322, static_cast<Abi::FaultReason>(0));
    run(guest + 5, 8, 0, Abi::FaultReason::Unmapped, 2, 64);
    run(guest - 1, 8, 0, Abi::FaultReason::Unmapped);
    run(std::numeric_limits<std::uint64_t>::max() - 1, 32, 0, Abi::FaultReason::Overflow);
    run(guest, 8, 0, Abi::FaultReason::InvalidTable, 3);
    run(std::numeric_limits<std::uint64_t>::max() - 2, 8, 0, Abi::FaultReason::Overflow, 2, 1, 4);
    run(1, 8, 0, Abi::FaultReason::Overflow, 2, 1, -4);
    Buffer words(context, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto read = [&](std::uint64_t address, std::uint32_t dwords, bool coherent, bool stops, std::uint32_t value, bool faults) {
        const Abi::Header header{Abi::Version, 2, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaDwordReadTestShader(address, dwords, coherent, stops), {&table, &fault, &words});
        pipeline.Run(1);
        Abi::Fault report{};
        std::array<std::uint32_t, 4> result{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(result.data(), words.Bytes().data(), sizeof(result));
        auto expected = sentinel;
        if (!faults || !stops) {
            expected[0] = value;
            for (std::uint32_t dword = 1; dword < dwords; ++dword) expected[dword] = 0;
        }
        if (faults) {
            Require(report.state == Abi::FaultState::Ready && report.reason == Abi::FaultReason::Unmapped && report.address == guest + 5 && report.bytes == 1 && report.instruction == 0x1234, "BDA dword read did not publish its first fault");
        } else {
            Require(report.state == Abi::FaultState::Empty, "mapped BDA dword read published a fault");
        }
        Require(result == expected, faults && stops ? "faulting BDA dword read continued to output a substitute value" : "BDA dword read produced incorrect data");
    };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            read(guest + 1, 1, coherent, stops, 0x55443322, false);
            read(guest + 3, 1, coherent, stops, 0x5544, true);
            read(guest + 1, 4, coherent, stops, 0x55443322, true);
        }
    }
    Buffer wide(context, 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    for (std::uint32_t dword = 0; dword < 8u; ++dword) {
        const std::uint32_t value = 0xa0b0c000u + dword;
        std::memcpy(wide.Bytes().data() + dword * 4u, &value, sizeof(value));
    }
    constexpr std::uint64_t spanGuest = 0x7fff56780000ULL;
    const std::array<Abi::Range, 3> spanRanges{{ranges[0], ranges[1], {spanGuest, spanGuest + 32, wide.DeviceAddress(), Abi::Read, 0}}};
    Buffer spanTable(context, sizeof(Abi::Header) + sizeof(spanRanges), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    const auto span = [&](std::uint64_t address, std::int32_t offset, std::uint32_t extracted, bool coherent, bool stops, std::array<std::uint32_t, 4> values, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const Abi::Header header{Abi::Version, 3, sizeof(Abi::Range), 0};
        std::memcpy(spanTable.Bytes().data(), &header, sizeof(header));
        std::memcpy(spanTable.Bytes().data() + sizeof(header), spanRanges.data(), sizeof(spanRanges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        const std::array<std::uint32_t, 4> sentinel{0xdeadbeef, 0xdeadbeef, 0xdeadbeef, 0xdeadbeef};
        std::memcpy(words.Bytes().data(), sentinel.data(), sizeof(sentinel));
        Pipeline pipeline(context, MakeBdaSpanReadTestShader(address, static_cast<std::uint32_t>(offset), extracted, coherent, stops), {&spanTable, &fault, &words});
        pipeline.Run(1);
        Abi::Fault report{};
        std::array<std::uint32_t, 4> result{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        std::memcpy(result.data(), words.Bytes().data(), sizeof(result));
        const bool faults = static_cast<std::uint32_t>(reason) != 0u;
        if (faults) {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.instruction == 0x1234, "BDA span read did not publish its first fault");
        } else {
            Require(report.state == Abi::FaultState::Empty, "mapped BDA span read published a fault");
        }
        Require(result == (faults && stops ? sentinel : values), faults && stops ? "faulting BDA span read continued to output a substitute value" : "BDA span read produced incorrect data");
    };
    const auto at = [&](std::uint32_t dword) { return 0xa0b0c000u + dword; };
    for (const bool stops : {true, false}) {
        for (const bool coherent : {false, true}) {
            span(spanGuest + 16, -12, 0b1010u, coherent, stops, {0, at(2), 0, at(4)});
            span(spanGuest, 4, 0b0101u, coherent, stops, {at(1), 0, at(3), 0});
            span(spanGuest + 24, 0, 0b0011u, coherent, stops, {at(6), at(7), 0, 0});
            span(spanGuest + 20, 8, 0b0001u, coherent, stops, {at(7), 0, 0, 0});
            span(spanGuest + 28, 0, 0b0101u, coherent, stops, {at(7), 0, 0, 0}, Abi::FaultReason::Unmapped, spanGuest + 36, 1);
            span(guest, 1, 0b1001u, coherent, stops, {0x55443322, 0, 0, 0}, Abi::FaultReason::Unmapped, guest + 13, 1);
            span(4, -8, 0b0001u, coherent, stops, {0, 0, 0, 0}, Abi::FaultReason::Overflow, 4, 0);
        }
    }
    ranges[0].permissions = 0;
    run(guest, 8, 0, Abi::FaultReason::Permission);
    const auto sequence = [&](std::initializer_list<Abi::FaultReason> reasons, Abi::FaultReason recorded, std::uint32_t otherFaults) {
        const Abi::Header header{Abi::Version, 2, sizeof(Abi::Range), 0};
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        Pipeline pipeline(context, MakeBdaFaultSequenceTestShader(reasons), {&table, &fault, &output});
        pipeline.Run(1);
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        Require(report.state == Abi::FaultState::Ready && report.reason == recorded && report.otherFaults == otherFaults, "BDA faults of one dispatch were not recorded with the first reason and the sticky other-fault flag");
    };
    sequence({Abi::FaultReason::Unaligned, Abi::FaultReason::Unaligned}, Abi::FaultReason::Unaligned, 0);
    sequence({Abi::FaultReason::Unaligned, Abi::FaultReason::Unmapped}, Abi::FaultReason::Unaligned, 1);
    sequence({Abi::FaultReason::Unmapped, Abi::FaultReason::Unaligned}, Abi::FaultReason::Unmapped, 1);
    RunBdaDwordWriteTests(context);
    RunBdaAnchorTests(context);
    RunBdaRangeMemoTests(context);
}

void RunBdaStoreExecutionTests(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    Buffer first(context, 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer second(context, 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    Buffer table(context, sizeof(Abi::Header) + 2 * sizeof(Abi::Range), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer fault(context, Abi::FaultBufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer output(context, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    constexpr std::uint64_t guest = 0x7fff1235fffdULL;
    constexpr std::uint32_t writable = Abi::Read | Abi::Write;
    const Abi::Header header{Abi::Version, 2, sizeof(Abi::Range), 0};
    std::array<Abi::Range, 2> ranges{};
    bool runtime = false;
    const auto run = [&](const char* name, std::uint64_t address, std::uint32_t bits, bool coherent, const std::array<std::uint8_t, 5>& expected, std::initializer_list<std::uint64_t> writtenAddresses, Abi::FaultReason reason = {}, std::uint64_t faultAddress = 0, std::uint32_t faultBytes = 0) {
        const std::string label = std::string(runtime ? "BDA runtime write " : "BDA store ") + name + (coherent ? " coherent: " : ": ");
        std::memset(first.Bytes().data(), 0xcd, first.Bytes().size());
        std::memset(second.Bytes().data(), 0xcd, second.Bytes().size());
        std::memcpy(table.Bytes().data(), &header, sizeof(header));
        std::memcpy(table.Bytes().data() + sizeof(header), ranges.data(), sizeof(ranges));
        std::memset(fault.Bytes().data(), 0, fault.Bytes().size());
        std::memset(output.Bytes().data(), 0, output.Bytes().size());
        Pipeline pipeline(context, runtime ? MakeBdaWriteTestShader(address, 0xa1b2c3d4u, bits, coherent) : MakeBdaStoreTestShader(address, 0xa1b2c3d4u, bits, coherent), {&table, &fault, &output});
        pipeline.Run(1);
        std::array<std::uint8_t, 5> actual{};
        std::memcpy(actual.data(), first.Bytes().data(), 2);
        std::memcpy(actual.data() + 2, second.Bytes().data(), 3);
        Require(actual == expected, label + "incorrect stored bytes or writes outside the mapped span");
        Abi::Fault report{};
        std::memcpy(&report, fault.Bytes().data(), sizeof(report));
        if (reason == Abi::FaultReason{}) {
            Require(report.state == Abi::FaultState::Empty && report.reason == Abi::FaultReason{} && report.address == 0 && report.bytes == 0 && report.stage == 0 && report.instruction == 0 && report.otherFaults == 0, label + "mapped store published a fault");
        } else {
            Require(report.state == Abi::FaultState::Ready && report.reason == reason && report.address == faultAddress && report.bytes == faultBytes && report.stage == static_cast<std::uint32_t>(ShaderRecompiler::IrShaderStage::Compute) && report.instruction == 0x1234 && report.otherFaults == 1, label + "incorrect first fault metadata");
        }
        std::array<std::uint32_t, Abi::FaultBufferBytes / sizeof(std::uint32_t)> words{};
        std::memcpy(words.data(), fault.Bytes().data(), sizeof(words));
        Require(words[Abi::WrittenOverflowWord] == 0, label + "written-page table overflowed");
        std::set<std::uint64_t> recordedPages;
        Require(Abi::ForEachWrittenRange(words.data(), [&](std::uint64_t address, std::size_t bytes) {
            for (std::size_t offset = 0; offset < bytes; offset += std::size_t{1} << Abi::WrittenPageShift) {
                Require(recordedPages.insert((address + offset) >> Abi::WrittenPageShift).second, label + "written-page table contains duplicate pages");
            }
        }), label + "invalid written-page record");
        std::set<std::uint64_t> expectedPages;
        std::set<std::uint32_t> expectedChunks;
        for (const auto writtenAddress : writtenAddresses) {
            expectedPages.insert(writtenAddress >> Abi::WrittenPageShift);
            expectedChunks.insert(Abi::WrittenChunkKey(writtenAddress));
        }
        Require(recordedPages == expectedPages, label + "incorrect written-page set");
        Require(words[Abi::WrittenCountWord] == expectedChunks.size(), label + "incorrect written-chunk claim count");
        std::uint32_t completed = 0;
        std::memcpy(&completed, output.Bytes().data(), sizeof(completed));
        Require(completed == 1, label + "store did not finish its per-byte accesses");
    };
    for (const auto& [coherent, writes] : std::initializer_list<std::pair<bool, bool>>{{false, false}, {false, true}, {true, false}, {true, true}}) {
        runtime = writes;
        ranges = {{{guest, guest + 2, first.DeviceAddress(), writable, 0}, {guest + 2, guest + 5, second.DeviceAddress(), writable, 0}}};
        run("byte", guest + 1, 8, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1});
        run("split short", guest + 1, 16, coherent, {0xcd, 0xd4, 0xc3, 0xcd, 0xcd}, {guest + 1, guest + 2});
        run("split unaligned dword", guest, 32, coherent, {0xd4, 0xc3, 0xb2, 0xa1, 0xcd}, {guest, guest + 3});
        ranges[1].permissions = Abi::Read;
        run("read-only byte", guest + 2, 8, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only short suffix", guest + 1, 16, coherent, {0xcd, 0xd4, 0xcd, 0xcd, 0xcd}, {guest + 1}, Abi::FaultReason::Permission, guest + 2, 1);
        run("read-only suffix", guest, 32, coherent, {0xd4, 0xc3, 0xcd, 0xcd, 0xcd}, {guest}, Abi::FaultReason::Permission, guest + 2, 1);
        ranges[0].end = guest + 1;
        ranges[1].permissions = writable;
        run("unmapped middle byte", guest, 32, coherent, {0xd4, 0xcd, 0xb2, 0xa1, 0xcd}, {guest, guest + 3}, Abi::FaultReason::Unmapped, guest + 1, 1);
        ranges = {{{guest + 3, guest + 5, first.DeviceAddress(), writable, 0}, {guest + 5, guest + 8, second.DeviceAddress(), writable, 0}}};
        run("split aligned dword", guest + 3, 32, coherent, {0xcd, 0xcd, 0xcd, 0xcd, 0xcd}, {}, Abi::FaultReason::Unmapped, guest + 3, 4);
    }
}

// Lane P0-shader E3 (diagnostic, run only by `agc_driver_bda_device_tests --p0s-lookup-bench`): the BDA lookup schemes
// in isolation. P0S_LOOKUP_SPV names the compiled research/speed/newpath/p0s/e3/lookup.comp; P0S_RANGES (5000),
// P0S_GROUPS (4096 x 256 invocations), P0S_LOOKUPS (32 per invocation), P0S_SEED. One table buffer holds the range
// table + index, the flat page section and the multi-level model; prints one line per mode x chain x table memory.
void RunP0sLookupBench(const Context& context) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    const auto env = [](const char* name, std::uint64_t fallback) { const char* text = std::getenv(name); return text != nullptr ? std::strtoull(text, nullptr, 0) : fallback; };
    const char* spvPath = std::getenv("P0S_LOOKUP_SPV");
    Require(spvPath != nullptr, "P0S_LOOKUP_SPV is not set");
    std::FILE* file = std::fopen(spvPath, "rb");
    Require(file != nullptr, "cannot open P0S_LOOKUP_SPV");
    std::vector<std::uint32_t> code;
    for (std::uint32_t word = 0; std::fread(&word, 4, 1, file) == 1;) code.push_back(word);
    std::fclose(file);
    const auto rangeCount = static_cast<std::size_t>(env("P0S_RANGES", 5000));
    const auto groups = static_cast<std::uint32_t>(env("P0S_GROUPS", 4096));
    const auto lookups = static_cast<std::uint32_t>(env("P0S_LOOKUPS", 32));
    std::uint64_t seed = env("P0S_SEED", 1);
    const auto next = [&] { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; };
    const auto logUniform = [&](double low, double high) { const double u = static_cast<double>(next() % 1000000u) / 1e6; return static_cast<std::uint64_t>(low * std::pow(high / low, u)); };
    // Disjoint ranges, sizes log-uniform 4 KiB - 16 MiB, gaps 0 (1 in 4) or log-uniform up to 1 MiB, 256-byte aligned.
    std::vector<Abi::Range> ranges;
    std::uint64_t cursor = 0x8000000000ull, device = 0x100000000ull;
    for (std::size_t index = 0; index < rangeCount; ++index) {
        if (next() % 4u != 0u) cursor += logUniform(256.0, 1048576.0) & ~std::uint64_t{255};
        const auto size = std::max<std::uint64_t>(256u, logUniform(4096.0, 16777216.0) & ~std::uint64_t{255});
        ranges.push_back({cursor, cursor + size, device, next() % 3u == 0u ? Abi::Read | Abi::Write : Abi::Read, 0u});
        cursor += size;
        device += size + 0x10000u;
    }
    const auto index = Abi::BuildRangeIndex(ranges.data(), ranges.size());
    Require(!index.empty(), "the E3 table has no index");
    std::vector<std::uint32_t> words(4u + 8u * ranges.size());
    const Abi::Header header{Abi::Version, static_cast<std::uint32_t>(ranges.size()), sizeof(Abi::Range), Abi::IndexedTable};
    std::memcpy(words.data(), &header, sizeof(header));
    std::memcpy(words.data() + 4, ranges.data(), ranges.size() * sizeof(Abi::Range));
    words.insert(words.end(), index.begin(), index.end());
    // Flat section (BdaResources p0sFlatSection's rule, at most 2^18 pages).
    const auto flatAt = static_cast<std::uint32_t>(words.size());
    std::uint32_t shift = 16;
    const auto first = ranges.front().begin, last = ranges.back().end;
    while (((last - (first & ~((std::uint64_t{1} << shift) - 1u)) + (std::uint64_t{1} << shift) - 1u) >> shift) > (std::uint64_t{1} << 18u)) ++shift;
    const auto flatBase = first & ~((std::uint64_t{1} << shift) - 1u);
    const auto pages = static_cast<std::uint32_t>((last - flatBase + (std::uint64_t{1} << shift) - 1u) >> shift);
    words.resize(words.size() + 4u + 2u * std::size_t{pages}, 0u);
    std::memcpy(words.data() + flatAt, &flatBase, 8);
    words[flatAt + 2] = pages;
    words[flatAt + 3] = shift;
    std::size_t following = 0, flatEntries = 0;
    for (std::uint32_t page = 0; page < pages; ++page) {
        const auto start = flatBase + (std::uint64_t{page} << shift), stop = start + (std::uint64_t{1} << shift);
        while (following < ranges.size() && ranges[following].begin <= start) ++following;
        if (following == 0 || (following < ranges.size() && ranges[following].begin < stop)) continue;
        const auto& range = ranges[following - 1u];
        if (range.end < stop || (range.permissions & Abi::Read) == 0u) continue;
        const std::uint64_t delta = range.deviceAddress - range.begin;
        std::memcpy(words.data() + flatAt + 4u + 2u * page, &delta, 8);
        ++flatEntries;
    }
    // Multi-level model: root word per 4 GiB window (leaf word index, 0 = none), leaf = 65536 16-byte entries.
    const auto rootFirst = static_cast<std::uint32_t>(first >> 32u), rootCount = static_cast<std::uint32_t>(((last - 1u) >> 32u) - rootFirst + 1u);
    const auto rootAt = static_cast<std::uint32_t>(words.size());
    words.resize(words.size() + rootCount, 0u);
    std::size_t segments = 0;
    for (std::uint32_t window = 0; window < rootCount; ++window) {
        const auto leaf = static_cast<std::uint32_t>(words.size());
        words[rootAt + window] = leaf;
        words.resize(words.size() + 65536u * 4u, 0u);
        const auto windowBase = std::uint64_t{rootFirst + window} << 32u;
        auto cursorRange = std::upper_bound(ranges.begin(), ranges.end(), windowBase, [](std::uint64_t value, const Abi::Range& range) { return value < range.end; });
        for (std::uint32_t page = 0; page < 65536u; ++page) {
            const auto start = windowBase + (std::uint64_t{page} << 16u), stop = start + 0x10000u;
            while (cursorRange != ranges.end() && cursorRange->end <= start) ++cursorRange;
            if (cursorRange == ranges.end() || cursorRange->begin >= stop) continue;
            const auto after = cursorRange + 1;
            if (after != ranges.end() && after->begin < stop) continue;  // two ranges in the page: Split in the real table
            if ((cursorRange->permissions & Abi::Read) == 0u) continue;
            const auto lo = static_cast<std::uint32_t>(std::max(cursorRange->begin, start) - start), hi = static_cast<std::uint32_t>(std::min(cursorRange->end, stop) - start);
            const std::uint64_t target = cursorRange->deviceAddress - cursorRange->begin;
            const auto at = leaf + page * 4u;
            std::memcpy(words.data() + at, &target, 8);
            words[at + 2] = lo | (1u << 18u) | (1u << 30u);
            words[at + 3] = hi;
            ++segments;
        }
    }
    // 65536 sample addresses: a uniformly chosen range, a uniform dword in it.
    constexpr std::uint32_t Samples = 65536;
    std::vector<std::uint64_t> samples(Samples);
    std::size_t flatSampleHits = 0, levelSampleHits = 0;
    for (auto& sample : samples) {
        const auto& range = ranges[next() % ranges.size()];
        sample = range.begin + ((next() % ((range.end - range.begin) / 4u)) * 4u);
        const auto flatPage = (sample - flatBase) >> shift;
        std::uint64_t delta = 0;
        std::memcpy(&delta, words.data() + flatAt + 4u + 2u * flatPage, 8);
        if (delta != 0u) ++flatSampleHits;
        const auto leaf = words[rootAt + static_cast<std::uint32_t>(sample >> 32u) - rootFirst];
        const auto at = leaf + static_cast<std::uint32_t>((sample >> 16u) & 0xffffu) * 4u;
        const auto offset = static_cast<std::uint32_t>(sample & 0xffffu);
        if ((words[at + 2] >> 30u) == 1u && offset >= (words[at + 2] & 0x1ffffu) && offset + 4u <= words[at + 3]) ++levelSampleHits;
    }
    std::printf("P0S-E3 table: %zu ranges over %.2f GiB, %u flat pages of %u KiB (%zu mapped), %u windows (%zu segment pages), %.1f MiB; sample hits flat %.1f %%, levels %.1f %%\n", ranges.size(), static_cast<double>(last - first) / 1073741824.0, pages, (1u << shift) >> 10u, flatEntries, rootCount, segments, static_cast<double>(words.size()) * 4.0 / 1048576.0, 100.0 * static_cast<double>(flatSampleHits) / Samples, 100.0 * static_cast<double>(levelSampleHits) / Samples);
    const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkMemoryPropertyFlags deviceVisible = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer params(context, 64 + Samples * 8u, usage, deviceVisible);
    const auto invocations = std::size_t{groups} * 256u;
    Buffer output(context, invocations * 4u, usage, deviceVisible);
    std::memcpy(params.Bytes().data() + 64, samples.data(), Samples * 8u);
    for (const bool deviceTable : {false, true}) {
        Buffer table(context, words.size() * 4u, usage, deviceTable ? deviceVisible : VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT});
        std::memcpy(table.Bytes().data(), words.data(), words.size() * 4u);
        Pipeline pipeline(context, code, {&table, &params, &output});
        for (const std::uint32_t chain : {0u, 1u}) {
            for (const std::uint32_t mode : {0u, 1u, 2u}) {
                const std::array<std::uint32_t, 8> fields{mode, lookups, chain, Samples, flatAt, rootAt, rootFirst, rootCount};
                std::memcpy(params.Bytes().data(), fields.data(), sizeof(fields));
                pipeline.Run(groups);
                double best = 1e30;
                for (int round = 0; round < 7; ++round) {
                    const auto started = std::chrono::steady_clock::now();
                    pipeline.Run(groups);
                    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
                }
                std::uint32_t checksum = 0;
                for (std::size_t word = 0; word < invocations; ++word) {
                    std::uint32_t value = 0;
                    std::memcpy(&value, output.Bytes().data() + word * 4u, 4);
                    checksum = checksum * 31u + value;
                }
                std::printf("P0S-E3 table=%s chain=%u mode=%s best_ms=%.3f ns_per_lookup=%.3f checksum=%08x\n", deviceTable ? "device" : "host", chain, mode == 0u ? "range" : mode == 1u ? "flat" : "levels", best, best * 1e6 / (static_cast<double>(invocations) * lookups), checksum);
                std::fflush(stdout);
            }
        }
    }
}
