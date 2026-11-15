#include "prx/libSceAgcDriver/Graphics/include/DrawBlock.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockVerify.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "Optimization/ShaderStageInputInfo.hpp"
#include "BlockAbi.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics {

namespace {

namespace Abi = ShaderRecompiler::BlockAbi;

const ShaderRecompiler::DescriptorBinding* bindingOf(const ShaderRecompiler::RecompileResult& program, ShaderRecompiler::DescriptorRole role) {
    for (const auto& binding : program.bindings) {
        if (binding.role == role) return &binding;
    }
    return nullptr;
}

bool classicBuffers(const ShaderRecompiler::RecompileResult& program) {
    return !program.blockModule && (bindingOf(program, ShaderRecompiler::DescriptorRole::GuestBuffers) != nullptr || bindingOf(program, ShaderRecompiler::DescriptorRole::FlattenedSrt) != nullptr);
}

void writeEntry(std::uint32_t* entry, const Core::BufferAnswer& answer, std::uint32_t misalignment) {
    if (answer.deviceAddress == 0 || answer.bytes == 0) {
        entry[Abi::EntryAddressLow] = 0;
        entry[Abi::EntryAddressHigh] = 0;
        entry[Abi::EntryBytes] = 0;
        entry[Abi::EntryService] = Abi::ServiceAbsent;
        return;
    }
    entry[Abi::EntryAddressLow] = static_cast<std::uint32_t>(answer.deviceAddress);
    entry[Abi::EntryAddressHigh] = static_cast<std::uint32_t>(answer.deviceAddress >> 32u);
    entry[Abi::EntryBytes] = answer.bytes;
    entry[Abi::EntryService] = misalignment & Abi::ServiceMisalignmentMask;
}

void report() {
    if (!DrawBlockStats()) return;
    static std::atomic<std::int64_t> lastReport{0};
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = lastReport.load(std::memory_order_relaxed);
    if (nowMs - last < 5000 || !lastReport.compare_exchange_strong(last, nowMs)) return;
    std::fprintf(stderr, "%s%c", Core::DrawBlockHost::Report().c_str(), 10);
}

struct StageTruth {
    std::vector<Core::BufferTruth> buffers;
    std::vector<std::vector<std::byte>> bytes;
};

void verifyStage(const ShaderRecompiler::RecompileResult& program, std::span<const std::uint32_t> section, std::span<const std::uint32_t> flat, std::span<const Core::BufferRequest> requests, Core::DrawBlockHost::Draw& draw) {
    const auto& layout = program.blockLayout;
    StageTruth truth;
    truth.buffers.resize(requests.size());
    truth.bytes.resize(requests.size());
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto* entry = section.data() + Abi::BufferEntryWord(layout.words, static_cast<std::uint32_t>(index));
        const auto found = static_cast<std::uint64_t>(entry[Abi::EntryAddressLow]) | (static_cast<std::uint64_t>(entry[Abi::EntryAddressHigh]) << 32u);
        const auto& request = requests[index];
        if (!request.present) continue;
        auto& bytes = truth.bytes[index];
        const auto wanted = static_cast<std::size_t>(request.bytes);
        const auto padded = ((wanted + request.misalignment + 3u) & ~std::size_t{3}) - request.misalignment;
        bytes.resize(padded);
        bool read = true;
        try {
            GuestMemory::Read(request.address, std::span<std::byte>(bytes.data(), wanted));
        } catch (const std::exception&) {
            read = false;
        }
        if (!read) continue;
        if (padded > wanted && !Core::ReadGuestBytes(request.address + wanted, std::span<std::byte>(bytes.data() + wanted, padded - wanted))) std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(wanted), bytes.end(), std::byte{0});
        truth.buffers[index] = {found, std::span<const std::byte>(bytes), true};
    }
    const auto mismatches = Core::VerifyBlock(layout, section, flat, truth.buffers, &Core::DrawBlockHost::Draw::BytesAt, &draw);
    Core::DrawBlockHost::NoteVerified(mismatches.size());
    if (mismatches.empty()) return;
    std::string text = "APS5_VERIFY_DRAW_BLOCK: a block section differs from the descriptor path:";
    for (std::size_t index = 0; index < mismatches.size() && index < 16; ++index) {
        const auto line = Core::Describe(mismatches[index]);
        std::fprintf(stderr, "[draw-block] verify: %s%c", line.c_str(), 10);
        text += " " + line + ";";
    }
    throw std::runtime_error(text);
}

}

void DrawBlockSections::Patch(std::array<std::byte, PipelinePushConstantBytes>& block) const {
    for (std::uint32_t index = 0; index < count; ++index) {
        Require(pushByte[index] <= block.size() - sizeof(std::uint64_t), "a block section address lies outside the push constant block");
        std::memcpy(block.data() + pushByte[index], &address[index], sizeof(std::uint64_t));
    }
}

bool UsesDrawBlocks(std::span<const CompiledShader> shaders) {
    return std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) { return shader.program != nullptr && shader.program->blockModule; });
}

void WriteDrawBlocks(const Context& context, Recorder* recorder, std::span<const CompiledShader> shaders, const ColorTarget* target, DrawBlockSections& sections, std::vector<std::shared_ptr<void>>* kept) {
    sections = {};
    if (!UsesDrawBlocks(shaders)) {
        if (DrawBlocks()) {
            Core::DrawBlockHost::NoteClassic();
            report();
        }
        return;
    }
    Require(context.bufferDeviceAddress, "a shader compiled for draw blocks needs the buffer device address feature");
    const bool verify = VerifyDrawBlocks();
    auto& host = Core::DrawBlockHost::For(context);
    Core::DrawBlockHost::Draw draw(host, recorder, kept, verify);
    std::size_t stages = 0;
    bool mixed = false;
    std::array<Core::BufferRequest, Abi::MaxBuffers> requests{};
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) continue;
        const auto& program = *shader.program;
        if (!program.blockModule) {
            mixed = mixed || classicBuffers(program);
            continue;
        }
        const auto& layout = program.blockLayout;
        const auto buffers = static_cast<std::uint32_t>(layout.buffers.size());
        Require(sections.count < sections.address.size() && layout.words <= Abi::MaxWords && buffers <= Abi::MaxBuffers, "a draw block exceeds its layout limits");
        Require(program.blockAddressDword != 0xffffffffu, "a block module without a pushed section address");
        const auto* flat = bindingOf(program, ShaderRecompiler::DescriptorRole::FlattenedSrt);
        const auto* guest = bindingOf(program, ShaderRecompiler::DescriptorRole::GuestBuffers);
        Require(layout.words == 0 || (flat != nullptr && flat->guestDescriptor.size() >= layout.words), "a block module holds fewer flattened words than its layout reads");
        const auto bytes = Abi::SectionBytes(layout.words, buffers);
        std::uint64_t address = 0;
        if (bytes != 0) {
            auto* section = draw.Section(bytes, address);
            if (layout.words != 0) std::memcpy(section, flat->guestDescriptor.data(), static_cast<std::size_t>(layout.words) * sizeof(std::uint32_t));
            for (std::uint32_t index = 0; index < buffers; ++index) {
                const auto resource = layout.buffers[index].resource;
                Require(guest != nullptr && guest->guestDescriptor.size() >= (static_cast<std::size_t>(resource) + 1u) * 4u, "a block module lists a buffer its bindings do not describe");
                const auto* words = guest->guestDescriptor.data() + static_cast<std::size_t>(resource) * 4u;
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
                auto& request = requests[index];
                request = {};
                request.buffer = index;
                request.address = descriptor.Base48();
                request.bytes = ShaderResources::BytesUpToTarget(request.address, descriptor.GetSize(), descriptor.Stride(), target);
                request.misalignment = static_cast<std::uint32_t>(request.address & Abi::ServiceMisalignmentMask);
                request.present = request.address != 0 && request.bytes != 0 && request.bytes <= 0xfffffff0ull;
                Core::BufferAnswer answer;
                static_cast<void>(Core::DrawBlockHost::Draw::Resolve(&draw, request, answer));
                if (answer.deviceAddress == 0) request.present = false;
                writeEntry(section + Abi::BufferEntryWord(layout.words, index), answer, request.misalignment);
            }
            const auto used = Abi::SectionWords(layout.words, buffers);
            std::fill(section + used, section + bytes / sizeof(std::uint32_t), 0u);
            if (verify) verifyStage(program, std::span<const std::uint32_t>(section, used), flat != nullptr ? std::span<const std::uint32_t>(flat->guestDescriptor) : std::span<const std::uint32_t>(), std::span<const Core::BufferRequest>(requests.data(), buffers), draw);
        }
        sections.address[sections.count] = address;
        sections.pushByte[sections.count] = shader.pushConstantOffset + program.blockAddressDword * static_cast<std::uint32_t>(sizeof(std::uint32_t));
        ++sections.count;
        ++stages;
    }
    draw.Served(stages, mixed);
    report();
}

}
