#include "BdaAbi.hpp"
#include "Recompiler.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <span>
#include <spirv/unified1/spirv.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace ShaderRecompiler;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("compute registration form: " + message);
}

SpirvTarget Target() {
    static constexpr std::array<std::uint32_t, 5> capabilities{
        spv::CapabilityInt64, spv::CapabilityPhysicalStorageBufferAddresses, spv::CapabilityStorageBuffer8BitAccess, spv::CapabilityClipDistance, spv::CapabilityInt64Atomics,
    };
    static constexpr std::array<std::string_view, 4> extensions{
        "SPV_EXT_descriptor_indexing", "SPV_KHR_8bit_storage", "SPV_KHR_float_controls", "SPV_KHR_physical_storage_buffer",
    };
    SpirvTarget target{0x00401000u, 0x00010300u, 32, BdaAbi::Version, capabilities, extensions, false,
            {1024, 1024, 64}, 1024, 32768, {}, {}};
    target.registrationModules = true;
    return target;
}

alignas(256) std::array<std::uint32_t, 64> Input{};
alignas(256) std::array<std::uint32_t, 64> Output{};
alignas(256) constexpr std::array<std::uint32_t, 7> BufferOnly{
    0x34020082, 0xe0301000, 0x80000201, 0xbf8c3f70, 0xe0701000, 0x80010201, 0xbf810000,
};
alignas(256) constexpr std::array<std::uint32_t, 7> FormattedBuffer{
    0x34020082, 0xe0001000, 0x80000201, 0xbf8c3f70, 0xe0701000, 0x80010201, 0xbf810000,
};
alignas(256) constexpr std::array<std::uint32_t, 7> WithImage{
    0x34020082, 0xf0002f28, 0x0002141e, 0xbf8c3f70, 0xe0701000, 0x80011401, 0xbf810000,
};

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
}

struct Fixture {
    std::array<std::uint32_t, 16> userData{};
    std::array<MemoryRegion, 1> memory{};
    RecompileRequest request{};
    explicit Fixture(std::span<const std::uint32_t> code) {
        const auto input = BufferDescriptor(Input.data(), sizeof(Input));
        const auto output = BufferDescriptor(Output.data(), sizeof(Output));
        std::copy(input.begin(), input.end(), userData.begin());
        std::copy(output.begin(), output.end(), userData.begin() + 4);
        memory = {{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
        request.shader = {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}};
        request.target = Target();
        request.layout = {0, 0, 0, 128};
        request.context.waveSize = 32u;
        request.context.userDataBaseRegister = 0;
        request.context.userData = userData;
        request.context.compute = ShaderComputeStageInfo{{32u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
        request.context.memory = memory;
    }
};

}

int main(int argc, char** argv) {
    try {
        const bool belowThreshold = argc > 1 && std::string_view(argv[1]) == "below";
        Fixture buffers(BufferOnly);
        const auto handle = PrepareShader(buffers.request);
        const auto registered = RegistrationResult(buffers.request, *handle);
        SrtRuntime runtime{};
        runtime.userData = buffers.userData;
        const auto capture = CaptureResources(buffers.request, runtime, *handle);
        const auto dispatched = MaterializeShader(buffers.request, *capture, *handle);
        if (belowThreshold) {
            Check(registered == nullptr, "a buffer-only program below the word threshold got a registration module");
            Check(!dispatched->registrationModule, "a buffer-only program below the word threshold was materialized in registration form");
        } else {
            Check(registered != nullptr && registered->registrationModule, "a buffer-only program has no registration module");
            Check(dispatched->registrationModule, "a buffer-only program was not materialized in registration form");
            Check(!registered->Identity().Empty(), "the registration module has no identity");
            Check(registered->spirv == dispatched->spirv, "the registration and dispatch modules differ");
            Check(registered->Identity() == dispatched->Identity(), "the registration and dispatch identities differ");
            const auto words = static_cast<std::uint32_t>(registered->spirv.size());
            Check(ComputeRegistrationPrebuildable(*registered), "a registration module at the word threshold is not prebuilt");
            Check(ComputeRegistrationPrebuildable(*registered, words), "a registration module of exactly the word threshold is not prebuilt");
            Check(!ComputeRegistrationPrebuildable(*registered, words + 1u), "a registration module under the word threshold is prebuilt");
            Check(!ComputeRegistrationPrebuildable(*dispatched, words + 1u), "a dispatch module under the word threshold is prebuilt");
        }
        if (!belowThreshold) {
            Fixture formatted(FormattedBuffer);
            const auto formattedHandle = PrepareShader(formatted.request);
            Check(GetPreparedArtifact(*formattedHandle).registrationModule, "a program with a formatted buffer load was not emitted in registration form");
            const auto formattedRegistered = RegistrationResult(formatted.request, *formattedHandle);
            Check(formattedRegistered != nullptr && formattedRegistered->registrationModule, "a program with a formatted buffer load has no registration module");
            SrtRuntime formattedRuntime{};
            formattedRuntime.userData = formatted.userData;
            const auto formattedCapture = CaptureResources(formatted.request, formattedRuntime, *formattedHandle);
            const auto formattedDispatched = MaterializeShader(formatted.request, *formattedCapture, *formattedHandle);
            Check(formattedDispatched->registrationModule, "a program with a formatted buffer load was not materialized in registration form");
            Check(formattedRegistered->spirv == formattedDispatched->spirv, "the registration and dispatch modules of a formatted buffer load differ");
            Check(formattedRegistered->Identity() == formattedDispatched->Identity(), "the registration and dispatch identities of a formatted buffer load differ");
        }
        Fixture images(WithImage);
        const auto imageHandle = PrepareShader(images.request);
        Check(!GetPreparedArtifact(*imageHandle).registrationModule, "a program with images was emitted in registration form");
        Check(RegistrationResult(images.request, *imageHandle) == nullptr, "a program with images got a registration module");
        std::puts("compute registration form tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
