#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"
#include "graphics/shader/recompiler/ir/Serialize.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::ShaderRecompiler::IR;

constexpr uint32_t GarbageType = 0x9008a150u;

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_shader_store_encode: %s\n", what);
    std::abort();
}

Value Corrupt(uint32_t type) {
    Value value;
    std::memcpy(&value, &type, sizeof(type));
    return value;
}

void Fill(ResourcePlan& plan, Value second) {
    auto& add = plan.value_storage.emplace_back(ValueOpcode::IAdd32);
    add.SetArg(0, Value(1u));
    add.SetArg(1, second);
    auto& multiply = plan.value_storage.emplace_back(ValueOpcode::IMul32);
    multiply.SetArg(0, Value(&add));
    multiply.SetArg(1, Value(7u));
}

void HealthyPlanRoundTrips() {
    ResourcePlan plan;
    Fill(plan, Value(2u));
    std::vector<uint8_t> bytes;
    EncodeFailure        failure;
    Require(SaveResourcePlan(plan, bytes, failure), "a healthy plan was refused");
    Require(!failure.Failed() && !bytes.empty(), "a healthy plan reported a failure or no bytes");
    ResourcePlan loaded;
    Require(DecodeResourcePlan(bytes, loaded), "a saved plan did not load back");
    Require(loaded.value_storage.size() == 2u, "the loaded plan lost instructions");
    const auto& add = loaded.value_storage.front();
    Require(add.GetOpcode() == ValueOpcode::IAdd32 && add.Arg(0) == Value(1u) && add.Arg(1) == Value(2u),
            "the loaded plan changed an operand");
    const auto& multiply = loaded.value_storage.back();
    Require(multiply.GetOpcode() == ValueOpcode::IMul32 && multiply.Arg(0) == Value(&loaded.value_storage.front()) &&
                multiply.Arg(1) == Value(7u),
            "the loaded plan changed an instruction reference");
    std::vector<uint8_t> roundTrip;
    Require(SaveResourcePlan(loaded, roundTrip, failure) && roundTrip == bytes, "the loaded plan encoded to different bytes");
}

void ImpossibleOperandRefusesThePlan() {
    ResourcePlan plan;
    Fill(plan, Corrupt(GarbageType));
    std::vector<uint8_t> bytes {1, 2, 3};
    EncodeFailure        failure;
    Require(!SaveResourcePlan(plan, bytes, failure), "a plan with an impossible operand was saved");
    Require(failure.reason == EncodeFailure::Reason::ValueType && failure.type == GarbageType,
            "the failure does not name the impossible value type");
    Require(failure.in_instruction && failure.opcode == ValueOpcode::IAdd32,
            "the failure does not name the opcode of the instruction holding the value");
    Require(bytes.empty(), "a refused plan left bytes behind");
}

void ImpossibleValueOutsideInstructionsRefusesThePlan() {
    ResourcePlan plan;
    Fill(plan, Value(2u));
    auto& source       = plan.descriptor_sources.emplace_back();
    source.dword_count = 1;
    source.dwords[0]   = Corrupt(GarbageType);
    std::vector<uint8_t> bytes;
    EncodeFailure        failure;
    Require(!SaveResourcePlan(plan, bytes, failure), "a plan with an impossible descriptor value was saved");
    Require(failure.reason == EncodeFailure::Reason::ValueType && failure.type == GarbageType &&
                !failure.in_instruction,
            "the failure does not say the value sits outside the instructions");
}

void InstructionOutsideThePlanRefusesThePlan() {
    Inst         outside(ValueOpcode::IAdd32);
    ResourcePlan plan;
    Fill(plan, Value(&outside));
    std::vector<uint8_t> bytes;
    EncodeFailure        failure;
    Require(!SaveResourcePlan(plan, bytes, failure), "a plan referring to a foreign instruction was saved");
    Require(failure.reason == EncodeFailure::Reason::UnknownInstruction && failure.in_instruction &&
                failure.opcode == ValueOpcode::IAdd32,
            "the failure does not describe the foreign instruction");
}

void RefusedPlanAbandonsTheStoreEntry() {
    const ShaderModule         module;
    const std::vector<uint32_t> state {1, 2, 3};
    const std::vector<uint32_t> spirv {0x07230203u, 0u};
    const std::vector<uint32_t> plain;

    ResourcePlan healthy;
    Fill(healthy, Value(2u));
    std::vector<uint8_t> record;
    Require(EncodeModuleRecord(ShaderType::Compute, 0x1111u, 4u, state, healthy, module, spirv, plain, record),
            "a healthy program was not encoded");
    Require(!record.empty(), "a healthy program produced an empty record");

    ResourcePlan broken;
    Fill(broken, Corrupt(GarbageType));
    std::vector<uint8_t> refused {9};
    Require(!EncodeModuleRecord(ShaderType::Compute, 0x2222u, 4u, state, broken, module, spirv, plain, refused),
            "a program with an impossible value was encoded");
    Require(refused.size() == 1u && refused.front() == 9u, "a refused program touched the record");
    Require(!EncodeModuleRecord(ShaderType::Compute, 0x2222u, 4u, state, broken, module, spirv, plain, refused),
            "a second encode of the same refused program was encoded");
    Require(!EncodeModuleRecord(ShaderType::Pixel, 0x3333u, 4u, state, broken, module, spirv, plain, refused),
            "a refused program of another hash was encoded");

    std::vector<uint8_t> after;
    Require(EncodeModuleRecord(ShaderType::Compute, 0x1111u, 4u, state, healthy, module, spirv, plain, after),
            "a healthy program was not encoded after a refusal");
    Require(after == record, "a refusal changed the encoding of the next program");
}

void StoredComputeModuleSelection() {
    using namespace ShaderStoreFormat;

    const auto frame = [](ShaderType stage, uint64_t id) {
        ShaderStoreFormat::Key key {.stage = stage, .hash = id};
        ShaderModule module;
        module.id = id;
        module.info.stage = stage;
        const std::vector<uint8_t> plan;
        const std::vector<uint32_t> spirv {0x07230203u, 0x00010300u, 0u, 1u, 0u};
        const std::vector<uint32_t> plain;
        Writer record;
        IoAll(record, Record::Module);
        IoStoreModule(record, key, module, plan, spirv, plain);
        std::string compressed(ZSTD_compressBound(record.bytes.size()), '\0');
        const auto size = ZSTD_compress(compressed.data(), compressed.size(), record.bytes.data(), record.bytes.size(), 1);
        Require(!ZSTD_isError(size), "could not encode the synthetic store frame");
        compressed.resize(size);
        const Frame header {static_cast<uint32_t>(size), static_cast<uint32_t>(record.bytes.size()),
                            XXH3_64bits(compressed.data(), size)};
        return std::string(reinterpret_cast<const char*>(&header), sizeof(header)) + compressed;
    };
    const std::array<uint8_t, 3> expected {1, 2, 3};
    const std::string header(reinterpret_cast<const char*>(expected.data()), expected.size());
    const auto compute = frame(ShaderType::Compute, 42);
    const auto pixel = frame(ShaderType::Pixel, 17);
    ShaderModule module;
    std::vector<uint32_t> spirv;
    std::string error;
    const auto load = [&](std::string contents, uint64_t id) {
        std::istringstream stream(std::move(contents));
        return LoadStoredComputeModule(stream, expected, id, module, spirv, error);
    };
    Require(load(header + pixel + compute, 42) && module.id == 42 && module.spirv_words == spirv.size() &&
                spirv.size() == 5 && spirv.front() == 0x07230203u,
            "replay did not select the requested compute record");
    Require(!load(header + compute, 43) && error.find("not found") != std::string::npos,
            "replay accepted a missing ID");
    Require(!load(header + pixel, 17) && error.find("not a compute") != std::string::npos,
            "replay accepted a graphics module");
    Require(!load(header + compute + compute, 42) && error.find("duplicate") != std::string::npos,
            "replay accepted duplicate IDs");
    Require(!load("bad" + compute, 42) && error.find("incompatible") != std::string::npos,
            "replay accepted an incompatible header");
    Require(!load(header + compute + pixel.substr(0, pixel.size() - 1), 42),
            "replay accepted a truncated frame after the selected module");
}

}

int main() {
    HealthyPlanRoundTrips();
    ImpossibleOperandRefusesThePlan();
    ImpossibleValueOutsideInstructionsRefusesThePlan();
    InstructionOutsideThePlanRefusesThePlan();
    RefusedPlanAbandonsTheStoreEntry();
    StoredComputeModuleSelection();
    std::puts("kyty_shim_shader_store_encode: ok");
    return 0;
}
