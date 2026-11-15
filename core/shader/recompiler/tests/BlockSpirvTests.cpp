#include "SpirvBackend/SpirvBlockAccess.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "BlockAbi.hpp"
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace ShaderRecompiler;

constexpr std::uint32_t BlockWords = 3u;
constexpr std::uint32_t MainBuffer = 1u;
constexpr std::uint32_t HelperBuffer = 0u;
constexpr std::uint32_t PushDwords = 32u;

bool failed = false;

void check(bool condition, const char* text) {
    if (condition) return;
    std::fprintf(stderr, "block spirv: %s\n", text);
    failed = true;
}

template<typename TFunction>
bool refuses(TFunction&& function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

struct Instruction {
    std::uint32_t opcode = 0;
    std::span<const std::uint32_t> words;
};

std::vector<Instruction> decode(const std::vector<std::uint32_t>& module) {
    std::vector<Instruction> instructions;
    for (std::size_t offset = 5u; offset < module.size();) {
        const std::size_t count = module[offset] >> spv::WordCountShift;
        if (count == 0u || offset + count > module.size()) break;
        instructions.push_back({module[offset] & spv::OpCodeMask, std::span<const std::uint32_t>(module).subspan(offset, count)});
        offset += count;
    }
    return instructions;
}

bool typedResult(std::uint32_t opcode) {
    switch (opcode) {
    case spv::OpLoad:
    case spv::OpAccessChain:
    case spv::OpConvertUToPtr:
    case spv::OpUConvert:
    case spv::OpBitwiseOr:
    case spv::OpBitwiseAnd:
    case spv::OpShiftLeftLogical:
    case spv::OpIEqual:
    case spv::OpINotEqual:
    case spv::OpLogicalAnd:
    case spv::OpULessThan:
    case spv::OpULessThanEqual:
    case spv::OpUGreaterThanEqual:
    case spv::OpISub:
    case spv::OpIAdd:
    case spv::OpPhi:
        return true;
    default:
        return false;
    }
}

struct Decoded {
    std::vector<Instruction> instructions;
    std::map<std::uint32_t, std::size_t> definitions;
    std::map<std::uint32_t, std::uint32_t> pointerStorage;

    explicit Decoded(const std::vector<std::uint32_t>& module) : instructions(decode(module)) {
        for (std::size_t index = 0; index < instructions.size(); ++index) {
            const auto& instruction = instructions[index];
            if (instruction.opcode == spv::OpTypePointer) pointerStorage[instruction.words[1]] = instruction.words[2];
            if (typedResult(instruction.opcode)) definitions[instruction.words[2]] = index;
        }
    }

    [[nodiscard]] const Instruction* Definition(std::uint32_t id) const {
        const auto found = definitions.find(id);
        return found == definitions.end() ? nullptr : &instructions[found->second];
    }

    [[nodiscard]] bool Physical(std::uint32_t pointer) const {
        const auto* definition = Definition(pointer);
        if (definition == nullptr) return false;
        const auto found = pointerStorage.find(definition->words[1]);
        return found != pointerStorage.end() && found->second == spv::StorageClassPhysicalStorageBuffer;
    }

    void Conjuncts(std::uint32_t condition, std::vector<std::uint32_t>& out) const {
        out.push_back(condition);
        const auto* definition = Definition(condition);
        if (definition == nullptr || definition->opcode != spv::OpLogicalAnd) return;
        Conjuncts(definition->words[3], out);
        Conjuncts(definition->words[4], out);
    }

    [[nodiscard]] std::vector<std::uint32_t> Guard(std::size_t load) const {
        std::size_t label = load;
        while (label > 0u && instructions[label].opcode != spv::OpLabel) --label;
        if (instructions[label].opcode != spv::OpLabel) return {};
        const auto block = instructions[label].words[1];
        std::vector<std::uint32_t> conjuncts;
        for (std::size_t index = 0; index < label; ++index) {
            const auto& instruction = instructions[index];
            if (instruction.opcode != spv::OpBranchConditional || instruction.words[2] != block || instruction.words[3] == block) continue;
            Conjuncts(instruction.words[1], conjuncts);
        }
        return conjuncts;
    }
};

struct GuardedRead {
    BlockBufferOperands buffer;
    std::uint32_t byteOffset = 0;
    std::uint32_t inBounds = 0;
    std::uint32_t value = 0;
};

bool contains(const std::vector<std::uint32_t>& ids, std::uint32_t id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

void checkGuard(const Decoded& decoded, std::size_t load, const GuardedRead& read, std::uint32_t four, std::uint32_t zero) {
    const auto guard = decoded.Guard(load);
    check(!guard.empty(), "a buffer load is not under a conditional branch");
    check(contains(guard, read.inBounds), "a buffer load is not guarded by the descriptor bound");
    check(contains(guard, read.buffer.present), "a buffer load is not guarded by the presence of its entry");
    bool holdsWord = false;
    bool withinBytes = false;
    for (const auto id : guard) {
        const auto* definition = decoded.Definition(id);
        if (definition == nullptr) continue;
        if (definition->opcode == spv::OpUGreaterThanEqual && definition->words[3] == read.buffer.bytes && definition->words[4] == four) holdsWord = true;
        if (definition->opcode != spv::OpULessThanEqual || definition->words[3] != read.byteOffset) continue;
        const auto* last = decoded.Definition(definition->words[4]);
        if (last != nullptr && last->opcode == spv::OpISub && last->words[3] == read.buffer.bytes && last->words[4] == four) withinBytes = true;
    }
    check(holdsWord, "a buffer load is not guarded by the size of its entry");
    check(withinBytes, "a buffer load is not guarded by the readable bytes of its entry");
    const auto* pointer = decoded.Definition(decoded.instructions[load].words[3]);
    const auto* aligned = pointer != nullptr ? decoded.Definition(pointer->words[3]) : nullptr;
    const auto* target = aligned != nullptr && aligned->opcode == spv::OpBitwiseAnd ? decoded.Definition(aligned->words[3]) : nullptr;
    check(target != nullptr && target->opcode == spv::OpIAdd && target->words[3] == read.buffer.address, "a buffer load does not start at the address of its entry");
    const auto* merged = decoded.Definition(read.value);
    check(merged != nullptr && merged->opcode == spv::OpPhi && merged->words.size() == 7u && merged->words[5] == zero, "a refused buffer load does not yield zero");
}

void definePushConstants(SpirvEmitterState& state) {
    const auto array = state.module.DecoratedType(spv::OpTypeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, TypeU32(state), ConstantU32(state, PushDwords));
    const auto block = state.module.DecoratedType(spv::OpTypeStruct,
        {{spv::OpMemberDecorate, {0u, spv::DecorationOffset, 0u}},
         {spv::OpDecorate, {spv::DecorationBlock}}},
        array);
    state.pushConstantVariable = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPushConstant, block), spv::StorageClassPushConstant);
}

std::uint32_t beginFunction(SpirvEmitterState& state) {
    const auto voidType = TypeVoid(state);
    const auto functionType = TypeFunction(state);
    const auto function = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, functionType);
    EmitLabel(state, state.module.AllocateId());
    return function;
}

void endFunction(SpirvEmitterState& state) {
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
}

void checkRefusals() {
    IrProgram program;
    const ShaderStageInputInfo inputInfo;
    SpirvEmitterState state(program, inputInfo);
    state.blockWords = BlockWords;
    state.module.EmitCapability(spv::CapabilityShader);
    check(refuses([&] { EmitBlockBase(state); }), "a block base without push constants is accepted");
    definePushConstants(state);
    static_cast<void>(beginFunction(state));
    check(refuses([&] { static_cast<void>(EmitBlockWord(state, 0u)); }), "a block word before the block base is accepted");
    check(refuses([&] { static_cast<void>(EmitBlockBuffer(state, 0u)); }), "a block buffer before the block base is accepted");
    EmitBlockBase(state);
    check(refuses([&] { static_cast<void>(EmitBlockWord(state, BlockWords)); }), "a block word past the layout is accepted");
    check(refuses([&] { static_cast<void>(EmitBlockBuffer(state, BlockAbi::MaxBuffers)); }), "a block buffer past the layout is accepted");
    state.module.BeginHelperFunction();
    check(refuses([&] { EmitBlockBase(state); }), "a block base inside a helper function is accepted");
    state.module.EndHelperFunction();
}

}

int main() {
    checkRefusals();

    IrProgram program;
    const ShaderStageInputInfo inputInfo;
    SpirvEmitterState state(program, inputInfo);
    state.blockWords = BlockWords;
    state.module.EmitCapability(spv::CapabilityShader);
    state.module.AddMemoryModel(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelGLSL450);
    definePushConstants(state);

    const auto entry = beginFunction(state);
    EmitBlockBase(state);
    const auto base = state.blockBase;
    std::array<std::uint32_t, BlockWords> words {};
    for (std::uint32_t word = 0; word < BlockWords; ++word) words[word] = EmitBlockWord(state, word);
    const auto buffer = EmitBlockBuffer(state, MainBuffer);
    std::array<GuardedRead, 3> reads {};
    reads[0].buffer = buffer;
    reads[0].byteOffset = words[2];
    reads[0].inBounds = Binary(state, spv::OpULessThan, TypeBool(state), words[0], words[1]);
    reads[0].value = EmitBlockLoad32(state, reads[0].buffer, reads[0].byteOffset, reads[0].inBounds);
    reads[1].buffer = buffer;
    reads[1].byteOffset = Binary(state, spv::OpIAdd, TypeU32(state), words[2], ConstantU32(state, 4u));
    reads[1].inBounds = Binary(state, spv::OpULessThan, TypeBool(state), reads[1].byteOffset, words[1]);
    reads[1].value = EmitBlockLoad32(state, reads[1].buffer, reads[1].byteOffset, reads[1].inBounds);
    endFunction(state);
    const auto entryLabel = state.currentLabel;

    state.module.BeginHelperFunction();
    static_cast<void>(beginFunction(state));
    const auto helperWord = EmitBlockWord(state, 0u);
    reads[2].buffer = EmitBlockBuffer(state, HelperBuffer);
    reads[2].byteOffset = helperWord;
    reads[2].inBounds = Binary(state, spv::OpULessThan, TypeBool(state), helperWord, reads[2].buffer.bytes);
    reads[2].value = EmitBlockLoad32(state, reads[2].buffer, reads[2].byteOffset, reads[2].inBounds);
    endFunction(state);
    state.module.EndHelperFunction();
    state.currentLabel = entryLabel;

    state.module.EmitEntryPoint(spv::ExecutionModelGLCompute, entry, "main", {});
    state.module.AddExecutionMode(entry, spv::ExecutionModeLocalSize, 1u, 1u, 1u);

    const auto four = ConstantU32(state, 4u);
    const auto zero = ConstantU32(state, 0u);
    std::vector<std::uint32_t> entryWordIndices;
    for (std::uint32_t word = 0; word < BlockWords; ++word) entryWordIndices.push_back(ConstantU32(state, word));
    for (std::uint32_t word = 0; word < BlockAbi::BufferEntryWords; ++word) entryWordIndices.push_back(ConstantU32(state, BlockAbi::BufferEntryWord(BlockWords, MainBuffer) + word));
    const auto module = state.module.Finalize();

    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        std::fprintf(stderr, "block spirv: validator at word %lu: %s\n", static_cast<unsigned long>(position.index), message);
    });
    check(tools.Validate(module), "the block access module is rejected by the validator");

    const Decoded decoded(module);
    bool capability = false;
    bool extension = false;
    for (const auto& instruction : decoded.instructions) {
        if (instruction.opcode == spv::OpCapability && instruction.words[1] == spv::CapabilityPhysicalStorageBufferAddresses) capability = true;
        if (instruction.opcode == spv::OpExtension) extension = true;
    }
    check(capability, "the physical storage buffer capability is missing");
    check(extension, "the physical storage buffer extension is missing");

    const auto* baseDefinition = decoded.Definition(base);
    check(base != 0u && baseDefinition != nullptr && baseDefinition->opcode == spv::OpConvertUToPtr && decoded.Physical(base), "the block base is not a physical storage buffer pointer");

    std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> wordLoads;
    std::map<std::uint32_t, std::uint32_t> loadedValues;
    std::vector<std::size_t> bufferLoads;
    std::uint32_t physicalLoads = 0;
    std::uint32_t splitLoads = 0;
    for (std::size_t index = 0; index < decoded.instructions.size(); ++index) {
        const auto& instruction = decoded.instructions[index];
        if (instruction.opcode != spv::OpLoad || !decoded.Physical(instruction.words[3])) continue;
        ++physicalLoads;
        check(instruction.words.size() == 6u && instruction.words[4] == spv::MemoryAccessAlignedMask, "a physical load does not state its alignment");
        const auto* pointer = decoded.Definition(instruction.words[3]);
        if (pointer->opcode == spv::OpAccessChain && pointer->words.size() == 6u) {
            ++wordLoads[{pointer->words[3], pointer->words[5]}];
            loadedValues[instruction.words[2]] = pointer->words[5];
            check(decoded.Physical(pointer->words[3]), "a block word is not read through the block pointer");
        } else if (pointer->opcode == spv::OpConvertUToPtr) {
            check(instruction.words[5] == 4u, "a buffer load is not aligned to its word");
            const auto* source = decoded.Definition(pointer->words[3]);
            if (source != nullptr && source->opcode == spv::OpBitwiseAnd) bufferLoads.push_back(index);
            else ++splitLoads;
        } else {
            check(false, "a physical load uses an unexpected pointer");
        }
    }

    const std::uint32_t helperWords = 1u + BlockAbi::BufferEntryWords;
    check(wordLoads.size() == entryWordIndices.size() + helperWords, "the number of distinct block words read is wrong");
    for (const auto& [key, count] : wordLoads) check(count == 1u, "a block word is loaded more than once");
    for (const auto index : entryWordIndices) check(wordLoads.count({base, index}) == 1u, "an entry function block word is not read through the stored base");
    for (std::uint32_t word = 0; word < BlockWords; ++word) {
        const auto found = loadedValues.find(words[word]);
        check(found != loadedValues.end() && found->second == entryWordIndices[word], "a block word value is not the load of its own index");
    }
    for (const auto& [key, count] : wordLoads) {
        const auto* index = decoded.Definition(key.second);
        check(index == nullptr, "a block word index is not a constant");
    }
    std::uint32_t helperLoads = 0;
    for (const auto& [key, count] : wordLoads) {
        if (key.first != base) helperLoads += count;
    }
    check(helperLoads == helperWords, "a helper function reads block words through the entry function base");

    check(bufferLoads.size() == reads.size(), "the number of buffer loads is wrong");
    check(splitLoads == reads.size(), "a buffer load lacks the second word of an unaligned read");
    check(physicalLoads == wordLoads.size() + 2u * reads.size(), "the number of physical loads is wrong");
    for (std::size_t read = 0; read < reads.size() && read < bufferLoads.size(); ++read) checkGuard(decoded, bufferLoads[read], reads[read], four, zero);

    const auto* misalignment = decoded.Definition(buffer.misalignment);
    check(misalignment != nullptr && misalignment->opcode == spv::OpBitwiseAnd, "the buffer misalignment is not taken from the service word");
    const auto* address = decoded.Definition(buffer.address);
    check(address != nullptr && address->opcode == spv::OpBitwiseOr, "the buffer address is not joined from its two words");
    const auto bytesWord = loadedValues.find(buffer.bytes);
    check(bytesWord != loadedValues.end() && bytesWord->second == entryWordIndices[BlockWords + BlockAbi::EntryBytes], "the buffer bytes are not the load of the entry bytes word");

    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
