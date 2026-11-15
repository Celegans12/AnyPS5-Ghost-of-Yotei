#include "SpirvBackend/SpirvBlockAccess.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include "BlockAbi.hpp"
#include "PipelineSpecialization.hpp"
#include <spirv/unified1/spirv.hpp>
#include <cstdint>

namespace ShaderRecompiler {
namespace {

std::uint32_t widen(SpirvEmitterState& state, std::uint32_t value) {
    return Unary(state, spv::OpUConvert, TypeScalarU64(state), value);
}

std::uint32_t joinAddress(SpirvEmitterState& state, std::uint32_t low, std::uint32_t high) {
    const auto u64 = TypeScalarU64(state);
    const auto wideLow = widen(state, low);
    const auto wideHigh = widen(state, high);
    return Binary(state, spv::OpBitwiseOr, u64, wideLow, Binary(state, spv::OpShiftLeftLogical, u64, wideHigh, ConstantU32(state, 32u)));
}

std::uint32_t blockWordsType(SpirvEmitterState& state) {
    const auto array = state.module.DecoratedType(spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, TypeU32(state));
    return state.module.DecoratedType(spv::OpTypeStruct,
        {{spv::OpDecorate, {spv::DecorationBlock}},
         {spv::OpMemberDecorate, {0u, spv::DecorationOffset, 0u}}},
        array);
}

std::uint32_t pushedWord(SpirvEmitterState& state, std::uint32_t index) {
    const auto elementType = TypePushConstantElementPointer(state);
    const auto member = ConstantU32(state, 0u);
    const auto& layout = state.program.Metadata().bindings;
    const auto first = state.module.SpecializationConstant(TypeU32(state), PipelineSpecialization::BlockAddressOffset, layout.UsesBlock() ? layout.blockAddressDword : 0u);
    const auto element = Binary(state, spv::OpIAdd, TypeU32(state), Binary(state, spv::OpISub, TypeU32(state), first, ConstantU32(state, layout.PushSlotDword())), ConstantU32(state, index));
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, elementType, pointer, state.pushConstantVariable, member, element);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
    return value;
}

std::uint32_t pushedBase(SpirvEmitterState& state) {
    if (state.pushConstantVariable == 0u) FailEmit("block access requires the stage push constant slice");
    const auto low = pushedWord(state, 0u);
    const auto high = pushedWord(state, 1u);
    const auto address = joinAddress(state, low, high);
    const auto pointerType = TypePointer(state, spv::StorageClassPhysicalStorageBuffer, blockWordsType(state));
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpConvertUToPtr, pointerType, pointer, address);
    return pointer;
}

std::uint32_t blockBase(SpirvEmitterState& state) {
    if (state.module.InHelperFunction()) return pushedBase(state);
    if (state.blockBase == 0u) FailEmit("block base was not emitted before a block read");
    return state.blockBase;
}

std::uint32_t loadWord(SpirvEmitterState& state, std::uint32_t base, std::uint32_t word) {
    const auto elementType = TypePhysicalU32Pointer(state);
    const auto member = ConstantU32(state, 0u);
    const auto element = ConstantU32(state, word);
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, elementType, pointer, base, member, element);
    const auto value = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, spv::MemoryAccessAlignedMask, 4u);
    return value;
}

}

void EmitBlockBase(SpirvEmitterState& state) {
    if (state.module.InHelperFunction()) FailEmit("block base must be emitted in the entry function");
    state.module.EmitCapability(spv::CapabilityInt64);
    state.module.EmitCapability(spv::CapabilityPhysicalStorageBufferAddresses);
    state.module.EmitExtension("SPV_KHR_physical_storage_buffer");
    state.blockBase = pushedBase(state);
}

std::uint32_t EmitBlockWord(SpirvEmitterState& state, std::uint32_t word) {
    if (word >= state.blockWords) FailEmit("block word is outside the block layout");
    const auto base = blockBase(state);
    return loadWord(state, base, word);
}

BlockBufferOperands EmitBlockBuffer(SpirvEmitterState& state, std::uint32_t buffer) {
    if (buffer >= BlockAbi::MaxBuffers || (state.blockAccess && buffer >= state.blockBuffers)) FailEmit("block buffer is outside the block layout");
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto first = BlockAbi::BufferEntryWord(state.blockWords, buffer);
    const auto base = blockBase(state);
    const auto low = loadWord(state, base, first + BlockAbi::EntryAddressLow);
    const auto high = loadWord(state, base, first + BlockAbi::EntryAddressHigh);
    const auto bytes = loadWord(state, base, first + BlockAbi::EntryBytes);
    const auto service = loadWord(state, base, first + BlockAbi::EntryService);
    BlockBufferOperands operands;
    operands.address = joinAddress(state, low, high);
    operands.bytes = bytes;
    operands.misalignment = Binary(state, spv::OpBitwiseAnd, u32, service, ConstantU32(state, BlockAbi::ServiceMisalignmentMask));
    const auto absent = Binary(state, spv::OpBitwiseAnd, u32, service, ConstantU32(state, BlockAbi::ServiceAbsent));
    const auto listed = Binary(state, spv::OpIEqual, boolean, absent, ConstantU32(state, 0u));
    const auto mapped = Binary(state, spv::OpINotEqual, boolean, operands.address, state.module.Constant(spv::OpConstant, TypeScalarU64(state), 0u, 0u));
    operands.present = Binary(state, spv::OpLogicalAnd, boolean, listed, mapped);
    return operands;
}

std::uint32_t EmitBlockLoad32(SpirvEmitterState& state, const BlockBufferOperands& buffer, std::uint32_t byteOffset, std::uint32_t inBounds) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto four = ConstantU32(state, 4u);
    const auto holdsWord = Binary(state, spv::OpUGreaterThanEqual, boolean, buffer.bytes, four);
    const auto lastOffset = Binary(state, spv::OpISub, u32, buffer.bytes, four);
    const auto withinBytes = Binary(state, spv::OpULessThanEqual, boolean, byteOffset, lastOffset);
    const auto readable = Binary(state, spv::OpLogicalAnd, boolean, holdsWord, withinBytes);
    const auto described = Binary(state, spv::OpLogicalAnd, boolean, inBounds, buffer.present);
    const auto allowed = Binary(state, spv::OpLogicalAnd, boolean, described, readable);
    return EmitValueOrZeroIfCondition(state, allowed, [&] {
        const auto u64 = TypeScalarU64(state);
        const auto target = Binary(state, spv::OpIAdd, u64, buffer.address, widen(state, byteOffset));
        const auto first = Binary(state, spv::OpBitwiseAnd, u64, target, state.module.Constant(spv::OpConstant, u64, 0xfffffffcu, 0xffffffffu));
        const auto load = [&](std::uint32_t address) {
            const auto pointer = state.module.AllocateId();
            state.module.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, address);
            const auto value = state.module.AllocateId();
            state.module.AddFunction(spv::OpLoad, u32, value, pointer, spv::MemoryAccessAlignedMask, 4u);
            return value;
        };
        const auto bytes = Binary(state, spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, target), ConstantU32(state, 3u));
        const auto shift = Binary(state, spv::OpShiftLeftLogical, u32, bytes, ConstantU32(state, 3u));
        const auto low = Binary(state, spv::OpShiftRightLogical, u32, load(first), shift);
        const auto split = Binary(state, spv::OpINotEqual, boolean, bytes, ConstantU32(state, 0u));
        const auto high = EmitValueOrZeroIfCondition(state, split, [&] {
            const auto second = Binary(state, spv::OpIAdd, u64, first, state.module.Constant(spv::OpConstant, u64, 4u, 0u));
            return Binary(state, spv::OpShiftLeftLogical, u32, load(second), Binary(state, spv::OpISub, u32, ConstantU32(state, 32u), shift));
        });
        return Binary(state, spv::OpBitwiseOr, u32, low, high);
    });
}

}
