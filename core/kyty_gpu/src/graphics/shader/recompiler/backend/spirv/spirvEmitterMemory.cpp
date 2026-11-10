#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "common/envRead.h"
#include "common/shaderEnv.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/indirectLoadFast.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <bit>
#include <map>
#include <string>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t AndCondition(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return Binary(state, spv::OpLogicalAnd, TypeBool(state), lhs, rhs);
}

uint32_t EmitDsMaskedLaneRead(EmitterState& state, uint32_t source, uint32_t target,
                              uint32_t exec) {
	if (state.lane_count == 2 || state.wave_pair) {
		target = Binary(state, spv::OpBitwiseAnd, TypeU32(state), target, ConstantU32(state, 31));
	}
	const auto shuffled = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), shuffled,
	                          ConstantU32(state, spv::ScopeSubgroup), source, target);
	const auto source_exec = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeBool(state), source_exec,
	                          ConstantU32(state, spv::ScopeSubgroup), exec, target);
	const auto source_active =
	    AndCondition(state, source_exec, EmitSubgroupLaneActiveBool(state, target));
	return Select(state, TypeU32(state), source_active, shuffled, ConstantU32(state, 0));
}

struct BufferAddress {
	uint32_t offset;
	uint32_t byte;
};

BufferAddress CalculateBufferAddress(EmitterState& state, uint32_t index, uint32_t offset,
                                     uint32_t soffset, uint32_t immediate, uint32_t stride,
                                     uint32_t swizzle, uint32_t index_stride) {
	const auto zero = ConstantU32(state, 0);
	const auto one  = ConstantU32(state, 1);
	const auto add = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero ? rhs : rhs == zero ? lhs
		                                      : Binary(state, spv::OpIAdd, TypeU32(state), lhs, rhs);
	};
	const auto mul = [&](uint32_t lhs, uint32_t rhs) {
		return lhs == zero || rhs == zero ? zero
		       : lhs == one              ? rhs
		       : rhs == one              ? lhs
		                                 : Binary(state, spv::OpIMul, TypeU32(state), lhs, rhs);
	};
	if (immediate != 0u) {
		offset = add(offset, ConstantU32(state, immediate));
	}
	auto address = add(mul(index, stride), offset);
	if (swizzle != 0u) {
		const auto index_shift =
		    Binary(state, spv::OpIAdd, TypeU32(state), index_stride, ConstantU32(state, 3));
		const auto indices = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		                            ConstantU32(state, 1), index_shift);
		const auto index_msb =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), index, index_shift);
		const auto index_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), index,
		           Binary(state, spv::OpISub, TypeU32(state), indices, ConstantU32(state, 1)));
		const auto offset_msb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, ~3u));
		const auto offset_lsb =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), offset, ConstantU32(state, 3u));
		const auto msb = mul(add(mul(index_msb, stride), offset_msb), indices);
		const auto lsb = add(Binary(state, spv::OpShiftLeftLogical, TypeU32(state), index_lsb,
		                            ConstantU32(state, 2u)), offset_lsb);
		address = Select(state, TypeU32(state), swizzle, add(msb, lsb), address);
	}
	return {offset, add(address, soffset)};
}

uint32_t BufferLane(EmitterState& state) {
	return Binary(state, spv::OpBitwiseAnd, TypeU32(state), EmitSubgroupLocalInvocationId(state),
	              ConstantU32(state, 63));
}

uint32_t BufferByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto&      state  = ctx.state;
	const auto packed = StorageBufferPackedStride(state, mem);
	const auto stride = packed & 0x3fffu;
	auto       index  = ctx.Arg(inst, 1);
	if ((packed & (1u << 20u)) != 0u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, BufferLane(state));
	}
	const bool swizzle = stride != 0u && (packed & (1u << 14u)) != 0u;
	uint32_t stride_id = ConstantU32(state, stride);
	if ((packed & (1u << 31u)) != 0u) {
		// The record stride is this binding's shader-data dword.
		const auto stride_dword = state.program.bindings.memory_stride_dword;
		if (stride_dword == IR::PushData::NoStart) EXIT("dynamic buffer stride has no shader-data slot\n");
		const auto binding = ResourceForDescriptor(state, IR::DescriptorBindingKind::Buffers, mem.resource);
		stride_id = EmitShaderDataDwordLoad(state, stride_dword + binding);
		const auto byte = CalculateBufferAddress(state, index, ctx.Arg(inst, 2), ctx.Arg(inst, 3),
		                                         mem.offset, stride_id,
		                                         swizzle ? ConstantBool(state, true) : 0u,
		                                         ConstantU32(state, (packed >> 16u) & 3u))
		                      .byte;
		// A zero run-time stride is a null buffer: every access is out of bounds (the caller maps
		// UINT32_MAX past its binding, where robust access reads zero and drops stores).
		return Select(state, TypeU32(state),
		              Binary(state, spv::OpIEqual, TypeBool(state), stride_id, ConstantU32(state, 0)),
		              ConstantU32(state, UINT32_MAX), byte);
	}
	return CalculateBufferAddress(state, index, ctx.Arg(inst, 2), ctx.Arg(inst, 3), mem.offset,
	                              stride_id,
	                              swizzle ? ConstantBool(state, true) : 0u,
	                              ConstantU32(state, (packed >> 16u) & 3u))
	    .byte;
}

uint32_t AddU64Low(EmitterState& state, uint32_t low, uint32_t high, uint32_t add_low,
                   uint32_t add_high, uint32_t& out_high) {
	const auto result = Binary(state, spv::OpIAdd, TypeU32(state), low, add_low);
	const auto carry  = Binary(state, spv::OpULessThan, TypeBool(state), result, low);
	out_high =
	    Binary(state, spv::OpIAdd, TypeU32(state),
	           Binary(state, spv::OpIAdd, TypeU32(state), high, add_high),
	           Select(state, TypeU32(state), carry, ConstantU32(state, 1), ConstantU32(state, 0)));
	return result;
}

uint32_t AddAddressOffset(EmitterState& state, const IR::MemoryInfo& mem, uint32_t low,
                          uint32_t& high) {
	const auto immediate = static_cast<int32_t>(mem.offset);
	if (immediate == 0) return low;
	const auto immediate_low  = ConstantU32(state, static_cast<uint32_t>(immediate));
	const auto immediate_high = ConstantU32(state, immediate < 0 ? UINT32_MAX : 0u);
	return AddU64Low(state, low, high, immediate_low, immediate_high, high);
}

uint32_t ScratchByteAddress(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t low,
                            uint32_t high) {
	auto& state = ctx.state;
	low = AddAddressOffset(state, mem, low, high);
	const auto valid = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, 0));
	return Select(state, TypeU32(state), valid, low, ConstantU32(state, UINT32_MAX));
}

uint32_t GuestAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	auto  low   = ctx.Arg(inst, 1);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, ~3u));
	}
	uint32_t address = 0;
	if (mem.address_is_full) {
		address = PackU64(state, low, ctx.Arg(inst, 2));
	} else {
		const auto* handle = inst.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) {
			ctx.Fail(inst, "has no address base pair");
			return ConstantU64(state, 0);
		}
		auto base_low = ctx.Arg(*handle, 0);
		if (mem.kind == IR::ResourceKind::ScalarAddress) {
			base_low = Binary(state, spv::OpBitwiseAnd, TypeU32(state), base_low,
			                  ConstantU32(state, ~3u));
		}
		const auto base = PackU64(state, base_low, ctx.Arg(*handle, 1));
		address         = Binary(state, spv::OpIAdd, TypeU64(state), base,
		                         Unary(state, spv::OpUConvert, TypeU64(state), low));
	}
	auto immediate = static_cast<int32_t>(mem.offset);
	if (mem.kind == IR::ResourceKind::ScalarAddress) {
		immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
	}
	return immediate == 0
	           ? address
	           : Binary(state, spv::OpIAdd, TypeU64(state), address,
	                    ConstantU64(state, static_cast<uint64_t>(static_cast<int64_t>(immediate))));
}

uint32_t FaultElementPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.fault_buffer_variable, ConstantU32(state, 0), index);
	return pointer;
}

void RecordBdaFault(EmitterState& state, uint32_t page) {
	const auto word =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5));
	const auto bit =
	    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	const auto pointer = FaultElementPointer(state, word);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	state.builder.AddFunction(spv::OpStore, pointer,
	                          Binary(state, spv::OpBitwiseOr, TypeU32(state), value, bit));
}

// The store report (KYTY_RECORD_KEYED_VERIFY, BufferCache::BdaStoreReportCompiled): a store through a BDA pointer sets
// its page's bit in the second half of the fault buffer, and the host checks the page against the record-time listing.
bool BdaStoreTrackingEnabled() {
	static const bool enabled = BufferCache::BdaStoreReportCompiled();
	return enabled;
}

static bool RecordKeyedVerifyRequested();

void RecordBdaWrite(EmitterState& state, uint32_t address) {
	if (!BdaStoreTrackingEnabled()) {
		return;
	}
	const auto type     = TypeU64(state);
	const auto extended = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address,
	                             ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase));
	const auto packed   = Select(
	    state, type, extended,
	    Binary(state, spv::OpISub, type, address,
	           ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	    address);
	const auto page64   = Binary(state, spv::OpShiftRightLogical, type, packed,
	                             ConstantU32(state, BufferCache::CACHING_PAGEBITS));
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), page64,
	                             ConstantU64(state, BufferCache::CACHING_NUMPAGES));
	const auto page     = Select(state, TypeU32(state), in_range,
	                             Unary(state, spv::OpUConvert, TypeU32(state), page64), ConstantU32(state, 0));
	const auto word     = Binary(state, spv::OpIAdd, TypeU32(state),
	                             ConstantU32(state, static_cast<uint32_t>(BufferCache::CACHING_NUMPAGES / 32u)),
	                             Binary(state, spv::OpShiftRightLogical, TypeU32(state), page, ConstantU32(state, 5)));
	const auto bit      = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	                             Binary(state, spv::OpBitwiseAnd, TypeU32(state), page, ConstantU32(state, 31)));
	// Set the page bit only when it is not set yet: an atomic per store serializes every lane writing
	// the same page on one word (GPU at 100%, device lost while loading).
	const auto pointer = FaultElementPointer(state, word);
	const auto current = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), current, pointer);
	const auto missing = Binary(state, spv::OpIEqual, TypeBool(state),
	                            Binary(state, spv::OpBitwiseAnd, TypeU32(state), current, bit), ConstantU32(state, 0));

	EmitIfCondition(state, missing, [&]() {
		state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(),
		                          FaultElementPointer(state, word), ConstantU32(state, spv::ScopeDevice),
		                          ConstantU32(state, 0), bit);
	});
}

// KYTY_RECORD_KEYED_VERIFY=1 (it compiles the store report by itself): a store or atomic whose page has no BDA pointer
// (no buffer backs it, so it is dropped) still reports its page, so the verification also sees writes that a missing
// record-keyed candidate would have lost.
static bool RecordKeyedVerifyRequested() {
	static const bool on = Common::EnvRead::RecordKeyedVerify();
	return on;
}

static void RecordBdaTarget(EmitterState& state, uint32_t address) {
	if (RecordKeyedVerifyRequested()) RecordBdaWrite(state, address);
}

// The dword at a BDA address. A GLC/DLC access (MemoryInfo::coherent) gets a Coherent pointee and a Volatile access
// (BdaDwordAccess): a loop that polls a value another workgroup stores (32acc45f's look-back at the previous block's
// status through a GPU-built descriptor) then reads memory every time instead of a cached or hoisted copy.
uint32_t BdaDwordPointer(EmitterState& state, uint32_t bda, bool coherent) {
	auto pointer = state.builder.AllocateId();
	if (!coherent) {
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
		return pointer;
	}
	const auto block = state.builder.DecoratedType(
	    spv::OpTypeStruct, {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                        {spv::OpMemberDecorate, {0, spv::DecorationCoherent}},
	                        {spv::OpDecorate, {spv::DecorationBlock}}}, TypeU32(state));
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePointer(state, spv::StorageClassPhysicalStorageBuffer, block),
	                          pointer, bda);
	const auto base = pointer;
	pointer         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePhysicalU32Pointer(state), pointer, base, ConstantU32(state, 0));
	return pointer;
}

uint32_t BdaDwordAccess(bool coherent) {
	return spv::MemoryAccessAlignedMask | (coherent ? spv::MemoryAccessVolatileMask : spv::MemoryAccessMaskNone);
}

uint32_t LoadBdaDword(ValueEmitContext& ctx, uint32_t address, bool coherent = false) {
	auto&      state   = ctx.state;
	const auto bda     = GetBdaPointer(state, address);
	const auto present =
	    Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
	return EmitValueOrZeroIfCondition(state, present, [&]() {
		const auto         pointer   = BdaDwordPointer(state, bda, coherent);
		const auto         value     = state.builder.AllocateId();
		constexpr uint32_t alignment = sizeof(uint32_t);
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, BdaDwordAccess(coherent), alignment);
		return value;
	});
}

uint32_t LoadBda(ValueEmitContext& ctx, uint32_t address, uint32_t active, uint32_t bits,
                 bool coherent = false) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, active, [&]() {
		const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address,
		                            ConstantU64(state, ~uint64_t {3}));
		const auto first   = LoadBdaDword(ctx, aligned, coherent);
		const auto byte =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		           Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3));
		const auto crosses =
		    bits == 8u ? ConstantBool(state, false)
		               : Binary(state, bits == 16u ? spv::OpUGreaterThan : spv::OpINotEqual,
		                        TypeBool(state), byte, ConstantU32(state, bits == 16u ? 2u : 0u));
		const auto second = EmitValueOrZeroIfCondition(state, crosses, [&]() {
			return LoadBdaDword(ctx, Binary(state, spv::OpIAdd, TypeU64(state), aligned,
			                                ConstantU64(state, sizeof(uint32_t))), coherent);
		});
		const auto shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state), byte, ConstantU32(state, 3));
		const auto upper_shift =
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		           Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                  Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 4), byte),
		                  ConstantU32(state, 3)),
		           ConstantU32(state, 3));
		const auto merged =
		    Binary(state, spv::OpBitwiseOr, TypeU32(state),
		           Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, shift),
		           Binary(state, spv::OpShiftLeftLogical, TypeU32(state), second, upper_shift));
		return bits == 32u ? merged
		                   : Binary(state, spv::OpBitwiseAnd, TypeU32(state), merged,
		                            ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
	});
}

uint32_t ByteAddress(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (mem.kind == IR::ResourceKind::Buffer) {
		const auto resource = ctx.state.program.info.buffers.at(mem.resource).descriptor_index;
		const auto prefix = ctx.state.memory_byte_offsets[resource];
		const auto address = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state),
		                            BufferByteAddress(ctx, inst, mem), prefix);
		// The host binding prefix must not wrap an out-of-range guest address into the buffer.
		return Select(ctx.state, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), address, prefix),
		              address, ConstantU32(ctx.state, UINT32_MAX));
	}
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		if (mem.offset == 0u) {
			return ctx.Arg(inst, 0);
		}
		return Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), ctx.Arg(inst, 0),
		              ConstantU32(ctx.state, mem.offset));
	}
	if (mem.kind != IR::ResourceKind::Scratch) {
		EXIT("physical address memory must use the BDA emitter\n");
	}
	return ScratchByteAddress(ctx, mem, ctx.Arg(inst, 1), ctx.Arg(inst, 2));
}

uint32_t DwordIndex(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (ctx.state.compact_lds && mem.kind == IR::ResourceKind::Lds) {
		uint32_t base = 0;
		if (!CompactLdsBase(ctx.state, inst, base)) EXIT("compact function LDS access lost its ADDTID pattern\n");
		return ConstantU32(ctx.state, ctx.state.compact_lds_slots.at(base) * ctx.state.lane_count + ctx.state.lane_half);
	}
	auto address = ByteAddress(ctx, inst, mem);
	if (mem.kind == IR::ResourceKind::Lds || mem.kind == IR::ResourceKind::Gds) {
		// RDNA2 DS region addresses retain bits [15:2] after adding the byte offset.
		address = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                 ConstantU32(ctx.state, 0xffffu));
	}
	return Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state),
	              address, ConstantU32(ctx.state, 2));
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index);

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend);

// Raw dword buffer reads are branchless (ODCHUDZANIE-SHADEROW P3 / M21 step A; the only form, shader
// project step 1): ok = exec && in bounds; the element at Select(ok, index, 0) is read and Select(ok, value, 0) is the
// result. An active in-bounds lane reads the same element; every other lane gets the same 0. The read of element 0 is
// defined (robustBufferAccess, which the emitter relies on already) and has no side effect; its value is discarded.
// Measured 27-28.10: driver compile time of the 2.000 journals -24% on the same shaders; card time per
// frame at the gate, the inn and the meadow 1.1-3.0 ms lower, fps unchanged or higher. (Until 28.10 the cloth shader only,
// after a card-time loss on c6b0a54e on 17.10 that the 28.10 runs no longer show: 0.81 -> 0.82 ms.)
// Compute stages (EmitterState::guardless, the only way, step K2):
// a buffer access through a descriptor without a branch. robustBufferAccess2 (required, vulkanWindow.cpp) makes the card
// return 0 for a read and drop a write outside the descriptor's range, so an inactive lane or an out-of-range index goes
// past the end of the range (RobustPastEnd): the same result as the guards.
// Driver time follows the branches: 48 compute shaders 28.6 -> 21.7 s, the giants -33..-37%; each kind alone gave 4-7%, so they
// went together. Card: fps +1.3% +-0.4 open world, +0.6% +-0.3 gate. =2: 303 shaders in game, what differed differs in a
// same-module control too, or by order of work on the card (6065034f with each of 5 disjoint parts), or by 1 ULP (159dbbd7).
bool RobustBuffer(const EmitterState& state, const MemoryResourceAccess& resource) {
	return state.guardless &&
	       (resource.kind == IR::ResourceKind::Buffer || resource.kind == IR::ResourceKind::ScalarBuffer);
}

bool RobustBufferStore(const EmitterState& state, const IR::MemoryInfo& mem) {
	return state.guardless && mem.kind == IR::ResourceKind::Buffer;
}

// The element for an access that must not happen: fully past what the card still takes as the range. OpArrayLength rounds a
// range down and the card rounds it up to 4 bytes (robustStorageBufferAccessSizeAlignment), and the guest's ranges are no
// multiples of 4, so the element just past the end can be inside: the card did the store of an inactive lane
// (BufferStoreDwordAppliesHostOffset). 4 bytes past the last whole element clear it for every element size: 4 elements of 1
// byte, 2 of 2, 1 of 4 or 8.
uint32_t RobustPastEnd(EmitterState& state, const MemoryResourceAccess& resource) {
	return EmitAddU32(state, resource.length, ConstantU32(state, std::max(32u / std::max(resource.element_bits, 1u), 1u)));
}

uint32_t RobustElement(EmitterState& state, const MemoryResourceAccess& resource, uint32_t guard, uint32_t index) {
	auto ok = EmitMemoryElementInBounds(state, resource, index);
	if (guard != 0u && guard != ConstantBool(state, true)) ok = AndCondition(state, guard, ok);
	return Select(state, TypeU32(state), ok, index, RobustPastEnd(state, resource));
}

// Branchless: `guard` (or 0 for none) and the bounds test select the element read and the result.
uint32_t LoadWordBranchless(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                            const MemoryResourceAccess& resource, uint32_t guard) {
	auto&      state = ctx.state;
	const auto index = DwordIndex(ctx, inst, mem);
	if (RobustBuffer(state, resource)) return LoadWordInBounds(ctx, resource, RobustElement(state, resource, guard, index));
	auto       ok    = EmitMemoryElementInBounds(state, resource, index);
	if (guard != 0u) ok = Binary(state, spv::OpLogicalAnd, TypeBool(state), guard, ok);
	const auto safe  = Select(state, TypeU32(state), ok, index, ConstantU32(state, 0));
	return Select(state, TypeU32(state), ok, LoadWordInBounds(ctx, resource, safe), ConstantU32(state, 0));
}

uint32_t LoadWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource) {
	if (RobustBuffer(ctx.state, resource)) return LoadWordBranchless(ctx, inst, mem, resource, 0u);
	const auto index = DwordIndex(ctx, inst, mem);
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index),
	    [&]() { return LoadWordInBounds(ctx, resource, index); });
}

uint32_t LoadWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
	return LoadWordBranchless(ctx, inst, mem, resource, ctx.Arg(inst, inst.NumArgs() - 1));
}

uint32_t LoadSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const MemoryResourceAccess& resource, uint32_t bits,
                             bool sign_extend) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	if (RobustBuffer(ctx.state, resource))
		return LoadSubwordInBounds(ctx, resource, address, RobustElement(ctx.state, resource, 0u, index), bits, sign_extend);
	return EmitValueOrZeroIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		    return LoadSubwordInBounds(ctx, resource, address, index, bits, sign_extend);
	    });
}

uint32_t LoadWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                          uint32_t index) {
	const auto value   = ctx.state.builder.AllocateId();
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	return LoadResourceElement(ctx.state, TypeU32(ctx.state), value, pointer, resource, index);
}

uint32_t LoadSubwordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource,
                             uint32_t address, uint32_t index, uint32_t bits, bool sign_extend) {
	uint32_t value;
	if (resource.element_bits < 32u) {
		const auto id      = ctx.state.builder.AllocateId();
		const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
		const auto loaded  = LoadResourceElement(ctx.state, ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type, id,
		                                         pointer, resource, index);
		value = Unary(ctx.state, spv::OpUConvert, TypeU32(ctx.state), loaded);
	} else {
		const auto word = LoadWordInBounds(ctx, resource, index);
		const auto byte = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
		                         ConstantU32(ctx.state, 3));
		const auto shift = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), byte,
		                          ConstantU32(ctx.state, 3));
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
		               Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), word, shift),
		               ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu));
	}
	if (!sign_extend) return value;
	const auto left = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state), value,
	                         ConstantU32(ctx.state, 32u - bits));
	return Binary(ctx.state, spv::OpShiftRightArithmetic, TypeU32(ctx.state), left,
	              ConstantU32(ctx.state, 32u - bits));
}

uint32_t LoadSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits,
                     bool sign_extend) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return LoadSubwordPrepared(ctx, inst, mem, resource, bits, sign_extend);
	});
}

Prospero::BufferFormat BufferFormat(const ValueEmitContext& ctx, const IR::MemoryInfo& mem) {
	return mem.typed ? Format::DecodeTBufferFormat(mem.data_format, mem.number_format)
	                 : StorageBufferFormat(ctx.state, mem);
}

IR::MemoryInfo RebaseRawComponent(IR::MemoryInfo mem, uint32_t component) {
	mem.offset += component * 4u;
	mem.data_dwords     = 1u;
	mem.component_index = component;
	return mem;
}

using Format::FormattedSource;
using Format::FormattedSourceKind;

FormattedSource ResolveFormattedSource(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                       const Format::BufferFormatInfo& info,
                                       uint32_t                        output_component) {
	if (mem.typed) {
		return output_component < info.component_count
		           ? FormattedSource {FormattedSourceKind::Memory, output_component}
		           : FormattedSource {};
	}
	const auto selector = GetDstSel(ctx.state.program.info.buffers[mem.resource].descriptor_swizzle,
	                                output_component);
	const auto source = Format::ResolveFormattedSource(info, selector);
	if (source.kind == FormattedSourceKind::Invalid) {
		ExitDescriptorBindingFailure(ctx.state, IR::DescriptorBindingKind::Buffers, mem.resource,
		                             "buffer descriptor has reserved dst_sel");
	}
	return source;
}

uint32_t FormattedConstant(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                           FormattedSourceKind kind) {
	return ConstantU32(ctx.state, Format::FormattedConstantBits(info, kind));
}

template <typename LoadWordFn, typename LoadSubwordFn>
uint32_t LoadFormattedComponent(ValueEmitContext& ctx, const Format::BufferFormatInfo& info,
                                FormattedSource source, LoadWordFn&& load_word,
                                LoadSubwordFn&& load_subword) {
	if (source.kind != FormattedSourceKind::Memory) {
		return FormattedConstant(ctx, info, source.kind);
	}
	const auto component = source.component;
	const auto bits      = info.component_bits[component];
	auto raw = info.packed_bitfield || bits == 32u ? load_word(component)
	                                                : load_subword(component, bits);
	if (info.packed_bitfield || (bits < 32u && IsSignedFormatComponent(info.type))) {
		const auto type =
		    IsSignedFormatComponent(info.type) ? TypeI32(ctx.state) : TypeU32(ctx.state);
		const auto source_value =
		    type == TypeI32(ctx.state) ? Unary(ctx.state, spv::OpBitcast, type, raw) : raw;
		const auto extracted = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(IsSignedFormatComponent(info.type) ? spv::OpBitFieldSExtract
		                                                                 : spv::OpBitFieldUExtract,
		                              type, extracted, source_value,
		                              ConstantU32(ctx.state, info.packed_bitfield
		                                                         ? info.component_bit_offset[component]
		                                                         : 0u),
		                              ConstantU32(ctx.state, bits));
		raw = type == TypeI32(ctx.state)
		          ? Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), extracted)
		          : extracted;
	}
	return NormalizeFormatComponent(ctx.state, info, component, raw);
}

void StoreSubwordInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t address, uint32_t index,
                          uint32_t bits, uint32_t data) {
	const auto pointer = EmitMemoryElementPointer(ctx.state, resource, index);
	if (resource.element_bits < 32u) {
		const auto element = Unary(ctx.state, spv::OpUConvert, ctx.state.storage_buffers[bits == 8u ? 0 : 1].element_type, data);
		ctx.state.builder.AddFunction(spv::OpStore, pointer, element, resource.memory_access);
		return;
	}
	const auto shift   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), address,
	                                   ConstantU32(ctx.state, 3)),
	                            ConstantU32(ctx.state, 3));
	const auto mask    = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu), shift);
	const auto value   = Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
	                            Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), data,
	                                   ConstantU32(ctx.state, bits == 8u ? 0xffu : 0xffffu)),
	                            shift);
	const auto merge   = [&](uint32_t old) {
		return Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
		              Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), old,
		                     Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), mask)),
		              value);
	};
	if (mem.kind == IR::ResourceKind::Scratch) {
		const auto old = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), old, pointer);
		ctx.state.builder.AddFunction(spv::OpStore, pointer, merge(old));
	} else {
		AtomicUpdate(ctx.state, pointer, mem.kind, merge);
	}
}

void StoreSubwordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                          const MemoryResourceAccess& resource, uint32_t bits, uint32_t data, uint32_t guard = 0u) {
	const auto address = ByteAddress(ctx, inst, mem);
	const auto index = Binary(
	    ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), address,
	    ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	if (RobustBuffer(ctx.state, resource) && resource.element_bits < 32u) { // (a plain store; the 32-bit form is an atomic update)
		StoreSubwordInBounds(ctx, mem, resource, address, RobustElement(ctx.state, resource, guard, index), bits, data);
		return;
	}
	EmitIfCondition(
	    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		    StoreSubwordInBounds(ctx, mem, resource, address, index, bits, data);
	    });
}

void StoreSubword(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem, uint32_t bits) {
	if (RobustBufferStore(ctx.state, mem) && IR::StorageBufferElementBits(ctx.state.program, mem) < 32u) {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		StoreSubwordPrepared(ctx, inst, mem, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2), ctx.Arg(inst, inst.NumArgs() - 1));
		return;
	}
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		StoreSubwordPrepared(ctx, inst, mem, resource, bits, ctx.Arg(inst, inst.NumArgs() - 2));
	});
}

void StoreWordInBounds(ValueEmitContext& ctx, const MemoryResourceAccess& resource, uint32_t index,
                       uint32_t data) {
	ctx.state.builder.AddFunction(spv::OpStore,
	                              EmitMemoryElementPointer(ctx.state, resource, index), data,
	                              resource.memory_access);
}

template <typename Fn>
void ForEachLocalFlatAccess(ValueEmitContext& ctx, const IR::Inst& inst, Fn&& emit) {
	auto& state = ctx.state;
	auto mem = ctx.Memory(inst);
	auto high = ctx.Arg(inst, 2);
	const auto low = AddAddressOffset(state, mem, ctx.Arg(inst, 1), high);
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), low, ConstantU32(state, 2));
	const auto aligned = Binary(state, spv::OpIEqual, TypeBool(state), ConstantU32(state, 0),
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), low, ConstantU32(state, 3)));
	for (const auto kind: {IR::ResourceKind::Scratch, IR::ResourceKind::Lds}) {
		mem.kind = kind;
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto aperture = kind == IR::ResourceKind::Scratch ? Decoder::PrivateApertureHigh
		                                                       : Decoder::SharedApertureHigh;
		const auto selected = Binary(state, spv::OpIEqual, TypeBool(state), high, ConstantU32(state, aperture));
		const auto valid = AndCondition(state, aligned,
		    AndCondition(state, selected, EmitMemoryElementInBounds(state, resource, index)));
		emit(resource, index, valid);
	}
}

uint32_t LoadLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		uint32_t result = 0;
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			const auto value = EmitValueOrZeroIfCondition(ctx.state, valid, [&]() {
				return LoadWordInBounds(ctx, resource, index);
			});
			result = result == 0 ? value : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), result, value);
		});
		return result;
	});
}

void StoreLocalFlat(ValueEmitContext& ctx, const IR::Inst& inst) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		ForEachLocalFlatAccess(ctx, inst, [&](const auto& resource, uint32_t index, uint32_t valid) {
			EmitIfCondition(ctx.state, valid, [&]() {
				StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, inst.NumArgs() - 2));
			});
		});
	});
}

void StoreWordPrepared(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t data, uint32_t guard = 0u) {
	const auto index = DwordIndex(ctx, inst, mem);
	if (RobustBuffer(ctx.state, resource)) {
		StoreWordInBounds(ctx, resource, RobustElement(ctx.state, resource, guard, index), data);
		return;
	}
	EmitIfCondition(ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
		StoreWordInBounds(ctx, resource, index, data);
	});
}

void StoreWord(ValueEmitContext& ctx, const IR::Inst& inst, IR::MemoryInfo mem) {
	auto& state = ctx.state;
	const auto body = [&](uint32_t guard) {
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto store = [&](uint32_t resource) {
			if (mem.kind == IR::ResourceKind::Buffer) {
				const auto& buffer = state.program.info.buffers[resource];
				if (buffer.packed_stride == 0u &&
				    buffer.source < state.program.descriptor_sources.size() &&
				    state.program.descriptor_sources[buffer.source].indirect_descriptor) return;
			}
			mem.resource = resource;
			const auto access = PrepareMemoryResourceAccess(state, mem);
			StoreWordPrepared(ctx, inst, mem, access, data, guard);
		};
		if (mem.kind != IR::ResourceKind::Buffer ||
		    state.program.info.buffers[mem.resource].indirect_root != mem.resource) {
			store(mem.resource);
			return;
		}
		const auto& buffer = state.program.info.buffers[mem.resource];
		const auto* handle = inst.Arg(0).ResolveInstruction();
		if (handle == nullptr || buffer.indirect_resources.empty() ||
		    state.flattened_srt_variable == 0 || buffer.indirect_search_iterations == 0u) {
			ctx.Fail(inst, "has no indirect buffer runtime mapping");
			return;
		}
		const auto selected = EmitIndirectResourceIndex(
		    state, ctx.Arg(*handle, 0), buffer.indirect_mapping_offset,
		    buffer.indirect_search_iterations, UINT32_MAX);
		const auto merge = state.builder.AllocateId();
		std::vector<uint32_t> labels(buffer.indirect_resources.size());
		std::vector<uint32_t> branches {spv::OpSwitch, selected, merge};
		for (uint32_t i = 0; i < labels.size(); ++i) {
			labels[i] = state.builder.AllocateId();
			branches.push_back(i);
			branches.push_back(labels[i]);
		}
		state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
		state.builder.AddFunction(branches);
		for (uint32_t i = 0; i < labels.size(); ++i) {
			EmitLabel(state, labels[i]);
			store(buffer.indirect_resources[i]);
			state.builder.AddFunction(spv::OpBranch, merge);
		}
		EmitLabel(state, merge);
	};
	const auto guard = ctx.Arg(inst, inst.NumArgs() - 1);
	if (RobustBufferStore(state, mem) && state.program.info.buffers[mem.resource].indirect_root != mem.resource) {
		body(guard);
		return;
	}
	EmitIfCondition(state, guard, [&]() { body(0u); });
}

// The lanes of a guest wave see each other's LDS writes in program order without S_BARRIER (the wave runs in lockstep),
// but the two host subgroups of a paired wave do not run together. The blur CS 099c56ee21c1665b writes its lane's texels
// to LDS and reads lanes +-1..4 away with no barrier: lanes 28..31 read 32..35 of the other half before they were written
// -> wrong texels, the black particles of the first pair rule (found by pairing subsets of the rule's shaders;
// =2: KYTY_DIAG_TWIN_DISPATCH of every dispatch, 0 differences of 4000, 3993 before). In a paired shader of one wave per
// workgroup (the barrier form) a workgroup barrier goes before an LDS write when an LDS read came since the last one (the
// other half's read must happen before the overwrite) and after every LDS write or atomic (its later reads see it).
bool PairLdsSync(const EmitterState& state, const IR::MemoryInfo& mem) {
	return state.wave_pair && state.pair_barrier && state.guard_depth == 0u && mem.kind == IR::ResourceKind::Lds &&
	       state.program.stage == ShaderType::Compute;
}

void EmitPairLdsBarrier(EmitterState& state) {
	const auto memory = state.lds_storage_class == spv::StorageClassStorageBuffer ? spv::MemorySemanticsUniformMemoryMask
	                                                                              : spv::MemorySemanticsWorkgroupMemoryMask;
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | memory));
	state.lds_pair_synced = true;
}

// Around an LDS write or atomic (`before` true: ahead of it).
void EmitPairLdsWriteSync(EmitterState& state, const IR::MemoryInfo& mem, bool before) {
	if (!PairLdsSync(state, mem) || (before && state.lds_pair_synced)) return;
	EmitPairLdsBarrier(state);
}

template <typename Fn>
uint32_t EmitAtomicAccessIf(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                            uint32_t condition, Fn&& operation, bool single_operation = false) {
	if (single_operation && RobustBufferStore(ctx.state, mem)) {
		// (RobustBuffer: the atomic past the end has no effect, but its result is undefined, so the bounds test stays as a
		// value. Not for a compare-exchange loop, which would spin on that result.)
		auto&      state    = ctx.state;
		const auto index    = DwordIndex(ctx, inst, mem);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto ok       = AndCondition(state, condition, EmitMemoryElementInBounds(state, resource, index));
		const auto safe     = Select(state, TypeU32(state), ok, index, RobustPastEnd(state, resource));
		return Select(state, TypeU32(state), ok, operation(EmitMemoryElementPointer(state, resource, safe)), ConstantU32(state, 0));
	}
	return EmitValueOrZeroIfCondition(ctx.state, condition, [&]() {
		const auto index = DwordIndex(ctx, inst, mem);
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		return EmitValueOrZeroIfCondition(
		    ctx.state, EmitMemoryElementInBounds(ctx.state, resource, index), [&]() {
			    return operation(EmitMemoryElementPointer(ctx.state, resource, index));
		    });
	});
}

template <typename Fn>
uint32_t EmitAtomicUpdate(ValueEmitContext& ctx, const IR::Inst& inst,
                          const IR::MemoryInfo& mem, Fn&& replacement) {
	const auto value = ctx.Arg(inst, inst.NumArgs() - 2);
	EmitPairLdsWriteSync(ctx.state, mem, true);
	const auto old = EmitAtomicAccessIf(ctx, inst, mem, ctx.Arg(inst, inst.NumArgs() - 1), [&](uint32_t pointer) {
		return AtomicUpdate(ctx.state, pointer, mem.kind, [&](uint32_t old) {
			return replacement(ctx.state, old, value);
		});
	});
	EmitPairLdsWriteSync(ctx.state, mem, false);
	return old;
}

uint32_t AtomicIncrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old >= limit ? 0 : old + 1 (unsigned).
	const auto wrap = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), old, limit);
	const auto next = Binary(state, spv::OpIAdd, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, ConstantU32(state, 0), next);
}

uint32_t AtomicDecrement(EmitterState& state, uint32_t old, uint32_t limit) {
	// old == 0 || old > limit ? limit : old - 1 (unsigned).
	const auto zero  = Binary(state, spv::OpIEqual, TypeBool(state), old, ConstantU32(state, 0));
	const auto above = Binary(state, spv::OpUGreaterThan, TypeBool(state), old, limit);
	const auto wrap  = Binary(state, spv::OpLogicalOr, TypeBool(state), zero, above);
	const auto next  = Binary(state, spv::OpISub, TypeU32(state), old, ConstantU32(state, 1));
	return Select(state, TypeU32(state), wrap, limit, next);
}

struct PreparedFormattedMemory {
	Format::BufferFormatInfo info;
	MemoryResourceAccess     resource;
	uint32_t                 base = 0;
	uint32_t                 in_bounds = 0;
};

PreparedFormattedMemory PrepareFormattedMemory(ValueEmitContext& ctx, const IR::Inst& inst,
                                               const IR::MemoryInfo& mem,
                                               const MemoryResourceAccess& resource,
                                               const Format::BufferFormatInfo& info) {
	PreparedFormattedMemory plan;
	plan.info      = info;
	plan.resource  = resource;
	const auto start = ByteAddress(ctx, inst, mem);
	// Check the whole, unaligned format span before aligning the actual memory address.
	const auto end = Binary(ctx.state, spv::OpIAdd, TypeU32(ctx.state), start,
	                        ConstantU32(ctx.state, info.byte_size - 1u));
	const auto last_index = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), end,
	                               ConstantU32(ctx.state, std::countr_zero(resource.element_bits / 8u)));
	plan.in_bounds = AndCondition(
	    ctx.state, Binary(ctx.state, spv::OpUGreaterThanEqual, TypeBool(ctx.state), end, start),
	    EmitMemoryElementInBounds(ctx.state, plan.resource, last_index));
	plan.base = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), start,
	                   ConstantU32(ctx.state, ~(std::min(4u, info.byte_size) - 1u)));
	return plan;
}

std::pair<uint32_t, uint32_t> FormattedComponentAddress(EmitterState& state,
                                                       const PreparedFormattedMemory& plan,
                                                       uint32_t component) {
	// Components advance sequentially from one address; do not reapply buffer swizzling.
	const auto offset = Format::GetFormatComponentByteOffset(plan.info, component);
	const auto address = offset == 0u ? plan.base : Binary(
	    state, spv::OpIAdd, TypeU32(state), plan.base, ConstantU32(state, offset));
	return {address, Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                        ConstantU32(state, std::countr_zero(plan.resource.element_bits / 8u)))};
}

uint32_t LoadFormattedInBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                               const PreparedFormattedMemory& plan, uint32_t output_component) {
	return LoadFormattedComponent(
	    ctx, plan.info, ResolveFormattedSource(ctx, mem, plan.info, output_component),
	    [&](uint32_t component) {
		    return LoadWordInBounds(ctx, plan.resource,
		                            FormattedComponentAddress(ctx.state, plan, component).second);
	    },
	    [&](uint32_t component, uint32_t bits) {
		    const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
		    return LoadSubwordInBounds(ctx, plan.resource, address, index, bits, false);
	    });
}

uint32_t ConstructU32Composite(EmitterState& state, uint32_t components,
                               const std::array<uint32_t, 4>& values) {
	if (components == 1u) return values[0];
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {spv::OpCompositeConstruct, TypeU32Composite(state, components),
	                             result};
	words.insert(words.end(), values.begin(), values.begin() + components);
	state.builder.AddFunction(words);
	return result;
}

// FORMAT_D16: each channel becomes 16 bits (half for float-class formats; integer formats
// saturate to the 16-bit range); channel pairs share one dword, low channel first.
uint32_t PackD16Channels(EmitterState& state, const Format::BufferFormatInfo& info,
                         const std::array<uint32_t, 4>& channels, uint32_t count, uint32_t dwords) {
	const bool integer = info.type == Format::ComponentType::Uint ||
	                     info.type == Format::ComponentType::Sint;
	const auto narrow = [&](uint32_t value) {
		if (info.type == Format::ComponentType::Uint) return EmitUMin32(state, value, ConstantU32(state, 0xffffu));
		const auto lower = EmitSMax32(state, value, ConstantU32(state, 0xffff8000u));
		return Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		              EmitSMin32(state, lower, ConstantU32(state, 0x7fffu)), ConstantU32(state, 0xffffu));
	};
	std::array<uint32_t, 4> words {};
	for (uint32_t word = 0; word < dwords; ++word) {
		const auto low  = channels[word * 2u];
		const auto high = word * 2u + 1u < count ? channels[word * 2u + 1u] : ConstantU32(state, 0);
		if (integer) {
			words[word] = Binary(state, spv::OpBitwiseOr, TypeU32(state), narrow(low),
			                     Binary(state, spv::OpShiftLeftLogical, TypeU32(state), narrow(high),
			                            ConstantU32(state, 16u)));
		} else {
			const auto pair = EmitCompositeConstructF32x2(state, EmitBitCastF32U32(state, low),
			                                              EmitBitCastF32U32(state, high));
			words[word] = EmitPackHalf2x16(state, pair);
		}
	}
	return ConstructU32Composite(state, dwords, words);
}

// The null formatted share: with a run-time stride the binding may hold a null V# that shares its live
// specialization; a zero run-time stride is that null V#, whose out-of-bounds result is zero in every channel (its
// DST_SEL is 0), not `oob` (the live specialization's constants). Without the switch, or with a fixed stride: `oob`.
uint32_t NullFormattedOutOfBounds(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t components, uint32_t oob) {
	auto&      state  = ctx.state;
	const auto packed = StorageBufferPackedStride(state, mem);
	if ((packed & (1u << 31u)) == 0u) return oob;
	const auto stride_dword = state.program.bindings.memory_stride_dword;
	if (stride_dword == IR::PushData::NoStart) EXIT("dynamic buffer stride has no shader-data slot\n");
	const auto binding = ResourceForDescriptor(state, IR::DescriptorBindingKind::Buffers, mem.resource);
	const auto live    = Binary(state, spv::OpINotEqual, TypeBool(state),
	                            EmitShaderDataDwordLoad(state, stride_dword + binding), ConstantU32(state, 0));
	return EmitValueOrDefaultIfCondition(state, live, TypeU32Composite(state, components),
	                                     ConstantU32CompositeZero(state, components), [&]() { return oob; });
}

uint32_t FormattedOutOfBoundsValue(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                   const PreparedFormattedMemory& plan, uint32_t components) {
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; component++) {
		const auto source = ResolveFormattedSource(ctx, mem, plan.info, component);
		values[component] = FormattedConstant(ctx, plan.info, source.kind);
	}
	return NullFormattedOutOfBounds(ctx, mem, components, ConstructU32Composite(ctx.state, components, values));
}

uint32_t FormattedLoad(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto info = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			return LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource);
		}
		const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
		const auto value = EmitValueOrDefaultIfCondition(
		    ctx.state, plan.in_bounds, TypeU32(ctx.state),
		    FormattedOutOfBoundsValue(ctx, mem, plan, 1u),
		    [&]() { return LoadFormattedInBounds(ctx, mem, plan, 0u); });
		if (mem.data_bits != 16u) {
			return value;
		}
		if (info.type == Format::ComponentType::Uint) {
			return EmitUMin32(ctx.state, value, ConstantU32(ctx.state, 0xffffu));
		}
		if (info.type == Format::ComponentType::Sint) {
			const auto lower = EmitSMax32(ctx.state, value, ConstantU32(ctx.state, 0xffff8000u));
			return EmitSMin32(ctx.state, lower, ConstantU32(ctx.state, 0x7fffu));
		}
		return EmitConvertF16F32(ctx.state, EmitBitCastF32U32(ctx.state, value));
	});
}

// EncodeFormattedStoreComponent (spirvEmitterMemoryHelpers.cpp, shared with image stores and exports since upstream c11254a99):
// the same conversion as before with its default round-to-even.

void StoreFormattedPrepared(ValueEmitContext& ctx, const IR::Inst& inst,
                             const IR::MemoryInfo& mem, const MemoryResourceAccess& resource,
                             const Format::BufferFormatInfo& info, uint32_t data,
                             uint32_t components) {
	// RDNA2 formatted stores transfer the entire format, zero-filling missing VGPRs.
	if (mem.data_bits == 16u) {
		// FORMAT_D16: unpack the 16-bit channels (half floats, or integers for integer formats).
		const bool integer = info.type == Format::ComponentType::Uint ||
		                     info.type == Format::ComponentType::Sint;
		const bool is_signed = info.type == Format::ComponentType::Sint;
		const auto dwords = (mem.component_count + 1u) / 2u;
		std::array<uint32_t, 4> channels {};
		for (uint32_t channel = 0; channel < mem.component_count; ++channel) {
			auto word = data;
			if (dwords > 1u) {
				word = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), word, data,
				                              channel / 2u);
			}
			if (integer) {
				channels[channel] = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(is_signed ? spv::OpBitFieldSExtract : spv::OpBitFieldUExtract,
				                              TypeU32(ctx.state), channels[channel], word,
				                              ConstantU32(ctx.state, (channel % 2u) * 16u),
				                              ConstantU32(ctx.state, 16u));
			} else {
				const auto pair = EmitGlsl<GLSLstd450UnpackHalf2x16, IR::Type::F32x2>(ctx.state, word);
				const auto value = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(ctx.state), value, pair,
				                              channel % 2u);
				channels[channel] = Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
			}
		}
		IR::MemoryInfo full = mem;
		full.data_bits = 32u;
		StoreFormattedPrepared(ctx, inst, full, resource, info,
		                       ConstructU32Composite(ctx.state, mem.component_count, channels),
		                       mem.component_count);
		return;
	}
	const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
	EmitIfCondition(ctx.state, plan.in_bounds, [&]() {
		auto packed = ConstantU32(ctx.state, 0);
		for (uint32_t component = 0; component < info.component_count; component++) {
			auto value = ConstantU32(ctx.state, 0);
			if (component < components) {
				value = data;
				if (components != 1u) {
					value = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
					                              value, data, component);
				}
			}
			value = EncodeFormattedStoreComponent(ctx.state, info, component, value, mem.data_bits);
			const auto bits = info.component_bits[component];
			if (info.packed_bitfield) {
				packed = EmitBitFieldInsert(
				    ctx.state, packed, value,
				    ConstantU32(ctx.state, info.component_bit_offset[component]),
				    ConstantU32(ctx.state, bits));
			} else {
				const auto [address, index] = FormattedComponentAddress(ctx.state, plan, component);
				if (bits == 8u || bits == 16u) {
					StoreSubwordInBounds(ctx, mem, plan.resource, address, index, bits, value);
				} else {
					StoreWordInBounds(ctx, plan.resource, index, value);
				}
			}
		}
		if (info.packed_bitfield) {
			StoreWordInBounds(ctx, plan.resource,
			                  FormattedComponentAddress(ctx.state, plan, 0u).second, packed);
		}
	});
}

void FormattedStore(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto resource = PrepareMemoryResourceAccess(ctx.state, mem);
		const auto data     = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info     = Format::GetFormatInfo(BufferFormat(ctx, mem));
		if (info.type == Format::ComponentType::Unknown) {
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, 0u), resource, data);
			return;
		}
		StoreFormattedPrepared(ctx, inst, mem, resource, info, data, 1u);
	});
}

// KYTY_PERF_BDA_ENTRY_SHARE: LoadBdaDword for a scalar load through a GPU-built V#, reusing the page-table entry of
// this invocation half's previous such load when the address is in the same 16 KiB page. The cloth shader fc6f8c56
// reads 27 dwords of one collider record per loop iteration through 7 S_BUFFER_LOADs:
// 27 calls of get_bda_pointer (page-table load, range checks, fault branch) for the same page. Exact: the pointer of
// a page is entry_base + (address & page mask) with the entry the host wrote before the submission (it does not change
// during a dispatch); a reused base came from a real lookup of that page by the same invocation that found it present,
// so get_bda_pointer would return the same pointer and record nothing new (a missing page, base 0, is looked up again
// and reported as before). =2 verify: the old lookup runs too and is used; checks and pointer mismatches are counted.
uint32_t LoadBdaDwordShared(ValueEmitContext& ctx, uint32_t address) {
	auto&          state = ctx.state;
	const uint32_t half  = ctx.half < 2u ? ctx.half : 0u;
	if (state.bda_memo_base[half] == 0u) {
		return LoadBdaDword(ctx, address);
	}
	const auto u64       = TypeU64(state);
	const auto low       = Binary(state, spv::OpBitwiseAnd, u64, address, ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1));
	const auto page      = Binary(state, spv::OpShiftRightLogical, u64, address, ConstantU32(state, BufferCache::CACHING_PAGEBITS));
	const auto memo_page = Unary(state, spv::OpLoad, u64, state.bda_memo_page[half]);
	const auto memo_base = Unary(state, spv::OpLoad, u64, state.bda_memo_base[half]);
	const auto hit       = Binary(state, spv::OpLogicalAnd, TypeBool(state),
	                              Binary(state, spv::OpIEqual, TypeBool(state), page, memo_page),
	                              Binary(state, spv::OpINotEqual, TypeBool(state), memo_base, ConstantU64(state, 0)));
	const auto hit_label   = state.builder.AllocateId();
	const auto miss_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, hit, hit_label, miss_label);
	EmitLabel(state, hit_label);
	const auto shared = Binary(state, spv::OpIAdd, u64, memo_base, low);
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, miss_label);
	const auto looked = GetBdaPointer(state, address);
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	auto bda = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, u64, bda, shared, hit_label, looked, miss_label);
	if (BufferCache::BdaEntryShareMode() == 2) {
		const auto old      = GetBdaPointer(state, address);
		const auto mismatch = Binary(state, spv::OpINotEqual, TypeBool(state), old, bda);
		(void)FaultDiagAdd(state, BufferCache::FAULT_DIAG_ENTRY_SHARE, ConstantU32(state, 1));
		(void)FaultDiagAdd(state, BufferCache::FAULT_DIAG_ENTRY_SHARE + 1,
		                   Select(state, TypeU32(state), mismatch, ConstantU32(state, 1), ConstantU32(state, 0)));
		bda = old;
	}
	const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
	state.builder.AddFunction(spv::OpStore, state.bda_memo_page[half], page);
	state.builder.AddFunction(spv::OpStore, state.bda_memo_base[half],
	                          Select(state, u64, present, Binary(state, spv::OpISub, u64, bda, low), ConstantU64(state, 0)));
	return EmitValueOrZeroIfCondition(state, present, [&]() {
		const auto         pointer   = BdaDwordPointer(state, bda, false);
		const auto         value     = state.builder.AllocateId();
		constexpr uint32_t alignment = sizeof(uint32_t);
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer, BdaDwordAccess(false), alignment);
		return value;
	});
}

// One dword of a scalar load through a GPU-built V# (`handle`), at dynamic offset `dynamic_offset` (bytes) plus the
// load's constant `memory_offset`: 0 past the V#'s size, else the dword through the BDA page table.
uint32_t LoadIndirectScalarDword(ValueEmitContext& ctx, const IR::Inst& handle, uint32_t dynamic_offset,
                                 uint32_t memory_offset);

uint32_t LoadIndirectScalarBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	return LoadIndirectScalarDword(ctx, *inst.Arg(0).ResolveInstruction(), ctx.Arg(inst, 1), ctx.Memory(inst).offset);
}

uint32_t LoadIndirectScalarDword(ValueEmitContext& ctx, const IR::Inst& handle, uint32_t dynamic_offset,
                                 uint32_t memory_offset) {
	auto&       state  = ctx.state;
	const auto  word1  = ctx.Arg(handle, 1);
	const auto  stride = EmitBitFieldUExtract(state, word1, ConstantU32(state, 16),
	                                         ConstantU32(state, 14));
	const auto  size   = Binary(state, spv::OpIMul, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  EmitUMax32(state, stride, ConstantU32(state, 1))),
	                            Unary(state, spv::OpUConvert, TypeU64(state), ctx.Arg(handle, 2)));
	const auto  offset = Binary(state, spv::OpIAdd, TypeU64(state),
	                            Unary(state, spv::OpUConvert, TypeU64(state),
	                                  Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                                         dynamic_offset, ConstantU32(state, ~3u))),
	                            ConstantU64(state, memory_offset & ~3u));
	const auto  end =
	    Binary(state, spv::OpIAdd, TypeU64(state), offset, ConstantU64(state, sizeof(uint32_t)));
	return EmitValueOrZeroIfCondition(
	    state, Binary(state, spv::OpULessThanEqual, TypeBool(state), end, size), [&]() {
		    const auto base = PackU64(state,
		                              Binary(state, spv::OpBitwiseAnd, TypeU32(state),
		                                     ctx.Arg(handle, 0), ConstantU32(state, ~3u)),
		                              Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1,
		                                     ConstantU32(state, 0xffffu)));
		    return LoadBdaDwordShared(ctx, Binary(state, spv::OpIAdd, TypeU64(state), base, offset));
	    });
}

// get_bda_pointer subtracts whole pages: two addresses in one guest page give one page index and one entry.
static_assert(LibKernel::Memory::kExtendedMemoryBase % BufferCache::CACHING_PAGESIZE == 0u);
static_assert((LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE) % BufferCache::CACHING_PAGESIZE == 0u);

// The leader of a group (FindScalarLoadGroups; cloth 6.72 -> 5.04 ms) loads every member's dword. Member k reads
// base + dyn + c_k (dyn = dynamic offset & ~3, c_k = constant offset & ~3) when dyn + c_k + 4 <= size, else 0. When the
// highest member is in bounds (then all are) and the members' span lies in the page of the lowest one, one
// get_bda_pointer of that address gives every member's pointer as pointer + (c_k - c_min) (the page's entry is
// base + offset in page; a missing page gives 0 for every member, as it would for each, and its fault bit is the same
// bit); otherwise each member takes the per-dword path.
void EmitScalarLoadGroup(ValueEmitContext& ctx, const IR::Inst& leader, const EmitterState::ScalarLoadGroup& group) {
	auto&          state   = ctx.state;
	const uint32_t half    = ctx.half < 2u ? ctx.half : 0u;
	const auto&    handle  = *leader.Arg(0).ResolveInstruction();
	const auto     dynamic = ctx.Arg(leader, 1);
	const auto     u64     = TypeU64(state);
	const auto     count   = group.members.size();
	if (half == 0u) {
		for (const auto* member: group.members) {
			state.scalar_group_values[1].erase(member);
		}
	}
	const auto word1  = ctx.Arg(handle, 1);
	const auto stride = EmitBitFieldUExtract(state, word1, ConstantU32(state, 16), ConstantU32(state, 14));
	const auto size   = Binary(state, spv::OpIMul, u64,
	                           Unary(state, spv::OpUConvert, u64, EmitUMax32(state, stride, ConstantU32(state, 1))),
	                           Unary(state, spv::OpUConvert, u64, ctx.Arg(handle, 2)));
	const auto dyn    = Unary(state, spv::OpUConvert, u64,
	                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), dynamic, ConstantU32(state, ~3u)));
	const auto base   = PackU64(state, Binary(state, spv::OpBitwiseAnd, TypeU32(state), ctx.Arg(handle, 0), ConstantU32(state, ~3u)),
	                            Binary(state, spv::OpBitwiseAnd, TypeU32(state), word1, ConstantU32(state, 0xffffu)));
	const auto first  = Binary(state, spv::OpIAdd, u64, Binary(state, spv::OpIAdd, u64, base, dyn),
	                           ConstantU64(state, group.c_min));
	const auto in_bounds =
	    Binary(state, spv::OpULessThanEqual, TypeBool(state),
	           Binary(state, spv::OpIAdd, u64, dyn, ConstantU64(state, group.c_max + sizeof(uint32_t))), size);
	const auto in_page = Binary(
	    state, spv::OpULessThanEqual, TypeBool(state),
	    Binary(state, spv::OpIAdd, u64,
	           Binary(state, spv::OpBitwiseAnd, u64, first, ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1)),
	           ConstantU64(state, group.c_max - group.c_min + sizeof(uint32_t))),
	    ConstantU64(state, BufferCache::CACHING_PAGESIZE));
	const auto fast = Binary(state, spv::OpLogicalAnd, TypeBool(state), in_bounds, in_page);

	const auto fast_label  = state.builder.AllocateId();
	const auto slow_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, fast, fast_label, slow_label);

	EmitLabel(state, fast_label);
	const auto pointer = GetBdaPointer(state, first);
	const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), pointer, ConstantU64(state, 0));
	const auto header  = state.current_label;
	const auto load_label  = state.builder.AllocateId();
	const auto loaded_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, loaded_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, present, load_label, loaded_label);
	EmitLabel(state, load_label);
	std::vector<uint32_t> loads(count);
	for (size_t k = 0; k < count; k++) {
		const auto address = group.deltas[k] == 0u
		                         ? pointer
		                         : Binary(state, spv::OpIAdd, u64, pointer, ConstantU64(state, group.deltas[k]));
		const auto         dword_pointer = BdaDwordPointer(state, address, false);
		constexpr uint32_t alignment     = sizeof(uint32_t);
		loads[k]                         = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), loads[k], dword_pointer, BdaDwordAccess(false), alignment);
	}
	state.builder.AddFunction(spv::OpBranch, loaded_label);
	EmitLabel(state, loaded_label);
	std::vector<uint32_t> fast_values(count);
	for (size_t k = 0; k < count; k++) {
		fast_values[k] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, TypeU32(state), fast_values[k], loads[k], load_label, ConstantU32(state, 0),
		                          header);
	}
	const auto fast_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);

	EmitLabel(state, slow_label);
	std::vector<uint32_t> slow_values(count);
	for (size_t k = 0; k < count; k++) {
		slow_values[k] = LoadIndirectScalarDword(ctx, handle, dynamic, static_cast<uint32_t>(group.c_min + group.deltas[k]));
	}
	const auto slow_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);

	EmitLabel(state, merge_label);
	std::vector<uint32_t> values(count);
	for (size_t k = 0; k < count; k++) {
		values[k] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, TypeU32(state), values[k], fast_values[k], fast_exit, slow_values[k],
		                          slow_exit);
	}
	for (size_t k = 0; k < count; k++) {
		state.scalar_group_values[half][group.members[k]] = values[k];
	}
	ctx.Define(leader, values[0]);
}

struct IndirectBufferAccess {
	uint32_t address;
	uint32_t offset;
	uint32_t stride;
	uint32_t swizzle;
	uint32_t format;
	uint32_t selector; // word 3's DST_SEL X of a formatted access: no reader left, its extract stays in the SPIR-V
	uint32_t mode;
	uint32_t records;
	uint32_t index_in_bounds;
	uint32_t scalar_in_bounds;
	uint32_t raw_records;
	uint32_t raw_index_in_bounds;
	bool     coherent = false; // GLC/DLC: BdaDwordPointer / BdaDwordAccess
};

IndirectBufferAccess PrepareIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto word1   = ctx.Arg(handle, 1);
	const auto records = ctx.Arg(handle, 2);
	const auto word3   = ctx.Arg(handle, 3);
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride  = field(word1, 16, 14);
	const auto swizzle = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto index = Binary(
	    state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	    Select(state, TypeU32(state), nonzero(field(word3, 23, 1)), BufferLane(state),
	           ConstantU32(state, 0)));
	const auto soffset = ctx.Arg(inst, 3);
	const auto address = CalculateBufferAddress(state, index, ctx.Arg(inst, 2), soffset,
	                                            ctx.Memory(inst).offset, stride, swizzle,
	                                            field(word3, 21, 2));
	const auto base        = PackU64(state, ctx.Arg(handle, 0), field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	return {
	    .address          = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                               Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset           = address.offset,
	    .stride           = stride,
	    .swizzle          = swizzle,
	    .format           = field(word3, 12, 7),
	    .selector         = ctx.Memory(inst).formatted ? field(word3, 0, 3) : 0u,
	    .mode             = field(word3, 28, 2),
	    .records          = records,
	    .index_in_bounds  = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records      = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	    .coherent         = ctx.Memory(inst).coherent,
	};
}

uint32_t IndirectBufferInBounds(EmitterState& state, const IndirectBufferAccess& buffer,
                                uint32_t displacement, uint32_t bytes, bool formatted) {
	const auto offset = displacement == 0u ? buffer.offset : Binary(
	    state, spv::OpIAdd, TypeU32(state), buffer.offset, ConstantU32(state, displacement));
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size,
		                  ConstantU32(state, bytes)),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, ConstantU32(state, bytes))));
	};
	const auto structured = AndCondition(
	    state, buffer.index_in_bounds,
	    formatted ? has_payload(buffer.stride)
	              : Binary(state, spv::OpULessThan, TypeBool(state), offset, buffer.stride));
	// OOB_SELECT=3 reduces NUM_RECORDS by SOFFSET before its offset/index checks.
	const auto raw = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           has_payload(buffer.raw_records)));
	auto in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	    structured, buffer.index_in_bounds);
	in_bounds = Select(
	    state, TypeBool(state),
	    Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	    in_bounds,
	    Select(state, TypeBool(state),
	           Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	           Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	           raw));
	return in_bounds;
}

// record-keyed (D6): raw DWORD stores through a GPU-selected descriptor. A store to a
// page without a resident BDA mapping is dropped (the load path reports such faults).
// Byte/short store through a GPU-selected descriptor (record-keyed, D6): the
// containing dword is updated with atomic AND/OR so neighbouring lanes' bytes are preserved.
void StoreIndirectSubword(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t bits) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
		                                  ConstantU32(state, 0));
		EmitIfCondition(state, AndCondition(state, valid_format,
		                                    IndirectBufferInBounds(state, buffer, 0u, bits / 8u, false)),
		                [&]() {
			const auto mask = ConstantU32(state, bits == 8u ? 0xffu : 0xffffu);
			const auto shift = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
			    Binary(state, spv::OpBitwiseAnd, TypeU32(state),
			           Unary(state, spv::OpUConvert, TypeU32(state), buffer.address), ConstantU32(state, 3)),
			    ConstantU32(state, 3));
			const auto data = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
			                         ctx.Arg(inst, inst.NumArgs() - 2), mask);
			const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
			                            ConstantU64(state, ~uint64_t {3}));
			const auto bda = GetBdaPointer(state, aligned);
			RecordBdaTarget(state, aligned); // KYTY_RECORD_KEYED_VERIFY
			ReportBdaStoreMiss(state, aligned, bda); // KYTY_DIAG_RT_TRIPS
			EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0)),
			                [&]() {
				const auto pointer = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
				const auto semantics = ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
				                                              spv::MemorySemanticsUniformMemoryMask);
				const auto keep = Unary(state, spv::OpNot, TypeU32(state),
				                        Binary(state, spv::OpShiftLeftLogical, TypeU32(state), mask, shift));
				state.builder.AddFunction(spv::OpAtomicAnd, TypeU32(state), state.builder.AllocateId(), pointer,
				                          ConstantU32(state, spv::ScopeDevice), semantics, keep);
				state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(), pointer,
				                          ConstantU32(state, spv::ScopeDevice), semantics,
				                          Binary(state, spv::OpShiftLeftLogical, TypeU32(state), data, shift));
				RecordBdaWrite(state, aligned);
			});
		});
	});
}

// One 8, 16 or 32-bit piece of a formatted store through BDA at the byte address `address`:
// 8 and 16-bit pieces update their dword with atomic AND/OR so neighbouring lanes' bytes are kept.
static void StoreBdaPiece(EmitterState& state, uint32_t address, uint32_t bits, uint32_t value) {
	const auto aligned = Binary(state, spv::OpBitwiseAnd, TypeU64(state), address, ConstantU64(state, ~uint64_t {3}));
	const auto bda     = GetBdaPointer(state, aligned);
	RecordBdaTarget(state, aligned); // KYTY_RECORD_KEYED_VERIFY
	ReportBdaStoreMiss(state, aligned, bda); // KYTY_DIAG_RT_TRIPS
	EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0)), [&]() {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, bda);
		if (bits == 32u) {
			state.builder.AddFunction(spv::OpStore, pointer, value, spv::MemoryAccessAlignedMask, 4u);
		} else {
			const auto mask  = ConstantU32(state, bits == 8u ? 0xffu : 0xffffu);
			const auto shift = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
			                          Binary(state, spv::OpBitwiseAnd, TypeU32(state),
			                                 Unary(state, spv::OpUConvert, TypeU32(state), address),
			                                 ConstantU32(state, bits == 8u ? 3u : 2u)),
			                          ConstantU32(state, 3));
			const auto semantics =
			    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
			const auto keep = Unary(state, spv::OpNot, TypeU32(state),
			                        Binary(state, spv::OpShiftLeftLogical, TypeU32(state), mask, shift));
			state.builder.AddFunction(spv::OpAtomicAnd, TypeU32(state), state.builder.AllocateId(), pointer,
			                          ConstantU32(state, spv::ScopeDevice), semantics, keep);
			state.builder.AddFunction(spv::OpAtomicOr, TypeU32(state), state.builder.AllocateId(), pointer,
			                          ConstantU32(state, spv::ScopeDevice), semantics,
			                          Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
			                                 Binary(state, spv::OpBitwiseAnd, TypeU32(state), value, mask), shift));
		}
		RecordBdaWrite(state, aligned);
	});
}

// The formats StoreIndirectFormatted writes: known, at most 16 bytes, packed ones only as one 16/32-bit piece.
constexpr bool IndirectFormattedStoreSupported(const Format::BufferFormatInfo& info) {
	if (info.type == Format::ComponentType::Unknown || info.component_count == 0u || info.byte_size == 0u ||
	    info.byte_size > 16u)
		return false;
	return !info.packed_bitfield || info.byte_size == 2u || info.byte_size == 4u;
}

// One format case of a formatted store through a GPU-selected descriptor: while the element is in bounds, the
// `components` channels of `data` (missing ones zero) encoded in `info` and written as BDA pieces (StoreBdaPiece).
void StoreIndirectFormattedCase(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                                const Format::BufferFormatInfo& info, uint32_t in_bounds, uint32_t data,
                                uint32_t components, uint32_t data_bits) {
	auto&      state = ctx.state;
	const auto value = [&](uint32_t component) {
		if (component >= components) return ConstantU32(state, 0);
		if (components == 1u) return data;
		const auto extracted = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), extracted, data, component);
		return extracted;
	};
	EmitIfCondition(state, in_bounds, [&]() {
		if (info.packed_bitfield) {
			auto packed = ConstantU32(state, 0);
			for (uint32_t component = 0; component < info.component_count; ++component) {
				packed = EmitBitFieldInsert(
				    state, packed, EncodeFormattedStoreComponent(ctx.state, info, component, value(component), data_bits),
				    ConstantU32(state, info.component_bit_offset[component]),
				    ConstantU32(state, info.component_bits[component]));
			}
			StoreBdaPiece(state, buffer.address, info.byte_size * 8u, packed);
			return;
		}
		for (uint32_t component = 0; component < info.component_count; ++component) {
			const auto bits = info.component_bits[component];
			if (bits != 8u && bits != 16u && bits != 32u) continue;
			const auto offset  = Format::GetFormatComponentByteOffset(info, component);
			const auto address = offset == 0u ? buffer.address
			                                  : Binary(state, spv::OpIAdd, TypeU64(state), buffer.address,
			                                           ConstantU64(state, offset));
			StoreBdaPiece(state, address, bits, EncodeFormattedStoreComponent(ctx.state, info, component, value(component), data_bits));
		}
	});
}

// The call-site part of a store through a GPU-selected descriptor outlined into a function
// (KYTY_PERF_SPIRV_STORE_OUTLINE), as for the outlined loads: the ADD_TID lane (it depends on the wave half) and the
// instruction's immediate offset are folded into `index` and `offset` here; the function decodes the rest of the V#.
struct IndirectStoreCall {
	uint32_t word0;
	uint32_t word1;
	uint32_t records;
	uint32_t word3;
	uint32_t index;
	uint32_t offset;
	uint32_t soffset;
};

IndirectStoreCall PrepareIndirectStoreCall(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&       state       = ctx.state;
	const auto& handle      = *inst.Arg(0).ResolveInstruction();
	const auto  word3       = ctx.Arg(handle, 3);
	const auto  lane_enable = Binary(state, spv::OpINotEqual, TypeBool(state),
	                                 EmitBitFieldUExtract(state, word3, ConstantU32(state, 23), ConstantU32(state, 1)),
	                                 ConstantU32(state, 0));
	const auto  index       = Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	                                 Select(state, TypeU32(state), lane_enable, BufferLane(state), ConstantU32(state, 0)));
	const auto  immediate   = ctx.Memory(inst).offset;
	const auto  offset      = immediate == 0u ? ctx.Arg(inst, 2)
	                                          : Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 2),
	                                                   ConstantU32(state, immediate));
	return {.word0   = ctx.Arg(handle, 0),
	        .word1   = ctx.Arg(handle, 1),
	        .records = ctx.Arg(handle, 2),
	        .word3   = word3,
	        .index   = index,
	        .offset  = offset,
	        .soffset = ctx.Arg(inst, 3)};
}

// The store opcodes whose data is the u32 composite of their component count (the parameter type of the outlined
// formatted store functions).
bool IsDwordBufferStore(IR::ValueOpcode op) {
	return op == IR::ValueOpcode::StoreBufferU32 || op == IR::ValueOpcode::StoreBufferU32x2 ||
	       op == IR::ValueOpcode::StoreBufferU32x3 || op == IR::ValueOpcode::StoreBufferU32x4;
}

// Formatted X/XY/XYZ/XYZW store through a GPU-selected descriptor (hardware behaviour, M48):
// the runtime format of descriptor word 3 selects the encoding; the whole format is written
// (missing VGPRs zero-filled, as RDNA2 does). Packed formats are packed into one 16/32-bit piece;
// other formats write each 8/16/32-bit channel at its byte offset. Wider channels stay dropped.
// KYTY_PERF_SPIRV_STORE_OUTLINE: with the function of its component count defined (DefineIndirectStoreFunctions)
// the store is one call of it; the switch below is then emitted once, in that function.
// Raw GLC/DLC accesses through a GPU-built descriptor are coherent (BdaDwordPointer); formatted ones are not yet (their
// outlined functions decode the descriptor without the access). Logged, so a shader that polls through one shows up.
void LogCoherentFormattedIndirect(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (!ctx.Memory(inst).coherent) return;
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 8u)
		std::fprintf(stderr, "COHERENT_INDIRECT: formatted GLC access through a GPU-built descriptor stays cached, shader 0x%016llx\n",
		             static_cast<unsigned long long>(ctx.state.program.shader_hash));
}

void StoreIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	LogCoherentFormattedIndirect(ctx, inst);
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto outlined = state.indirect_formatted_store_functions.find(components);
		if (outlined != state.indirect_formatted_store_functions.end() && outlined->second != 0u &&
		    IsDwordBufferStore(inst.GetOpcode()) && ctx.Memory(inst).data_bits == 32u) {
			const auto call = PrepareIndirectStoreCall(ctx, inst);
			state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
			                          outlined->second, call.word0, call.word1, call.records, call.word3, call.index,
			                          call.offset, call.soffset, ctx.Arg(inst, inst.NumArgs() - 2));
			return;
		}
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		const auto data   = ctx.Arg(inst, inst.NumArgs() - 2);
		(void)EmitIndexSwitch(
		    state, buffer.format, static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u,
		    TypeU32(state), [&](uint32_t format) {
			    const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
			    if (!IndirectFormattedStoreSupported(info)) return ConstantU32(state, 0);
			    const auto in_bounds = IndirectBufferInBounds(state, buffer, 0u, info.byte_size, true);
			    StoreIndirectFormattedCase(ctx, buffer, info, in_bounds, data, components, ctx.Memory(inst).data_bits);
			    return ConstantU32(state, 0);
		    });
	});
}

void StoreIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto outlined = ctx.Memory(inst).coherent ? state.indirect_raw_store_coherent_function
		                                                : state.indirect_raw_store_function;
		if (outlined != 0u && IsDwordBufferStore(inst.GetOpcode())) {
			// KYTY_PERF_SPIRV_STORE_OUTLINE: decode, bounds, BDA translation and the store of each dword happen in
			// indirect_raw_store_dword, called per component as the loads call indirect_raw_dword.
			const auto call = PrepareIndirectStoreCall(ctx, inst);
			const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
			for (uint32_t component = 0; component < components; ++component) {
				uint32_t value = data;
				if (components > 1u) {
					value = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data, component);
				}
				state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
				                          outlined, call.word0, call.word1, call.records,
				                          call.word3, call.index, call.offset, call.soffset,
				                          ConstantU32(state, component * 4u), value);
			}
			return;
		}
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		const auto data = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
		                                  ConstantU32(state, 0));
		const auto base = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
		                         ConstantU64(state, ~uint64_t {3}));
		for (uint32_t component = 0; component < components; ++component) {
			uint32_t value = data;
			if (components > 1u) {
				value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), value, data, component);
			}
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU64(state), base,
			                                              ConstantU64(state, component * 4u));
			EmitIfCondition(state, AndCondition(state, valid_format,
			                                    IndirectBufferInBounds(state, buffer, component * 4u, 4u, false)),
			                [&]() {
				                const auto bda = GetBdaPointer(state, address);
				                RecordBdaTarget(state, address); // KYTY_RECORD_KEYED_VERIFY
				                ReportBdaStoreMiss(state, address, bda); // KYTY_DIAG_RT_TRIPS
				                EmitIfCondition(state, Binary(state, spv::OpINotEqual, TypeBool(state), bda,
				                                              ConstantU64(state, 0)), [&]() {
					                state.builder.AddFunction(spv::OpStore, BdaDwordPointer(state, bda, buffer.coherent),
					                                          value, BdaDwordAccess(buffer.coherent), 4u);
					                RecordBdaWrite(state, address);
				                });
			                });
		}
	});
}

// Raw byte/short load through a GPU-selected descriptor; the translator widens the result.
// 8/16-bit raw loads through a GPU-selected buffer
// descriptor (BUFFER_LOAD_UBYTE/USHORT/SBYTE/SSHORT with a V# loaded at run time) take LoadIndirectSubword again.
// 5f01308b routes every single-component raw indirect load to LoadIndirectBuffer, which returns the whole aligned
// dword: the neighbouring bytes come along (and the wrong half for an odd ushort). In Ghost of Yotei's class-9
// tiled lighting (85312e99a3c69cef) the light tree's 0xffff "no child" entries read as 0xffffffff, the walk never
// reached a leaf and every class-9 tile stayed black (KYTY_DIAG_PROBE vs refinterp).
uint32_t LoadIndirectSubword(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t bits) {
	auto& state = ctx.state;
	return EmitValueOrZeroIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto buffer = PrepareIndirectBuffer(ctx, inst);
		const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
		                                  ConstantU32(state, 0));
		return LoadBda(ctx, buffer.address,
		               AndCondition(state, valid_format,
		                            IndirectBufferInBounds(state, buffer, 0u, bits / 8u, false)),
		               bits, buffer.coherent);
	});
}

// Compute stages (RobustBuffer): word 3 of the V# for a call of an outlined indirect load. An inactive lane passes 0, a null descriptor
// (no format, every DST_SEL 0): the function gives 0 with no memory access and no BDA fault, what the exec branch around
// the call gave. (Not Select on the result: a stale V# in an inactive lane would report false faults.)
uint32_t IndirectCallWord3(EmitterState& state, uint32_t word3) {
	if (state.indirect_guard == 0u) return word3;
	return Select(state, TypeU32(state), state.indirect_guard, word3, ConstantU32(state, 0));
}

struct IndirectGuardScope {
	IndirectGuardScope(EmitterState& s, uint32_t guard): state(s) { state.indirect_guard = guard; }
	~IndirectGuardScope() { state.indirect_guard = 0u; }
	EmitterState& state;
};

uint32_t LoadIndirectBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto&      state    = ctx.state;
	const auto outlined = ctx.Memory(inst).coherent ? state.indirect_raw_coherent_function
	                                                : state.indirect_raw_function;
	if (outlined != 0u) {
		// Descriptor decode, bounds and BDA load all happen in the outlined function.
		const auto& handle = *inst.Arg(0).ResolveInstruction();
		const auto  word3  = ctx.Arg(handle, 3);
		const auto  lane_enable = Binary(state, spv::OpINotEqual, TypeBool(state),
		                                  EmitBitFieldUExtract(state, word3, ConstantU32(state, 23), ConstantU32(state, 1)),
		                                  ConstantU32(state, 0));
		const auto index = Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
		                          Select(state, TypeU32(state), lane_enable, BufferLane(state),
		                                 ConstantU32(state, 0)));
		const auto immediate = ctx.Memory(inst).offset;
		const auto offset = immediate == 0u ? ctx.Arg(inst, 2)
		                                    : Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 2),
		                                             ConstantU32(state, immediate));
		// F1: the whole X2..X4 load in one call (DefineIndirectRawDword).
		const auto group = components >= 2u && components <= 4u
		                       ? state.indirect_raw_group_functions[(components - 2u) + (ctx.Memory(inst).coherent ? 3u : 0u)]
		                       : 0u;
		if (group != 0u) {
			const auto result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionCall, TypeU32Composite(state, components), result, group,
			                          ctx.Arg(handle, 0), ctx.Arg(handle, 1), ctx.Arg(handle, 2),
			                          IndirectCallWord3(state, word3), index, offset, ctx.Arg(inst, 3));
			return result;
		}
		std::array<uint32_t, 4> values {};
		const auto call_word3 = IndirectCallWord3(state, word3);
		for (uint32_t component = 0; component < components; ++component) {
			values[component] = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionCall, TypeU32(state), values[component],
			                          outlined, ctx.Arg(handle, 0), ctx.Arg(handle, 1),
			                          ctx.Arg(handle, 2), call_word3, index, offset, ctx.Arg(inst, 3),
			                          ConstantU32(state, component * 4u));
		}
		return ConstructU32Composite(state, components, values);
	}
	const auto buffer = PrepareIndirectBuffer(ctx, inst);
	const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
	                                  ConstantU32(state, 0));
	const auto base         = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                                 ConstantU64(state, ~uint64_t {3}));
	std::array<uint32_t, 4> values {};
	for (uint32_t component = 0; component < components; ++component) {
		const auto address = component == 0u ? base
		                                     : Binary(state, spv::OpIAdd, TypeU64(state), base,
		                                              ConstantU64(state, component * 4u));
		values[component] = EmitValueOrZeroIfCondition(
		    state, AndCondition(state, valid_format,
		                        IndirectBufferInBounds(state, buffer, component * 4u, 4u, false)),
		    [&]() { return LoadBdaDword(ctx, address, buffer.coherent); });
	}
	return ConstructU32Composite(state, components, values);
}

// A formatted load through a GPU-selected descriptor is one call of a function made for its shape (channels, result
// dwords, D16). The function decodes the V# once, checks the element's bounds once, reads each dword of the element at
// most once through BDA (only the dwords some channel's DST_SEL selects) and converts each source component once; every
// channel then takes its component or its constant. 3ab9b653825476ab: 928 channel calls -> 256 calls.

// The bits LoadFormattedComponent reads for a source component: whole dwords for packed formats and 32-bit
// components, the component's own 8 or 16 bits otherwise.
constexpr uint32_t FormattedReadBits(const Format::BufferFormatInfo& info, uint32_t component) {
	return info.packed_bitfield || info.component_bits[component] == 32u ? 32u : info.component_bits[component];
}

// The element address is aligned down to min(4, byte size).
constexpr uint32_t FormattedBaseAlignment(const Format::BufferFormatInfo& info) {
	return std::min(4u, info.byte_size);
}

// The dword of the element (from its aligned base) that holds a source component's read.
constexpr uint32_t FormattedReadDword(const Format::BufferFormatInfo& info, uint32_t component) {
	return FormattedBaseAlignment(info) == 4u ? Format::GetFormatComponentByteOffset(info, component) / 4u : 0u;
}

// True when every component read of the format stays inside one dword for every base its alignment allows: then
// LoadBda never reads a second dword ("crosses" is always false) and the components of one dword share its read.
constexpr bool FormattedReadsInDwords(const Format::BufferFormatInfo& info) {
	const auto alignment = FormattedBaseAlignment(info);
	if (alignment != 1u && alignment != 2u && alignment != 4u) return false;
	if (alignment == 4u && (info.byte_size % 4u != 0u || info.byte_size > 16u)) return false;
	for (uint32_t component = 0; component < info.component_count; ++component) {
		const auto offset = Format::GetFormatComponentByteOffset(info, component);
		const auto bytes  = FormattedReadBits(info, component) / 8u;
		if (alignment == 4u ? offset % 4u + bytes > 4u : offset + bytes > alignment) return false;
	}
	return true;
}

constexpr bool IndirectFormatsReadInDwords() {
	for (uint32_t format = 0; format <= static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float); ++format) {
		const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
		if (info.type != Format::ComponentType::Unknown && !FormattedReadsInDwords(info)) return false;
	}
	return true;
}

static_assert(IndirectFormatsReadInDwords(),
              "The formatted load reads each element dword once: a format reads across a dword");

// PrepareIndirectBuffer inside a function: `index` already holds the ADD_TID lane and `offset` the instruction's
// immediate offset (as in DefineIndirectRawDword).
IndirectBufferAccess DecodeIndirectBuffer(EmitterState& state, uint32_t word0, uint32_t word1, uint32_t records,
                                          uint32_t word3, uint32_t index, uint32_t offset, uint32_t soffset) {
	const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto nonzero = [&](uint32_t value) {
		return Binary(state, spv::OpINotEqual, TypeBool(state), value, ConstantU32(state, 0));
	};
	const auto stride      = field(word1, 16, 14);
	const auto swizzle     = AndCondition(state, nonzero(stride), nonzero(field(word1, 31, 1)));
	const auto address     = CalculateBufferAddress(state, index, offset, soffset, 0u, stride, swizzle,
	                                                field(word3, 21, 2));
	const auto base        = PackU64(state, word0, field(word1, 0, 16));
	const auto raw_records = Binary(state, spv::OpISub, TypeU32(state), records, soffset);
	return {
	    .address             = Binary(state, spv::OpIAdd, TypeU64(state), base,
	                                  Unary(state, spv::OpUConvert, TypeU64(state), address.byte)),
	    .offset              = address.offset,
	    .stride              = stride,
	    .swizzle             = swizzle,
	    .format              = field(word3, 12, 7),
	    .selector            = 0u,
	    .mode                = field(word3, 28, 2),
	    .records             = records,
	    .index_in_bounds     = Binary(state, spv::OpULessThan, TypeBool(state), index, records),
	    .scalar_in_bounds    = Binary(state, spv::OpULessThanEqual, TypeBool(state), soffset, records),
	    .raw_records         = raw_records,
	    .raw_index_in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index, raw_records),
	};
}

// Grouped IndirectBuffer reads (F1): if (condition) fast() else slow(), each giving `count` u32 values; the merged values.
template <typename Fast, typename Slow>
static std::array<uint32_t, 4> EmitIfElseU32s(EmitterState& state, uint32_t condition, uint32_t count, Fast&& fast,
                                              Slow&& slow) {
	const auto fast_label  = state.builder.AllocateId();
	const auto slow_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, fast_label, slow_label);
	EmitLabel(state, fast_label);
	state.guard_depth++;
	const std::array<uint32_t, 4> fast_values = fast();
	const auto                    fast_exit   = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, slow_label);
	const std::array<uint32_t, 4> slow_values = slow();
	state.guard_depth--;
	const auto slow_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	std::array<uint32_t, 4> values {};
	for (uint32_t i = 0; i < count; ++i) {
		values[i] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, TypeU32(state), values[i], fast_values[i], fast_exit, slow_values[i],
		                          slow_exit);
	}
	return values;
}

// Grouped IndirectBuffer reads (F1): `count` dwords at the BDA pointer `bda` (a u64, nonzero) and the following dwords of
// its page, loaded back to back exactly as LoadBdaDword loads one (same pointer type, access and alignment).
static std::array<uint32_t, 4> LoadBdaDwordsAt(EmitterState& state, uint32_t bda, uint32_t count, bool coherent) {
	std::array<uint32_t, 4> values {};
	for (uint32_t i = 0; i < count; ++i) {
		const auto address = i == 0u ? bda : Binary(state, spv::OpIAdd, TypeU64(state), bda, ConstantU64(state, i * 4u));
		const auto pointer = BdaDwordPointer(state, address, coherent);
		values[i]          = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), values[i], pointer, BdaDwordAccess(coherent), 4u);
	}
	return values;
}

// Grouped IndirectBuffer reads (F1): the dwords [first, first + bytes) lie in one 16 KiB page (`first`: a u64 address).
static uint32_t InOnePage(EmitterState& state, uint32_t first, uint32_t bytes) {
	return Binary(state, spv::OpULessThanEqual, TypeBool(state),
	              Binary(state, spv::OpBitwiseAnd, TypeU64(state), first, ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1)),
	              ConstantU64(state, BufferCache::CACHING_PAGESIZE - bytes));
}

// One known format of LoadIndirectFormattedVector. Per channel: a
// DST_SEL below 4 gives 0 or the format's one, else source component (sel - 4) % count, read through LoadBda
// while the whole element is in bounds (0 otherwise) and converted; D16 narrows every channel.
uint32_t LoadIndirectFormattedElement(ValueEmitContext& ctx, const IndirectBufferAccess& buffer,
                                      const Format::BufferFormatInfo& info, uint32_t channels,
                                      const std::array<uint32_t, 4>& selector, const std::array<uint32_t, 4>& memory,
                                      const std::array<uint32_t, 4>& one, uint32_t any_memory, bool d16) {
	auto&      state     = ctx.state;
	const auto count     = info.component_count;
	const auto alignment = FormattedBaseAlignment(info);
	const auto dwords    = alignment == 4u ? info.byte_size / 4u : 1u;
	const auto in_bounds = IndirectBufferInBounds(state, buffer, 0u, info.byte_size, true);
	const auto base      = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
	                              ConstantU64(state, ~(uint64_t(alignment) - 1u)));
	std::array<uint32_t, 4> component {};
	for (uint32_t channel = 0; channel < channels; ++channel) {
		component[channel] = count == 1u ? ConstantU32(state, 0)
		                                 : Binary(state, spv::OpUMod, TypeU32(state),
		                                          Binary(state, spv::OpISub, TypeU32(state), selector[channel],
		                                                 ConstantU32(state, 4)),
		                                          ConstantU32(state, count));
	}
	// A dword is read only when some channel selects one of its components: a lookup of a missing page records a
	// fault, so only those pages are looked up.
	uint32_t selected_components = 0;
	if (dwords > 1u) {
		for (uint32_t channel = 0; channel < channels; ++channel) {
			const auto bit = Select(state, TypeU32(state), memory[channel],
			                        Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
			                               component[channel]),
			                        ConstantU32(state, 0));
			selected_components =
			    channel == 0u ? bit : Binary(state, spv::OpBitwiseOr, TypeU32(state), selected_components, bit);
		}
	}
	std::array<uint32_t, 4> selected {};
	std::array<uint32_t, 4> addresses {};
	const auto prepare = [&](uint32_t dword) {
		uint32_t dword_components = 0;
		for (uint32_t source = 0; source < count; ++source) {
			if (FormattedReadDword(info, source) == dword) dword_components |= 1u << source;
		}
		selected[dword] =
		    dwords == 1u ? any_memory
		                 : Binary(state, spv::OpINotEqual, TypeBool(state),
		                          Binary(state, spv::OpBitwiseAnd, TypeU32(state), selected_components,
		                                 ConstantU32(state, dword_components)),
		                          ConstantU32(state, 0));
		// Below 4-byte alignment the element lies in the dword holding its base.
		addresses[dword] = alignment < 4u ? Binary(state, spv::OpBitwiseAnd, TypeU64(state), base,
		                                           ConstantU64(state, ~uint64_t {3}))
		                   : dword == 0u  ? base
		                                  : Binary(state, spv::OpIAdd, TypeU64(state), base,
		                                           ConstantU64(state, dword * 4u));
	};
	// Today's read of one dword: a selected dword of an in-bounds element through its own page lookup.
	const auto read_dword = [&](uint32_t dword) {
		return EmitValueOrZeroIfCondition(state, AndCondition(state, in_bounds, selected[dword]),
		                                  [&]() { return LoadBdaDword(ctx, addresses[dword]); });
	};
	std::array<uint32_t, 4> words {};
	for (uint32_t dword = 0; dword < dwords; ++dword) {
		prepare(dword);
		words[dword] = read_dword(dword);
	}
	std::array<uint32_t, 4> converted {};
	for (uint32_t source = 0; source < count; ++source) {
		const auto bits   = FormattedReadBits(info, source);
		const auto offset = Format::GetFormatComponentByteOffset(info, source);
		auto       raw    = words[FormattedReadDword(info, source)];
		if (bits != 32u) {
			// LoadBda's merge without a second dword: the component's byte offset in its dword, then its bits.
			uint32_t shift = 0;
			if (alignment < 4u) {
				const auto address = offset == 0u ? base
				                                  : Binary(state, spv::OpIAdd, TypeU64(state), base,
				                                           ConstantU64(state, offset));
				shift = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
				               Binary(state, spv::OpBitwiseAnd, TypeU32(state),
				                      Unary(state, spv::OpUConvert, TypeU32(state), address), ConstantU32(state, 3)),
				               ConstantU32(state, 3));
			} else if (offset % 4u != 0u) {
				shift = ConstantU32(state, offset % 4u * 8u);
			}
			if (shift != 0u) raw = Binary(state, spv::OpShiftRightLogical, TypeU32(state), raw, shift);
			raw = Binary(state, spv::OpBitwiseAnd, TypeU32(state), raw,
			             ConstantU32(state, bits == 8u ? 0xffu : 0xffffu));
		}
		converted[source] = LoadFormattedComponent(
		    ctx, info, {FormattedSourceKind::Memory, source}, [&](uint32_t) { return raw; },
		    [&](uint32_t, uint32_t) { return raw; });
	}
	const auto one_value = FormattedConstant(ctx, info, FormattedSourceKind::One);
	std::array<uint32_t, 4> values {};
	for (uint32_t channel = 0; channel < channels; ++channel) {
		auto from_memory = converted[0];
		for (uint32_t source = 1; source < count; ++source) {
			from_memory = Select(state, TypeU32(state),
			                     Binary(state, spv::OpIEqual, TypeBool(state), component[channel],
			                            ConstantU32(state, source)),
			                     converted[source], from_memory);
		}
		const auto constant = Select(state, TypeU32(state), one[channel], one_value, ConstantU32(state, 0));
		auto       value    = Select(state, TypeU32(state), memory[channel], from_memory, constant);
		if (d16) {
			std::array<uint32_t, 4> single {value};
			value = Binary(state, spv::OpBitwiseAnd, TypeU32(state), PackD16Channels(state, info, single, 1u, 1u),
			               ConstantU32(state, 0xffffu));
		}
		values[channel] = value;
	}
	return ConstructU32Composite(state, channels, values);
}

// LoadIndirectFormattedVector's 78 format cases as one read driven by a table of the
// formats' layouts. Each case of the switch is LoadIndirectFormattedElement with constants; the table holds those
// constants per format and the code below performs the same operations on them:
//   word 0: bit 0 known format, 1-5 element bytes, 6-7 alignment - 1, 8-10 source components (1 for an unknown
//           format: the modulo below needs a divisor), 11 "one" is the integer 1, 12-13 D16 class (0 Uint, 1 Sint,
//           2 float), 16 + 4d: the source components read from dword d (0: the dword is never read);
//   word 1 + s (source component s): bits 0-1 its dword, 2-3 its byte offset within the dword, 4-8 its packed bit
//           offset, 9-14 its bits (0 past the format's components), 15 sign-extended, 16-20 conversion (FormattedConv).
// Same values: a component is BitField[US]Extract(dword, 8 * ((base + byte offset) & 3) + packed offset, bits) - what
// the case's shift and mask (or its extract) compute -, converted by the case's own NormalizeFormatComponent (a switch
// over the conversions keeps every constant divisor a constant); the same dwords are read under the same conditions,
// so the page lookups and fault records are the case's. An unknown format, and a FORMAT value past the last one (the
// switch's default is case 0), gives 0 in every channel.
enum class FormattedConv : uint32_t {
	Identity, // Uint, Sint, 32-bit Float, unused components
	Uscaled,
	Sscaled,
	Unorm2,
	Unorm8,
	Unorm10,
	Unorm11,
	Unorm16,
	Snorm2,
	Snorm8,
	Snorm10,
	Snorm11,
	Snorm16,
	Float16,
	UFloat11,
	UFloat10,
	Count,
};

constexpr uint32_t FormattedTableFormats = static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u;
constexpr uint32_t FormattedTableWords   = 5u;

constexpr FormattedConv FormattedConvOf(const Format::BufferFormatInfo& info, uint32_t component) {
	using Format::ComponentType;
	const auto bits   = info.component_bits[component];
	const auto bucket = [&](FormattedConv first) {
		const uint32_t step = bits == 2u ? 0u : bits == 8u ? 1u : bits == 10u ? 2u : bits == 11u ? 3u : 4u;
		return static_cast<FormattedConv>(static_cast<uint32_t>(first) + step);
	};
	switch (info.type) {
		case ComponentType::Uscaled: return FormattedConv::Uscaled;
		case ComponentType::Sscaled: return FormattedConv::Sscaled;
		case ComponentType::Unorm: return bucket(FormattedConv::Unorm2);
		case ComponentType::Snorm: return bucket(FormattedConv::Snorm2);
		case ComponentType::Float:
			return bits == 16u ? FormattedConv::Float16
			       : bits == 11u ? FormattedConv::UFloat11
			       : bits == 10u ? FormattedConv::UFloat10
			                     : FormattedConv::Identity;
		default: return FormattedConv::Identity;
	}
}

// The one-component format info whose NormalizeFormatComponent is the conversion.
constexpr Format::BufferFormatInfo FormattedConvInfo(FormattedConv conv) {
	using Format::ComponentType;
	constexpr uint32_t bits[] = {2u, 8u, 10u, 11u, 16u};
	const auto         code   = static_cast<uint32_t>(conv);
	Format::BufferFormatInfo info {};
	info.component_count = 1u;
	info.byte_size       = 4u;
	switch (conv) {
		case FormattedConv::Uscaled: info.type = ComponentType::Uscaled; info.component_bits[0] = 32u; break;
		case FormattedConv::Sscaled: info.type = ComponentType::Sscaled; info.component_bits[0] = 32u; break;
		case FormattedConv::Float16: info.type = ComponentType::Float; info.component_bits[0] = 16u; break;
		case FormattedConv::UFloat11: info.type = ComponentType::Float; info.component_bits[0] = 11u; break;
		case FormattedConv::UFloat10: info.type = ComponentType::Float; info.component_bits[0] = 10u; break;
		default:
			if (code >= static_cast<uint32_t>(FormattedConv::Unorm2) && code <= static_cast<uint32_t>(FormattedConv::Unorm16)) {
				info.type              = ComponentType::Unorm;
				info.component_bits[0] = bits[code - static_cast<uint32_t>(FormattedConv::Unorm2)];
			} else if (code >= static_cast<uint32_t>(FormattedConv::Snorm2) &&
			           code <= static_cast<uint32_t>(FormattedConv::Snorm16)) {
				info.type              = ComponentType::Snorm;
				info.component_bits[0] = bits[code - static_cast<uint32_t>(FormattedConv::Snorm2)];
			} else {
				info.type              = ComponentType::Uint;
				info.component_bits[0] = 32u;
			}
	}
	return info;
}

constexpr std::array<uint32_t, FormattedTableFormats * FormattedTableWords> BuildFormattedTable() {
	using Format::ComponentType;
	std::array<uint32_t, FormattedTableFormats * FormattedTableWords> table {};
	for (uint32_t format = 0; format < FormattedTableFormats; ++format) {
		auto*      entry = &table[format * FormattedTableWords];
		const auto info  = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
		if (info.type == ComponentType::Unknown) {
			entry[0] = 1u << 8u;
			continue;
		}
		const auto alignment = FormattedBaseAlignment(info);
		const auto dwords    = alignment == 4u ? info.byte_size / 4u : 1u;
		const bool integer   = info.type == ComponentType::Uint || info.type == ComponentType::Sint;
		const auto d16_class = info.type == ComponentType::Uint ? 0u : info.type == ComponentType::Sint ? 1u : 2u;
		const bool is_signed = info.type == ComponentType::Sint || info.type == ComponentType::Snorm ||
		                       info.type == ComponentType::Sscaled;
		uint32_t head = 1u | info.byte_size << 1u | (alignment - 1u) << 6u | info.component_count << 8u |
		                (integer ? 1u : 0u) << 11u | d16_class << 12u;
		for (uint32_t component = 0; component < info.component_count; ++component) {
			const auto dword = FormattedReadDword(info, component);
			if (dword < dwords) head |= (1u << component) << (16u + 4u * dword);
			const auto bits = info.component_bits[component];
			const bool sign = is_signed && (info.packed_bitfield || bits < 32u);
			entry[1u + component] =
			    dword | (Format::GetFormatComponentByteOffset(info, component) & 3u) << 2u |
			    (info.packed_bitfield ? info.component_bit_offset[component] : 0u) << 4u | bits << 9u |
			    (sign ? 1u : 0u) << 15u | static_cast<uint32_t>(FormattedConvOf(info, component)) << 16u;
		}
		entry[0] = head;
	}
	return table;
}

// IndirectBufferInBounds with the element's byte count as a value (the table's).
uint32_t IndirectBufferInBoundsBytes(EmitterState& state, const IndirectBufferAccess& buffer, uint32_t bytes) {
	const auto offset      = buffer.offset;
	const auto has_payload = [&](uint32_t size) {
		return AndCondition(
		    state, Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), size, bytes),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state), offset,
		           Binary(state, spv::OpISub, TypeU32(state), size, bytes)));
	};
	const auto structured = AndCondition(state, buffer.index_in_bounds, has_payload(buffer.stride));
	const auto raw        = AndCondition(
	    state, buffer.scalar_in_bounds,
	    Select(state, TypeBool(state), buffer.swizzle,
	           AndCondition(state, buffer.raw_index_in_bounds, has_payload(buffer.stride)),
	           has_payload(buffer.raw_records)));
	auto in_bounds = Select(state, TypeBool(state),
	                        Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 0)),
	                        structured, buffer.index_in_bounds);
	in_bounds = Select(state, TypeBool(state),
	                   Binary(state, spv::OpULessThan, TypeBool(state), buffer.mode, ConstantU32(state, 2)), in_bounds,
	                   Select(state, TypeBool(state),
	                          Binary(state, spv::OpIEqual, TypeBool(state), buffer.mode, ConstantU32(state, 2)),
	                          Binary(state, spv::OpINotEqual, TypeBool(state), buffer.records, ConstantU32(state, 0)),
	                          raw));
	return in_bounds;
}

// The `channels` channel values of LoadIndirectFormattedVector's format switch, from the table.
uint32_t LoadIndirectFormattedTable(ValueEmitContext& ctx, const IndirectBufferAccess& buffer, uint32_t channels,
                                    const std::array<uint32_t, 4>& selector, const std::array<uint32_t, 4>& memory,
                                    const std::array<uint32_t, 4>& one, bool d16) {
	auto&      state   = ctx.state;
	const auto u32     = TypeU32(state);
	const auto u64     = TypeU64(state);
	const auto boolean = TypeBool(state);
	// The format's five table words: a constant u32[5] picked by a switch over the format (one phi per word, no code in
	// the cases; FORMAT values past the last format take case 0, kInvalid, as the format switch's default). A Private
	// table indexed by the format (25.10, first version) gave the same values but cost +1.2 ms of GPU time per frame:
	// the driver set up the 390-word array in every invocation (CS 8457901d 0.44 -> 1.28 ms per dispatch).
	static constexpr auto table_words = BuildFormattedTable();
	const auto            entry_type =
	    state.builder.Type(spv::OpTypeArray, TypeU32(state), ConstantU32(state, FormattedTableWords));
	const auto picked = EmitIndexSwitch(state, buffer.format, FormattedTableFormats, entry_type, [&](uint32_t format) {
		std::array<uint32_t, FormattedTableWords> ids {};
		for (uint32_t i = 0; i < FormattedTableWords; ++i) {
			ids[i] = ConstantU32(state, table_words[format * FormattedTableWords + i]);
		}
		return state.builder.Constant(spv::OpConstantComposite, entry_type, std::span<const uint32_t>(ids));
	});
	const auto field   = [&](uint32_t word, uint32_t first, uint32_t count) {
        return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
	};
	const auto flag = [&](uint32_t word, uint32_t bit) {
		return Binary(state, spv::OpINotEqual, boolean, field(word, bit, 1u), ConstantU32(state, 0));
	};
	const auto word = [&](uint32_t index) {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, u32, value, picked, index);
		return value;
	};
	const auto head      = word(0);
	const auto valid     = flag(head, 0);
	const auto count     = field(head, 8, 3);
	const auto in_bounds = IndirectBufferInBoundsBytes(state, buffer, field(head, 1, 5));
	const auto base      = Binary(state, spv::OpBitwiseAnd, u64, buffer.address,
	                              Unary(state, spv::OpNot, u64, Unary(state, spv::OpUConvert, u64, field(head, 6, 2))));
	std::array<uint32_t, 4> component {};
	uint32_t                selected_components = 0;
	for (uint32_t channel = 0; channel < channels; ++channel) {
		component[channel] = Binary(state, spv::OpUMod, u32,
		                            Binary(state, spv::OpISub, u32, selector[channel], ConstantU32(state, 4)), count);
		const auto bit = Select(state, u32, memory[channel],
		                        Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1), component[channel]),
		                        ConstantU32(state, 0));
		selected_components = channel == 0u ? bit : Binary(state, spv::OpBitwiseOr, u32, selected_components, bit);
	}
	// The case's dword reads: dword d of the element (below 4-byte alignment the dword holding its base) while it is in
	// bounds and some channel selects one of its components.
	const auto              dword_base = Binary(state, spv::OpBitwiseAnd, u64, base, ConstantU64(state, ~uint64_t {3}));
	std::array<uint32_t, 4> words {};
	for (uint32_t dword = 0; dword < 4u; ++dword) {
		const auto selected = Binary(state, spv::OpINotEqual, boolean,
		                             Binary(state, spv::OpBitwiseAnd, u32, selected_components, field(head, 16u + 4u * dword, 4u)),
		                             ConstantU32(state, 0));
		const auto address  = dword == 0u ? dword_base
		                                  : Binary(state, spv::OpIAdd, u64, dword_base, ConstantU64(state, dword * 4u));
		words[dword] = EmitValueOrZeroIfCondition(state, AndCondition(state, in_bounds, selected),
		                                          [&]() { return LoadBdaDword(ctx, address); });
	}
	const auto              base_low = Unary(state, spv::OpUConvert, u32, base);
	std::array<uint32_t, 4> converted {};
	for (uint32_t source = 0; source < 4u; ++source) {
		const auto desc   = word(1u + source);
		const auto dword  = field(desc, 0, 2);
		auto       value  = words[0];
		for (uint32_t d = 1; d < 4u; ++d) {
			value = Select(state, u32, Binary(state, spv::OpIEqual, boolean, dword, ConstantU32(state, d)), words[d], value);
		}
		const auto byte     = Binary(state, spv::OpBitwiseAnd, u32,
		                             Binary(state, spv::OpIAdd, u32, base_low, field(desc, 2, 2)), ConstantU32(state, 3));
		const auto position = Binary(state, spv::OpIAdd, u32,
		                             Binary(state, spv::OpShiftLeftLogical, u32, byte, ConstantU32(state, 3)),
		                             field(desc, 4, 5));
		const auto bits     = field(desc, 9, 6);
		const auto raw      = Select(state, u32, flag(desc, 15), EmitBitFieldSExtract(state, value, position, bits),
		                             EmitBitFieldUExtract(state, value, position, bits));
		const auto convert  = [&]() {
			return EmitIndexSwitch(
			    state, field(desc, 16, 5), static_cast<uint32_t>(FormattedConv::Count), u32, [&](uint32_t conv) {
				    const auto kind = static_cast<FormattedConv>(conv);
				    return kind == FormattedConv::Identity ? raw
				                                           : NormalizeFormatComponent(state, FormattedConvInfo(kind), 0u, raw);
			    });
		};
		// A source past the format's components is never selected (component < count), so it is not converted.
		converted[source] = source == 0u ? convert()
		                                 : EmitValueOrZeroIfCondition(
		                                       state, Binary(state, spv::OpULessThan, boolean, ConstantU32(state, source), count),
		                                       convert);
	}
	const auto one_value = Select(state, u32, flag(head, 11), ConstantU32(state, 1), ConstantU32(state, 0x3f800000u));
	const auto d16_class = d16 ? field(head, 12, 2) : 0u;
	std::array<uint32_t, 4> values {};
	for (uint32_t channel = 0; channel < channels; ++channel) {
		auto from_memory = converted[0];
		for (uint32_t source = 1; source < 4u; ++source) {
			from_memory = Select(state, u32,
			                     Binary(state, spv::OpIEqual, boolean, component[channel], ConstantU32(state, source)),
			                     converted[source], from_memory);
		}
		const auto constant = Select(state, u32, one[channel], one_value, ConstantU32(state, 0));
		auto       value    = Select(state, u32, memory[channel], from_memory, constant);
		if (d16) {
			// PackD16Channels of the case's format class, for one channel.
			const auto narrow = [&](Format::ComponentType type) {
				Format::BufferFormatInfo info {};
				info.type = type;
				std::array<uint32_t, 4> single {value};
				return Binary(state, spv::OpBitwiseAnd, u32, PackD16Channels(state, info, single, 1u, 1u),
				              ConstantU32(state, 0xffffu));
			};
			const auto as_uint  = narrow(Format::ComponentType::Uint);
			const auto as_sint  = narrow(Format::ComponentType::Sint);
			const auto as_float = narrow(Format::ComponentType::Float);
			value = Select(state, u32, Binary(state, spv::OpIEqual, boolean, d16_class, ConstantU32(state, 0)), as_uint,
			               Select(state, u32, Binary(state, spv::OpIEqual, boolean, d16_class, ConstantU32(state, 1)),
			                      as_sint, as_float));
		}
		values[channel] = Select(state, u32, valid, value, ConstantU32(state, 0));
	}
	return ConstructU32Composite(state, channels, values);
}

// The whole formatted load through a GPU-selected descriptor: `channels`
// DST_SEL channels of word 3 as `components` result dwords (D16: two 16-bit channels per dword, low first).
uint32_t LoadIndirectFormattedVector(ValueEmitContext& ctx, const IndirectBufferAccess& buffer, uint32_t word3,
                                     uint32_t channels, uint32_t components, bool d16) {
	auto&                   state      = ctx.state;
	std::array<uint32_t, 4> selector   {};
	std::array<uint32_t, 4> memory     {};
	std::array<uint32_t, 4> one        {};
	uint32_t                any_memory = 0;
	for (uint32_t channel = 0; channel < channels; ++channel) {
		selector[channel] = EmitBitFieldUExtract(state, word3, ConstantU32(state, channel * 3u),
		                                         ConstantU32(state, 3u));
		memory[channel]   = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), selector[channel],
		                           ConstantU32(state, 4));
		one[channel]      = Binary(state, spv::OpIEqual, TypeBool(state), selector[channel], ConstantU32(state, 1));
		any_memory        = channel == 0u ? memory[channel]
		                                  : Binary(state, spv::OpLogicalOr, TypeBool(state), any_memory, memory[channel]);
	}
	const auto format_switch = [&]() {
		return EmitIndexSwitch(
		    state, buffer.format, static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u,
		    TypeU32Composite(state, channels), [&](uint32_t format) {
			    const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
			    if (info.type == Format::ComponentType::Unknown) return ConstantU32CompositeZero(state, channels);
			    return LoadIndirectFormattedElement(ctx, buffer, info, channels, selector, memory, one, any_memory, d16);
		    });
	};
	// The census formats as their own cases - each the format switch's case, which the compiler specializes -, every other
	// format through the table (LoadIndirectFormattedTable). The table alone decoded runtime layout words generically:
	// 2.4x the GPU time per load of the case (--bench-indirect-formatted: 1016 vs 428 ns per k16UInt index + k16_16Float
	// vertex pair); at the inn the triangle searches 4d6df08d/15f485d4/c089d15d went 8.1 -> 3.6 ms and the card 47.4 ->
	// 42.4 ms per frame, fps +7.7% (two pairs, 12.8 -> 13.8); in-game check 1.8e10 loads, 0 differences (27.10, k60).
	// Census at the inn (waves of formatted loads through GPU-selected V#s): k16_16_16_16Float XYZW 82%, k8UNorm 15%,
	// k8SInt 1.6%, k16UInt 1.0%, k16_16Float XY 1.0%, k10_10_10_2SNorm and k16_16_16_16SNorm XYZ < 0.01%; nothing else.
	// The 78-case switch is the reference the compute test checks the cases and the table against
	// (SetSpirvFormattedSwitchReference).
	using F = Prospero::BufferFormat;
	static constexpr F fast[] = {F::k16_16_16_16Float, F::k8UNorm, F::k8SInt, F::k16UInt, F::k16_16Float,
	                             F::k10_10_10_2SNorm, F::k16_16_16_16SNorm};
	const auto cases = [&]() {
		const auto u32  = TypeU32(state);
		auto       pick = ConstantU32(state, 0);
		for (uint32_t k = 0; k < std::size(fast); ++k) {
			pick = Select(state, u32,
			              Binary(state, spv::OpIEqual, TypeBool(state), buffer.format,
			                     ConstantU32(state, static_cast<uint32_t>(fast[k]))),
			              ConstantU32(state, k + 1u), pick);
		}
		return EmitIndexSwitch(state, pick, static_cast<uint32_t>(std::size(fast)) + 1u, TypeU32Composite(state, channels),
		                       [&](uint32_t item) {
			                       if (item == 0u) {
				                       return LoadIndirectFormattedTable(ctx, buffer, channels, selector, memory, one, d16);
			                       }
			                       const auto info = Format::GetFormatInfo(fast[item - 1u]);
			                       return LoadIndirectFormattedElement(ctx, buffer, info, channels, selector, memory, one,
			                                                           any_memory, d16);
		                       });
	};
	const auto values = SpirvFormattedSwitchReference() ? format_switch() : cases();
	if (!d16) return values;
	std::array<uint32_t, 4> channel_values {values};
	for (uint32_t channel = 0; channel < channels && channels > 1u; ++channel) {
		channel_values[channel] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), channel_values[channel], values, channel);
	}
	std::array<uint32_t, 4> words {};
	for (uint32_t word = 0; word < components; ++word) {
		const auto high = word * 2u + 1u < channels ? channel_values[word * 2u + 1u] : ConstantU32(state, 0);
		words[word] = Binary(state, spv::OpBitwiseOr, TypeU32(state), channel_values[word * 2u],
		                     Binary(state, spv::OpShiftLeftLogical, TypeU32(state), high, ConstantU32(state, 16u)));
	}
	return ConstructU32Composite(state, components, words);
}

uint32_t IndirectFormattedLoadKey(uint32_t channels, uint32_t components, bool d16) {
	return channels | components << 4u | (d16 ? 1u : 0u) << 8u;
}

// KYTY_PERF_INDIRECT_LOAD_FAST (indirectLoadFast.h): LoadIndirectFormattedVector's case of a D16 XYZW load from a
// k16_16_16_16Float V# with DST_SEL X,Y,Z,W, written out. There LoadIndirectFormattedElement reads both dwords of the
// 8-byte element while it is in bounds (0 otherwise), converts each half to f32 and PackD16Channels converts it back;
// the round trip returns every half unchanged except a signaling NaN, which gets its quiet bit (exhaustively checked
// on all 2^16 halves, verify_d16_passthrough.py), so each result dword is its memory
// dword with Quiet applied to both halves. The second dword reuses the first one's page-table entry when it is in the
// same 16 KiB page (the entry of a page does not change during a dispatch; a missing page is reported by the first
// lookup, so the second one would only set the same fault bit again).
uint32_t LoadIndirectD16x4HalfFast(ValueEmitContext& ctx, const IndirectBufferAccess& buffer) {
	auto&      state     = ctx.state;
	const auto u32       = TypeU32(state);
	const auto u64       = TypeU64(state);
	const auto pair      = TypeU32Composite(state, 2);
	const auto in_bounds = IndirectBufferInBounds(state, buffer, 0u, 8u, true);
	const auto base      = Binary(state, spv::OpBitwiseAnd, u64, buffer.address, ConstantU64(state, ~uint64_t {3}));
	const auto load      = [&](uint32_t bda) {
        const auto present = Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0));
        return EmitValueOrZeroIfCondition(state, present, [&]() {
            const auto         pointer   = BdaDwordPointer(state, bda, false);
            const auto         value     = state.builder.AllocateId();
            constexpr uint32_t alignment = sizeof(uint32_t);
            state.builder.AddFunction(spv::OpLoad, u32, value, pointer, BdaDwordAccess(false), alignment);
            return value;
        });
	};
	// Quiet(x) = x | 0x0200 per half that is a NaN: (x & 0x7fff) + 0x3ff carries into bit 15 exactly for a NaN.
	const auto quiet = [&](uint32_t x) {
		const auto nan = Binary(state, spv::OpBitwiseAnd, u32,
		                        Binary(state, spv::OpIAdd, u32, Binary(state, spv::OpBitwiseAnd, u32, x, ConstantU32(state, 0x7fff7fffu)),
		                               ConstantU32(state, 0x03ff03ffu)),
		                        ConstantU32(state, 0x80008000u));
		return Binary(state, spv::OpBitwiseOr, u32, x, Binary(state, spv::OpShiftRightLogical, u32, nan, ConstantU32(state, 6u)));
	};
	return EmitValueOrDefaultIfCondition(state, in_bounds, pair, ConstantU32CompositeZero(state, 2), [&]() {
		const auto bda0    = GetBdaPointer(state, base);
		const auto low     = Binary(state, spv::OpBitwiseAnd, u64, base, ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1u));
		const auto crosses = Binary(state, spv::OpUGreaterThan, TypeBool(state), low,
		                            ConstantU64(state, BufferCache::CACHING_PAGESIZE - 8u));
		const auto shared  = Select(state, u64, Binary(state, spv::OpINotEqual, TypeBool(state), bda0, ConstantU64(state, 0)),
		                            Binary(state, spv::OpIAdd, u64, bda0, ConstantU64(state, 4u)), ConstantU64(state, 0));
		const auto bda1    = EmitValueOrDefaultIfCondition(state, crosses, u64, shared, [&]() {
            return GetBdaPointer(state, Binary(state, spv::OpIAdd, u64, base, ConstantU64(state, 4u)));
        });
		const auto word0 = load(bda0);
		const auto word1 = load(bda1);
		return ConstructU32Composite(state, 2u, {quiet(word0), quiet(word1), 0u, 0u});
	});
}

// The call of a formatted load's shape function (DefineIndirectFormattedLoads), or
// the same code inline for a shape that has none.
uint32_t CallIndirectFormattedLoad(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t channels,
                                   uint32_t components, bool d16) {
	auto&       state  = ctx.state;
	const auto& handle = *inst.Arg(0).ResolveInstruction();
	const auto  word3  = ctx.Arg(handle, 3);
	const auto  found  = state.indirect_formatted_load_functions.find(IndirectFormattedLoadKey(channels, components, d16));
	if (found == state.indirect_formatted_load_functions.end() || found->second == 0u) {
		return LoadIndirectFormattedVector(ctx, PrepareIndirectBuffer(ctx, inst), word3, channels, components, d16);
	}
	// The ADD_TID lane (it depends on the wave half) and the immediate offset stay at the call.
	const auto lane_enable = Binary(state, spv::OpINotEqual, TypeBool(state),
	                                EmitBitFieldUExtract(state, word3, ConstantU32(state, 23), ConstantU32(state, 1)),
	                                ConstantU32(state, 0));
	const auto index       = Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 1),
	                                Select(state, TypeU32(state), lane_enable, BufferLane(state),
	                                       ConstantU32(state, 0)));
	const auto immediate   = ctx.Memory(inst).offset;
	const auto offset      = immediate == 0u ? ctx.Arg(inst, 2)
	                                         : Binary(state, spv::OpIAdd, TypeU32(state), ctx.Arg(inst, 2),
	                                                  ConstantU32(state, immediate));
	const auto call = [&]() {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionCall, TypeU32Composite(state, components), result, found->second,
		                          ctx.Arg(handle, 0), ctx.Arg(handle, 1), ctx.Arg(handle, 2),
		                          IndirectCallWord3(state, word3), index, offset, ctx.Arg(inst, 3));
		return result;
	};
	const int fast_mode = IndirectLoadFast::Mode();
	if (fast_mode == 0 || !d16 || channels != 4u || components != 2u ||
	    !IndirectLoadFast::Listed(state.program.shader_hash)) {
		return call();
	}
	// KYTY_PERF_INDIRECT_LOAD_FAST: k16_16_16_16Float (71) with DST_SEL X,Y,Z,W (4 | 5 << 3 | 6 << 6 | 7 << 9 = 0xfac)
	// inline, any other V# through the function.
	const auto u32   = TypeU32(state);
	const auto field = [&](uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word3, ConstantU32(state, first), ConstantU32(state, count));
	};
	auto guard = Binary(state, spv::OpLogicalAnd, TypeBool(state),
	                    Binary(state, spv::OpIEqual, TypeBool(state), field(12, 7),
	                           ConstantU32(state, static_cast<uint32_t>(Prospero::BufferFormat::k16_16_16_16Float))),
	                    Binary(state, spv::OpIEqual, TypeBool(state), field(0, 12), ConstantU32(state, 0xfacu)));
	if (fast_mode == 3) {
		// Live mode: also bit 0 of the live shader flags word (SpirvLive, written at frame boundaries).
		if (state.fault_buffer_variable == 0u) return call();
		const auto flags = Unary(state, spv::OpLoad, u32,
		                         FaultElementPointer(state, ConstantU32(state, BufferCache::FAULT_DIAG_BASE +
		                                                                           BufferCache::FAULT_DIAG_WORDS)));
		guard = Binary(state, spv::OpLogicalAnd, TypeBool(state), guard,
		               Binary(state, spv::OpINotEqual, TypeBool(state),
		                      Binary(state, spv::OpBitwiseAnd, u32, flags, ConstantU32(state, SpirvLive::FlagIndirectLoadFast)),
		                      ConstantU32(state, 0)));
	}
	return EmitIndexSwitch(
	    state, Select(state, u32, guard, ConstantU32(state, 1), ConstantU32(state, 0)), 2u, TypeU32Composite(state, 2),
	    [&](uint32_t item) {
		    if (item == 0u) {
			    if (fast_mode == 2) {
				    (void)FaultDiagAdd(state, IndirectLoadFast::DiagWord + 2u, ConstantU32(state, 1));
				    // The V#'s format and DST_SEL of a load that took the function: a bit per format (words +3..+6) and
				    // per DST_SEL value of channel X (word +7).
				    const auto format = field(12, 7);
				    const auto bit    = [&](uint32_t value) {
                        return Binary(state, spv::OpShiftLeftLogical, u32, ConstantU32(state, 1),
                                      Binary(state, spv::OpBitwiseAnd, u32, value, ConstantU32(state, 31)));
				    };
				    const auto word = Binary(state, spv::OpIAdd, u32,
				                             ConstantU32(state, BufferCache::FAULT_DIAG_BASE + IndirectLoadFast::DiagWord + 3u),
				                             Binary(state, spv::OpShiftRightLogical, u32, format, ConstantU32(state, 5)));
				    state.builder.AddFunction(spv::OpAtomicOr, u32, state.builder.AllocateId(), FaultElementPointer(state, word),
				                              ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0), bit(format));
				    state.builder.AddFunction(
				        spv::OpAtomicOr, u32, state.builder.AllocateId(),
				        FaultElementPointer(state, ConstantU32(state, BufferCache::FAULT_DIAG_BASE + IndirectLoadFast::DiagWord + 7u)),
				        ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0), bit(field(0, 3)));
			    }
			    return call();
		    }
		    const auto fast = LoadIndirectD16x4HalfFast(
		        ctx, DecodeIndirectBuffer(state, ctx.Arg(handle, 0), ctx.Arg(handle, 1), ctx.Arg(handle, 2), word3, index,
		                                  offset, ctx.Arg(inst, 3)));
		    if (fast_mode != 2) return fast;
		    const auto old      = call();
		    const auto element  = [&](uint32_t composite, uint32_t i) {
                const auto id = state.builder.AllocateId();
                state.builder.AddFunction(spv::OpCompositeExtract, u32, id, composite, i);
                return id;
		    };
		    const auto mismatch = Binary(state, spv::OpLogicalOr, TypeBool(state),
		                                 Binary(state, spv::OpINotEqual, TypeBool(state), element(fast, 0), element(old, 0)),
		                                 Binary(state, spv::OpINotEqual, TypeBool(state), element(fast, 1), element(old, 1)));
		    (void)FaultDiagAdd(state, IndirectLoadFast::DiagWord, ConstantU32(state, 1));
		    (void)FaultDiagAdd(state, IndirectLoadFast::DiagWord + 1u,
		                       Select(state, u32, mismatch, ConstantU32(state, 1), ConstantU32(state, 0)));
		    return old;
	    });
}

// One DontInline function per shape of the formatted loads through GPU-selected
// descriptors that EmitLoadMemory sends to LoadIndirectFormatted and LoadIndirectFormattedX.
void DefineIndirectFormattedLoads(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op != IR::ValueOpcode::LoadBufferU32 && op != IR::ValueOpcode::LoadBufferU32x2 &&
			    op != IR::ValueOpcode::LoadBufferU32x3 && op != IR::ValueOpcode::LoadBufferU32x4)
				continue;
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= state.program.memory_info.size()) continue;
			const auto& mem = state.program.memory_info[index];
			if (mem.kind != IR::ResourceKind::IndirectBuffer || !mem.formatted) continue;
			const bool d16        = mem.data_bits == 16u;
			const auto components = IR::BufferComponentCount(op);
			const auto channels   = d16 ? mem.component_count : components;
			if (channels < 1u || channels > 4u) continue;
			state.indirect_formatted_load_functions.emplace(IndirectFormattedLoadKey(channels, components, d16), 0u);
		}
	}
	const auto u32 = TypeU32(state);
	for (auto& [key, function]: state.indirect_formatted_load_functions) {
		const auto channels      = key & 0xfu;
		const auto components    = (key >> 4u) & 0xfu;
		const bool d16           = ((key >> 8u) & 1u) != 0u;
		const auto result_type   = TypeU32Composite(state, components);
		const auto function_type = state.builder.Type(spv::OpTypeFunction, result_type, u32, u32, u32, u32, u32, u32,
		                                              u32);
		function = state.builder.AllocateId();
		std::string name = "indirect_formatted_load_";
		name.append("xyzw", channels);
		if (d16) name += "_d16";
		state.builder.AddName(function, name.c_str());
		state.builder.AddFunction(spv::OpFunction, result_type, function, spv::FunctionControlDontInlineMask,
		                          function_type);
		const auto parameter = [&]() {
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionParameter, u32, id);
			return id;
		};
		const auto word0   = parameter();
		const auto word1   = parameter();
		const auto records = parameter();
		const auto word3   = parameter();
		const auto index   = parameter();
		const auto offset  = parameter();
		const auto soffset = parameter();
		EmitLabel(state, state.builder.AllocateId());
		const auto buffer = DecodeIndirectBuffer(state, word0, word1, records, word3, index, offset, soffset);
		const auto result = LoadIndirectFormattedVector(ctx, buffer, word3, channels, components, d16);
		state.builder.AddFunction(spv::OpReturnValue, result);
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
}

// Formatted XY/XYZ/XYZW through a GPU-selected descriptor: each channel applies its own DST_SEL.
uint32_t LoadIndirectFormatted(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	LogCoherentFormattedIndirect(ctx, inst);
	const auto& mem = ctx.Memory(inst);
	const bool  d16 = mem.data_bits == 16u;
	return CallIndirectFormattedLoad(ctx, inst, d16 ? mem.component_count : components, components, d16);
}

// Compute stages (RobustBuffer): this load is a function call with no inline path, so the exec branch around it can go (IndirectCallWord3).
bool IndirectFormattedCallTakesGuard(const EmitterState& state, uint32_t channels, uint32_t components, bool d16) {
	if (!state.guardless) return false;
	// (KYTY_PERF_INDIRECT_LOAD_FAST's inline case of a listed shader reads memory with word 3 as it is.)
	if (IndirectLoadFast::Mode() != 0 && d16 && channels == 4u && components == 2u &&
	    IndirectLoadFast::Listed(state.program.shader_hash)) {
		return false;
	}
	const auto found = state.indirect_formatted_load_functions.find(IndirectFormattedLoadKey(channels, components, d16));
	return found != state.indirect_formatted_load_functions.end() && found->second != 0u;
}

uint32_t LoadIndirectFormattedX(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	LogCoherentFormattedIndirect(ctx, inst);
	if (IndirectFormattedCallTakesGuard(state, 1u, 1u, false)) {
		IndirectGuardScope guard(state, ctx.Arg(inst, inst.NumArgs() - 1));
		return CallIndirectFormattedLoad(ctx, inst, 1u, 1u, false);
	}
	return EmitValueOrZeroIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1),
	                                  [&]() { return CallIndirectFormattedLoad(ctx, inst, 1u, 1u, false); });
}

// Raw dword loads of a direct buffer. A GLC BUFFER_LOAD_DWORDX2 of an aligned pair reads both dwords with one 64-bit
// atomic load (upstream d1fe1324c), so a 64-bit value another invocation updates atomically is never seen torn; the
// u64 alias exists for it (ShaderInfoCollection sets buffer_int64_atomics). Partial, swizzled or out-of-range pairs keep
// the dword loads.
uint32_t LoadRawBuffer(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                       const MemoryResourceAccess& resource, uint32_t components) {
	auto&      state  = ctx.state;
	const auto scalar = [&] {
		std::array<uint32_t, 4> values {};
		for (uint32_t component = 0; component < components; component++) {
			values[component] =
			    LoadWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource);
		}
		return ConstructU32Composite(state, components, values);
	};
	if (!mem.IsCoherentBufferPairLoad(inst.GetOpcode())) return scalar();

	const auto first  = ByteAddress(ctx, inst, mem);
	const auto second = ByteAddress(ctx, inst, RebaseRawComponent(mem, 1));
	const auto wide   = PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3]);
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), first, ConstantU32(state, 3));
	auto atomic =
	    Binary(state, spv::OpIEqual, TypeBool(state),
	           Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 7)),
	           ConstantU32(state, 0));
	atomic = AndCondition(state, atomic,
	                      Binary(state, spv::OpIEqual, TypeBool(state), second,
	                             EmitAddU32(state, first, ConstantU32(state, 4))));
	// The 64-bit alias requires the entire pair; partial and swizzled reads retain DWORD bounds.
	atomic = AndCondition(state, atomic, EmitMemoryElementInBounds(state, wide, index));
	const auto atomic_label = state.builder.AllocateId(), scalar_label = state.builder.AllocateId();
	const auto merge = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, atomic, atomic_label, scalar_label);
	EmitLabel(state, atomic_label);
	const auto pointer = EmitStorageBufferElementPointer(
	    state, wide, index, TypeStorageBufferElementPointer(state, 64));
	const auto snapshot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU64(state), snapshot, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsAcquireMask |
	                                                 spv::MemorySemanticsUniformMemoryMask));
	const std::array<uint32_t, 4> words {
	    EmitCompositeExtractU64(state, snapshot, IR::Value(0u)),
	    EmitCompositeExtractU64(state, snapshot, IR::Value(1u))};
	const auto pair = ConstructU32Composite(state, 2, words);
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, scalar_label);
	const auto fallback    = scalar();
	const auto scalar_exit = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge);
	EmitLabel(state, merge);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32Composite(state, 2), result, pair, atomic_label,
	                          fallback, scalar_exit);
	return result;
}

uint32_t LoadBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	{
		// Raw (unformatted, direct) multi-dword reads are branchless; a coherent pair load keeps LoadRawBuffer's 64-bit
		// atomic read (never torn), below.
		const auto mem = ctx.Memory(inst);
		if (mem.kind != IR::ResourceKind::IndirectBuffer && !mem.formatted && !mem.IsCoherentBufferPairLoad(inst.GetOpcode())) {
			const auto              resource = PrepareMemoryResourceAccess(state, mem);
			const auto              guard    = ctx.Arg(inst, inst.NumArgs() - 1);
			std::array<uint32_t, 4> values {};
			for (uint32_t component = 0; component < components; component++) {
				values[component] = LoadWordBranchless(ctx, inst, RebaseRawComponent(mem, component), resource, guard);
			}
			return ConstructU32Composite(state, components, values);
		}
		if (mem.kind == IR::ResourceKind::IndirectBuffer && state.guardless) {
			const bool d16 = mem.data_bits == 16u;
			if (mem.formatted && IndirectFormattedCallTakesGuard(state, d16 ? mem.component_count : components, components, d16)) {
				IndirectGuardScope guard(state, ctx.Arg(inst, inst.NumArgs() - 1));
				return LoadIndirectFormatted(ctx, inst, components);
			}
			if (!mem.formatted && (mem.coherent ? state.indirect_raw_coherent_function : state.indirect_raw_function) != 0u) {
				IndirectGuardScope guard(state, ctx.Arg(inst, inst.NumArgs() - 1));
				return LoadIndirectBuffer(ctx, inst, components);
			}
		}
	}
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto mem      = ctx.Memory(inst);
		    if (mem.kind == IR::ResourceKind::IndirectBuffer) {
			    return mem.formatted ? LoadIndirectFormatted(ctx, inst, components)
			                         : LoadIndirectBuffer(ctx, inst, components);
		    }
		    const auto resource = PrepareMemoryResourceAccess(state, mem);
		    const auto info = Format::GetFormatInfo(
		        mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		    if (info.type != Format::ComponentType::Unknown) {
			    const auto plan = PrepareFormattedMemory(ctx, inst, mem, resource, info);
			    if (mem.data_bits == 16u) {
				    const auto channels = mem.component_count;
				    std::array<uint32_t, 4> oob {};
				    for (uint32_t channel = 0; channel < channels; ++channel)
					    oob[channel] = FormattedConstant(ctx, info,
					                                     ResolveFormattedSource(ctx, mem, info, channel).kind);
				    return EmitValueOrDefaultIfCondition(
				        state, plan.in_bounds, TypeU32Composite(state, components),
				        NullFormattedOutOfBounds(ctx, mem, components, PackD16Channels(state, info, oob, channels, components)),
				        [&]() {
					        std::array<uint32_t, 4> values {};
					        for (uint32_t channel = 0; channel < channels; ++channel)
						        values[channel] = LoadFormattedInBounds(ctx, mem, plan, channel);
					        return PackD16Channels(state, info, values, channels, components);
				        });
			    }
			    return EmitValueOrDefaultIfCondition(
			        state, plan.in_bounds, TypeU32Composite(state, components),
			        FormattedOutOfBoundsValue(ctx, mem, plan, components), [&]() {
				        std::array<uint32_t, 4> values {};
				        for (uint32_t component = 0; component < components; component++) {
					        values[component] = LoadFormattedInBounds(ctx, mem, plan, component);
				        }
				        return ConstructU32Composite(state, components, values);
			        });
		    }
		    return LoadRawBuffer(ctx, inst, mem, resource, components);
	    });
}

void StoreWideBuffer(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	const auto body = [&](uint32_t guard) {
		const auto mem       = ctx.Memory(inst);
		const auto resource  = PrepareMemoryResourceAccess(state, mem);
		const auto composite = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto info = Format::GetFormatInfo(
		    mem.formatted ? BufferFormat(ctx, mem) : Prospero::BufferFormat::kInvalid);
		if (info.type != Format::ComponentType::Unknown) {
			StoreFormattedPrepared(ctx, inst, mem, resource, info, composite, components);
			return;
		}
		for (uint32_t component = 0; component < components; component++) {
			const auto data = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), data, composite,
			                          component);
			StoreWordPrepared(ctx, inst, RebaseRawComponent(mem, component), resource, data, guard);
		}
	};
	const auto guard = ctx.Arg(inst, inst.NumArgs() - 1);
	if (const auto mem = ctx.Memory(inst); !mem.formatted && RobustBufferStore(state, mem)) {
		body(guard);
		return;
	}
	EmitIfCondition(state, guard, [&]() { body(0u); });
}

uint32_t LoadWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU32Composite(state, components),
	    ConstantU32CompositeZero(state, components), [&]() {
		    const auto              mem      = ctx.Memory(inst);
		    const auto              resource = PrepareMemoryResourceAccess(state, mem);
		    const auto              base     = ByteAddress(ctx, inst, mem);
		    std::array<uint32_t, 4> values {};
		    for (uint32_t component = 0; component < components; component++) {
			    const auto address   = component == 0u
			                               ? base
			                               : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                        ConstantU32(state, component * 4u));
			    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
			                                  address, ConstantU32(state, 2));
			    values[component]    = EmitValueOrZeroIfCondition(
			        state, EmitMemoryElementInBounds(state, resource, index),
			        [&]() { return LoadWordInBounds(ctx, resource, index); });
		    }
		    return ConstructU32Composite(state, components, values);
	    });
}

void StoreWideShared(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t components) {
	auto& state = ctx.state;
	EmitIfCondition(state, ctx.Arg(inst, inst.NumArgs() - 1), [&]() {
		const auto mem      = ctx.Memory(inst);
		const auto resource = PrepareMemoryResourceAccess(state, mem);
		const auto base     = ByteAddress(ctx, inst, mem);
		for (uint32_t component = 0; component < components; component++) {
			const auto address = component == 0u ? base
			                                     : Binary(state, spv::OpIAdd, TypeU32(state), base,
			                                              ConstantU32(state, component * 4u));
			const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
			                              ConstantU32(state, 2));
			EmitIfCondition(state, EmitMemoryElementInBounds(state, resource, index),
			                [&]() {
				                StoreWordInBounds(ctx, resource, index, ctx.Arg(inst, component + 1u));
			                });
		}
	});
}

} // namespace

uint32_t GetBdaPointer(EmitterState& state, uint32_t address) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunctionCall, TypeU64(state), result,
	                          state.bda_pointer_function, address);
	return result;
}

uint32_t FaultDiagAdd(EmitterState& state, uint32_t word, uint32_t value) {
	const auto old = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), old,
	                          FaultElementPointer(state, ConstantU32(state, BufferCache::FAULT_DIAG_BASE + word)),
	                          ConstantU32(state, spv::ScopeDevice), ConstantU32(state, 0), value);
	return old;
}

void FaultDiagStore(EmitterState& state, uint32_t first, uint32_t slot, uint32_t stride, uint32_t field, uint32_t value) {
	const auto index = Binary(state, spv::OpIAdd, TypeU32(state),
	                          ConstantU32(state, BufferCache::FAULT_DIAG_BASE + first + field),
	                          Binary(state, spv::OpIMul, TypeU32(state), slot, ConstantU32(state, stride)));
	state.builder.AddFunction(spv::OpStore, FaultElementPointer(state, index), value);
}

uint32_t WatchdogTripValues() {
	return BufferCache::FAULT_DIAG_TRIP_PHIS;
}

void ReportWatchdogTrip(EmitterState& state, uint32_t cap, uint32_t block_id, uint32_t counter, uint32_t lane,
                        uint32_t group, const std::vector<uint32_t>& low, const std::vector<uint32_t>& high) {
	const auto over = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), counter, ConstantU32(state, cap));
	const auto close_to_cap = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), counter,
	                         ConstantU32(state, cap > BufferCache::FAULT_DIAG_TRIP_LAST
	                                                ? cap - BufferCache::FAULT_DIAG_TRIP_LAST
	                                                : 0u));
	EmitIfCondition(state, close_to_cap, [&]() {
		// Trips (word 0) count invocations at the cap only; the reports take the last iterations before it.
		(void)FaultDiagAdd(state, 0, Select(state, TypeU32(state), over, ConstantU32(state, 1), ConstantU32(state, 0)));
		const auto slot = FaultDiagAdd(state, 1, ConstantU32(state, 1));
		EmitIfCondition(state,
		                Binary(state, spv::OpULessThan, TypeBool(state), slot,
		                       ConstantU32(state, BufferCache::FAULT_DIAG_TRIP_SLOTS)),
		                [&]() {
			const auto store = [&](uint32_t field, uint32_t value) {
				FaultDiagStore(state, BufferCache::FAULT_DIAG_TRIP_FIRST, slot, BufferCache::FAULT_DIAG_TRIP_WORDS, field,
				               value);
			};
			const auto hash = state.program.shader_hash;
			store(1, ConstantU32(state, static_cast<uint32_t>(hash)));
			store(2, ConstantU32(state, static_cast<uint32_t>(hash >> 32u)));
			store(3, counter);
			store(4, ConstantU32(state, static_cast<uint32_t>(low.size())));
			store(5, lane);
			store(6, group);
			for (size_t i = 0; i < low.size() && i < BufferCache::FAULT_DIAG_TRIP_PHIS; ++i)
				store(16 + static_cast<uint32_t>(i), low[i]);
			for (size_t i = 0; i < high.size() && i < BufferCache::FAULT_DIAG_TRIP_PHIS; ++i)
				store(16 + BufferCache::FAULT_DIAG_TRIP_PHIS + static_cast<uint32_t>(i), high[i]);
			store(0, ConstantU32(state, (0x5744u << 16u) | (block_id & 0xffffu))); // last: the report is complete
		});
	});
}

void ReportBdaStoreMiss(EmitterState& state, uint32_t address, uint32_t bda) {
	if (!DiagRtTripsFor(state.program.shader_hash) || state.fault_buffer_variable == 0) return;
	const auto missing = Binary(state, spv::OpIEqual, TypeBool(state), bda, ConstantU64(state, 0));
	EmitIfCondition(state, missing, [&]() {
		(void)FaultDiagAdd(state, 2, ConstantU32(state, 1));
		const auto slot = FaultDiagAdd(state, 3, ConstantU32(state, 1));
		EmitIfCondition(state,
		                Binary(state, spv::OpULessThan, TypeBool(state), slot,
		                       ConstantU32(state, BufferCache::FAULT_DIAG_MISS_SLOTS)),
		                [&]() {
			const auto u32  = TypeU32(state);
			const auto hash = state.program.shader_hash;
			const auto high = Binary(state, spv::OpShiftRightLogical, TypeU64(state), address, ConstantU32(state, 32));
			FaultDiagStore(state, BufferCache::FAULT_DIAG_MISS_FIRST, slot, 4, 0, Unary(state, spv::OpUConvert, u32, address));
			FaultDiagStore(state, BufferCache::FAULT_DIAG_MISS_FIRST, slot, 4, 1, Unary(state, spv::OpUConvert, u32, high));
			FaultDiagStore(state, BufferCache::FAULT_DIAG_MISS_FIRST, slot, 4, 2,
			               ConstantU32(state, static_cast<uint32_t>(hash)));
			FaultDiagStore(state, BufferCache::FAULT_DIAG_MISS_FIRST, slot, 4, 3,
			               ConstantU32(state, static_cast<uint32_t>(hash >> 32u)));
		});
	});
}

void DefineIndirectFormattedLoadFunctions(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	if (std::ranges::none_of(state.program.memory_info, [](const IR::MemoryInfo& memory) {
		    return memory.kind == IR::ResourceKind::IndirectBuffer && memory.formatted;
	    }))
		return;
	DefineIndirectFormattedLoads(ctx);
}

// A raw dword load through a GPU-selected buffer descriptor
// (bounds check + BDA translation + load) is emitted once as a non-inlined function and called per
// component, instead of being expanded at every load. Same result; much smaller SPIR-V for shaders
// with hundreds of such loads (driver compile time and memory).
void DefineIndirectRawDword(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	if (state.bda_pointer_function == 0u) return;
	// A GLC load (MemoryInfo::coherent) calls its own copy, whose load is coherent.
	for (const bool coherent: {false, true}) {
		if (std::ranges::none_of(state.program.memory_info, [&](const IR::MemoryInfo& memory) {
			    return memory.kind == IR::ResourceKind::IndirectBuffer && !memory.formatted &&
			           memory.coherent == coherent;
		    }))
			continue;
		const auto u32           = TypeU32(state);
		const auto u64           = TypeU64(state);
		const auto boolean       = TypeBool(state);
		const auto function_type = state.builder.Type(spv::OpTypeFunction, u32, u32, u32, u32, u32, u32, u32, u32, u32);
		auto&      function      = coherent ? state.indirect_raw_coherent_function : state.indirect_raw_function;
		function                 = state.builder.AllocateId();
		state.builder.AddName(function, coherent ? "indirect_raw_dword_coherent" : "indirect_raw_dword");
		state.builder.AddFunction(spv::OpFunction, u32, function, spv::FunctionControlDontInlineMask, function_type);
		const auto parameter = [&](uint32_t type) {
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionParameter, type, id);
			return id;
		};
		const auto word0        = parameter(u32);
		const auto word1        = parameter(u32);
		const auto records      = parameter(u32);
		const auto word3        = parameter(u32);
		const auto index        = parameter(u32);
		const auto offset       = parameter(u32);
		const auto soffset      = parameter(u32);
		const auto displacement = parameter(u32);
		EmitLabel(state, state.builder.AllocateId());
		// Same decode as PrepareIndirectBuffer, with the immediate already folded into `offset`.
		const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
			return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
		};
		const auto nonzero = [&](uint32_t value) {
			return Binary(state, spv::OpINotEqual, boolean, value, ConstantU32(state, 0));
		};
		IndirectBufferAccess buffer {};
		buffer.stride      = field(word1, 16, 14);
		buffer.swizzle     = AndCondition(state, nonzero(buffer.stride), nonzero(field(word1, 31, 1)));
		const auto address = CalculateBufferAddress(state, index, offset, soffset, 0u, buffer.stride, buffer.swizzle,
		                                            field(word3, 21, 2));
		const auto base    = PackU64(state, word0, field(word1, 0, 16));
		buffer.address     = Binary(state, spv::OpIAdd, u64, base, Unary(state, spv::OpUConvert, u64, address.byte));
		buffer.offset      = Binary(state, spv::OpIAdd, u32, address.offset, displacement);
		buffer.mode        = field(word3, 28, 2);
		buffer.records     = records;
		buffer.index_in_bounds     = Binary(state, spv::OpULessThan, boolean, index, records);
		buffer.scalar_in_bounds    = Binary(state, spv::OpULessThanEqual, boolean, soffset, records);
		buffer.raw_records         = Binary(state, spv::OpISub, u32, records, soffset);
		buffer.raw_index_in_bounds = Binary(state, spv::OpULessThan, boolean, index, buffer.raw_records);
		buffer.coherent            = coherent;
		const auto valid_format    = nonzero(field(word3, 12, 7));
		const auto dword_address   = Binary(state, spv::OpIAdd, u64,
		                                    Binary(state, spv::OpBitwiseAnd, u64, buffer.address,
		                                           ConstantU64(state, ~uint64_t {3})),
		                                    Unary(state, spv::OpUConvert, u64, displacement));
		const auto result = EmitValueOrZeroIfCondition(
		    state, AndCondition(state, valid_format, IndirectBufferInBounds(state, buffer, 0u, 4u, false)),
		    [&]() { return LoadBdaDword(ctx, dword_address, coherent); });
		state.builder.AddFunction(spv::OpReturnValue, result);
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
	// F1: indirect_raw_x<N>[_coherent] for the raw X2..X4 loads through a GPU-selected V# (ShaderIR.h, grouped
	// IndirectBuffer reads). Arguments as indirect_raw_dword's without the displacement; the result is the N dwords
	// indirect_raw_dword returns for displacements 0, 4, .. 4 (N - 1).
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			const auto components = IR::BufferComponentCount(inst.GetOpcode());
			if (components < 2u || components > 4u || IR::BufferAccessOf(inst.GetOpcode()) != IR::BufferAccess::Read) continue;
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= state.program.memory_info.size()) continue;
			const auto& mem = state.program.memory_info[index];
			if (mem.kind != IR::ResourceKind::IndirectBuffer || mem.formatted) continue;
			const auto per_dword = mem.coherent ? state.indirect_raw_coherent_function : state.indirect_raw_function;
			if (per_dword == 0u) continue;
			state.indirect_raw_group_functions[(components - 2u) + (mem.coherent ? 3u : 0u)] = per_dword; // marked
		}
	}
	for (uint32_t slot = 0; slot < state.indirect_raw_group_functions.size(); ++slot) {
		const auto per_dword = state.indirect_raw_group_functions[slot];
		if (per_dword == 0u) continue;
		const uint32_t components = slot % 3u + 2u;
		const bool     coherent   = slot >= 3u;
		const auto     u32        = TypeU32(state);
		const auto     u64        = TypeU64(state);
		const auto     boolean    = TypeBool(state);
		const auto     result_type = TypeU32Composite(state, components);
		const auto function_type = state.builder.Type(spv::OpTypeFunction, result_type, u32, u32, u32, u32, u32, u32, u32);
		const auto function      = state.builder.AllocateId();
		state.indirect_raw_group_functions[slot] = function;
		std::string name = "indirect_raw_x" + std::to_string(components);
		if (coherent) name += "_coherent";
		state.builder.AddName(function, name.c_str());
		state.builder.AddFunction(spv::OpFunction, result_type, function, spv::FunctionControlDontInlineMask, function_type);
		const auto parameter = [&]() {
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpFunctionParameter, u32, id);
			return id;
		};
		const auto word0   = parameter();
		const auto word1   = parameter();
		const auto records = parameter();
		const auto word3   = parameter();
		const auto index   = parameter();
		const auto offset  = parameter();
		const auto soffset = parameter();
		EmitLabel(state, state.builder.AllocateId());
		// indirect_raw_dword's decode, once: the per-dword displacement goes into each dword's bounds test instead.
		const auto field = [&](uint32_t word, uint32_t first, uint32_t count) {
			return EmitBitFieldUExtract(state, word, ConstantU32(state, first), ConstantU32(state, count));
		};
		const auto nonzero = [&](uint32_t value) {
			return Binary(state, spv::OpINotEqual, boolean, value, ConstantU32(state, 0));
		};
		IndirectBufferAccess buffer {};
		buffer.stride      = field(word1, 16, 14);
		buffer.swizzle     = AndCondition(state, nonzero(buffer.stride), nonzero(field(word1, 31, 1)));
		const auto address = CalculateBufferAddress(state, index, offset, soffset, 0u, buffer.stride, buffer.swizzle,
		                                            field(word3, 21, 2));
		const auto base    = PackU64(state, word0, field(word1, 0, 16));
		buffer.address     = Binary(state, spv::OpIAdd, u64, base, Unary(state, spv::OpUConvert, u64, address.byte));
		buffer.offset      = address.offset;
		buffer.mode        = field(word3, 28, 2);
		buffer.records     = records;
		buffer.index_in_bounds     = Binary(state, spv::OpULessThan, boolean, index, records);
		buffer.scalar_in_bounds    = Binary(state, spv::OpULessThanEqual, boolean, soffset, records);
		buffer.raw_records         = Binary(state, spv::OpISub, u32, records, soffset);
		buffer.raw_index_in_bounds = Binary(state, spv::OpULessThan, boolean, index, buffer.raw_records);
		buffer.coherent            = coherent;
		// Every dword in bounds (its own test, as indirect_raw_dword makes it) and all of them in one page.
		auto all = nonzero(field(word3, 12, 7));
		for (uint32_t i = 0; i < components; ++i) {
			all = AndCondition(state, all, IndirectBufferInBounds(state, buffer, i * 4u, 4u, false));
		}
		const auto first = Binary(state, spv::OpBitwiseAnd, u64, buffer.address, ConstantU64(state, ~uint64_t {3}));
		const auto fast  = AndCondition(state, all, InOnePage(state, first, components * 4u));
		const auto today = [&]() {
			std::array<uint32_t, 4> values {};
			for (uint32_t i = 0; i < components; ++i) {
				values[i] = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpFunctionCall, u32, values[i], per_dword, word0, word1, records, word3,
				                          index, offset, soffset, ConstantU32(state, i * 4u));
			}
			return values;
		};
		const auto values = EmitIfElseU32s(
		    state, fast, components,
		    [&]() {
			    const auto bda     = GetBdaPointer(state, first);
			    const auto present = Binary(state, spv::OpINotEqual, boolean, bda, ConstantU64(state, 0));
			    const auto loaded  = EmitValueOrDefaultIfCondition(
			         state, present, result_type, ConstantU32CompositeZero(state, components),
			         [&]() { return ConstructU32Composite(state, components, LoadBdaDwordsAt(state, bda, components, coherent)); });
			    std::array<uint32_t, 4> fresh {};
			    for (uint32_t i = 0; i < components; ++i) {
				    fresh[i] = state.builder.AllocateId();
				    state.builder.AddFunction(spv::OpCompositeExtract, u32, fresh[i], loaded, i);
			    }
			    return fresh;
		    },
		    today);
		state.builder.AddFunction(spv::OpReturnValue, ConstructU32Composite(state, components, values));
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
}

// A formatted X..XYZW store of 32-bit VGPR data through a GPU-selected descriptor, written by StoreIndirectFormatted
// (record-keyed or not): its component count, else 0.
static uint32_t IndirectFormattedStoreComponents(const IR::Inst& inst, const IR::MemoryInfo& mem) {
	if (mem.kind != IR::ResourceKind::IndirectBuffer || !mem.formatted || mem.data_bits != 32u) return 0u;
	return std::max(IR::BufferComponentCount(inst.GetOpcode()), 1u);
}

// KYTY_PERF_SPIRV_STORE_OUTLINE=1 (SpirvStoreOutlineEnabled, default off): one DontInline function per shape of the
// stores through GPU-selected descriptors, each defined at most once per module and called where the store is
// (StoreIndirectFormatted, StoreIndirectBuffer):
//   indirect_formatted_store_x..xyzw(word0, word1, records, word3, index, offset, soffset, data): the V# decode, the
//     element's bounds (once per element size of the supported formats instead of once per format case), the
//     runtime-format switch and the BDA piece writes with their tracking (RecordBdaTarget / RecordBdaWrite) exactly as
//     the inline path expands them at every store;
//   indirect_raw_store_dword(word0, word1, records, word3, index, offset, soffset, displacement, value): one dword of
//     a raw store, as indirect_raw_dword is for the raw loads.
// The ADD_TID lane (it depends on the wave half) and the instruction's immediate offset stay at the call
// (PrepareIndirectStoreCall). Same bytes written in the same order per lane; the switch is emitted once.
void DefineIndirectStoreFunctions(ValueEmitContext& ctx) {
	auto& state = ctx.state;
	if (!SpirvStoreOutlineEnabled() || state.bda_pointer_function == 0u) return;
	bool raw_store = false, raw_store_coherent = false;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (!IsDwordBufferStore(op)) continue;
			const auto index = inst.Flags<IR::MemoryFlags>().index;
			if (index >= state.program.memory_info.size()) continue;
			const auto& mem = state.program.memory_info[index];
			if (mem.kind != IR::ResourceKind::IndirectBuffer) continue;
			if (const auto components = IndirectFormattedStoreComponents(inst, mem); components != 0u) {
				state.indirect_formatted_store_functions.emplace(components, 0u);
			} else if (!mem.formatted) {
				(mem.coherent ? raw_store_coherent : raw_store) = true; // StoreIndirectBuffer (EmitStoreMemory)
			}
		}
	}
	const auto u32     = TypeU32(state);
	const auto u64     = TypeU64(state);
	const auto boolean = TypeBool(state);
	const auto parameter = [&](uint32_t type) {
		const auto id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionParameter, type, id);
		return id;
	};
	constexpr auto format_count = static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) + 1u;
	for (auto& [shape, function]: state.indirect_formatted_store_functions) {
		const uint32_t components    = shape;
		const auto     data_type     = TypeU32Composite(state, components);
		const auto function_type = state.builder.Type(spv::OpTypeFunction, TypeVoid(state), u32, u32, u32, u32, u32,
		                                              u32, u32, data_type);
		function                 = state.builder.AllocateId();
		std::string name         = "indirect_formatted_store_";
		name.append("xyzw", components);
		state.builder.AddName(function, name.c_str());
		state.builder.AddFunction(spv::OpFunction, TypeVoid(state), function, spv::FunctionControlDontInlineMask,
		                          function_type);
		const auto word0   = parameter(u32);
		const auto word1   = parameter(u32);
		const auto records = parameter(u32);
		const auto word3   = parameter(u32);
		const auto index   = parameter(u32);
		const auto offset  = parameter(u32);
		const auto soffset = parameter(u32);
		const auto data    = parameter(data_type);
		EmitLabel(state, state.builder.AllocateId());
		const auto buffer = DecodeIndirectBuffer(state, word0, word1, records, word3, index, offset, soffset);
		// The element's bounds once per element size (1, 2, 4, 8, 12, 16 bytes); every format case of that size
		// takes its result (the inline path rebuilds the check in every case).
		std::map<uint32_t, uint32_t> in_bounds;
		for (uint32_t format = 0; format < format_count; ++format) {
			const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
			if (!IndirectFormattedStoreSupported(info) || in_bounds.contains(info.byte_size)) continue;
			in_bounds.emplace(info.byte_size, IndirectBufferInBounds(state, buffer, 0u, info.byte_size, true));
		}
		EmitIndexSwitch(state, buffer.format, format_count, 0u, [&](uint32_t format) {
			const auto info = Format::GetFormatInfo(static_cast<Prospero::BufferFormat>(format));
			if (!IndirectFormattedStoreSupported(info)) return;
			StoreIndirectFormattedCase(ctx, buffer, info, in_bounds.at(info.byte_size), data, components, 32u);
		});
		state.builder.AddFunction(spv::OpReturn);
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
	// A GLC store (MemoryInfo::coherent) calls its own copy, whose store is coherent.
	for (const bool coherent: {false, true}) {
		if (!(coherent ? raw_store_coherent : raw_store)) continue;
		const auto function_type = state.builder.Type(spv::OpTypeFunction, TypeVoid(state), u32, u32, u32, u32, u32,
		                                              u32, u32, u32, u32);
		auto& function = coherent ? state.indirect_raw_store_coherent_function : state.indirect_raw_store_function;
		function       = state.builder.AllocateId();
		state.builder.AddName(function, coherent ? "indirect_raw_store_dword_coherent" : "indirect_raw_store_dword");
		state.builder.AddFunction(spv::OpFunction, TypeVoid(state), function, spv::FunctionControlDontInlineMask,
		                          function_type);
		const auto word0        = parameter(u32);
		const auto word1        = parameter(u32);
		const auto records      = parameter(u32);
		const auto word3        = parameter(u32);
		const auto index        = parameter(u32);
		const auto offset       = parameter(u32);
		const auto soffset      = parameter(u32);
		const auto displacement = parameter(u32);
		const auto value        = parameter(u32);
		EmitLabel(state, state.builder.AllocateId());
		// StoreIndirectBuffer's component: the dword `displacement` bytes past the aligned element, in bounds when the
		// element's offset plus the displacement is (as DefineIndirectRawDword folds it into the decoded offset).
		auto buffer   = DecodeIndirectBuffer(state, word0, word1, records, word3, index, offset, soffset);
		buffer.offset = Binary(state, spv::OpIAdd, u32, buffer.offset, displacement);
		const auto valid_format = Binary(state, spv::OpINotEqual, boolean, buffer.format, ConstantU32(state, 0));
		const auto address = Binary(state, spv::OpIAdd, u64,
		                            Binary(state, spv::OpBitwiseAnd, u64, buffer.address, ConstantU64(state, ~uint64_t {3})),
		                            Unary(state, spv::OpUConvert, u64, displacement));
		EmitIfCondition(state, AndCondition(state, valid_format, IndirectBufferInBounds(state, buffer, 0u, 4u, false)),
		                [&]() {
			                const auto bda = GetBdaPointer(state, address);
			                RecordBdaTarget(state, address); // KYTY_RECORD_KEYED_VERIFY
			                ReportBdaStoreMiss(state, address, bda); // KYTY_DIAG_RT_TRIPS
			                EmitIfCondition(state, Binary(state, spv::OpINotEqual, boolean, bda, ConstantU64(state, 0)),
			                                [&]() {
				                                state.builder.AddFunction(spv::OpStore,
				                                                          BdaDwordPointer(state, bda, coherent), value,
				                                                          BdaDwordAccess(coherent), 4u);
				                                RecordBdaWrite(state, address);
			                                });
		                });
		state.builder.AddFunction(spv::OpReturn);
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
}

static bool ScalarGroupCandidate(const EmitterState& state, const IR::Inst& inst) {
	if (inst.GetOpcode() != IR::ValueOpcode::ReadConstBuffer) return false;
	const auto index = inst.Flags<IR::MemoryFlags>().index;
	if (index >= state.program.memory_info.size()) return false;
	const auto& mem = state.program.memory_info[index];
	return mem.kind == IR::ResourceKind::IndirectBuffer && !mem.planning_only;
}

// Same V# words and dynamic offset: the loads differ only in their constant offsets.
static bool ScalarGroupSibling(const IR::Inst& leader, const IR::Inst& inst) {
	if (!(inst.Arg(1).Resolve() == leader.Arg(1).Resolve())) return false;
	const auto* a = leader.Arg(0).ResolveInstruction();
	const auto* b = inst.Arg(0).ResolveInstruction();
	if (a == b) return true;
	if (a == nullptr || b == nullptr || a->GetOpcode() != b->GetOpcode() || a->NumArgs() != b->NumArgs()) return false;
	for (size_t i = 0; i < a->NumArgs(); i++) {
		if (!(a->Arg(i).Resolve() == b->Arg(i).Resolve())) return false;
	}
	return true;
}

void FindScalarLoadGroups(EmitterState& state) {
	state.scalar_groups.clear();
	state.scalar_group_leader.clear();
	state.scalar_group_members.clear();
	state.scalar_group_values[0].clear();
	state.scalar_group_values[1].clear();
	if (state.bda_pointer_function == 0u) {
		return;
	}
	constexpr size_t   max_gap  = 64;   // instructions between two members
	constexpr uint64_t max_span = 4096; // bytes from the lowest to the end of the highest member
	for (const auto* block: state.program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			if (!ScalarGroupCandidate(state, *it) || state.scalar_group_members.contains(&*it)) continue;
			EmitterState::ScalarLoadGroup group;
			std::vector<uint64_t>         offsets;
			const uint64_t first = state.program.memory_info[it->Flags<IR::MemoryFlags>().index].offset & ~3u;
			group.members.push_back(&*it);
			offsets.push_back(first);
			group.c_min = first;
			group.c_max = first;
			size_t gap  = 0;
			for (auto next = std::next(it); next != block->end() && gap < max_gap; ++next) {
				if (ScalarGroupCandidate(state, *next) && !state.scalar_group_members.contains(&*next) &&
				    ScalarGroupSibling(*it, *next)) {
					const uint64_t offset = state.program.memory_info[next->Flags<IR::MemoryFlags>().index].offset & ~3u;
					const auto     low    = std::min(group.c_min, offset);
					const auto     high   = std::max(group.c_max, offset);
					if (high - low + sizeof(uint32_t) > max_span) break;
					group.members.push_back(&*next);
					offsets.push_back(offset);
					group.c_min = low;
					group.c_max = high;
					gap         = 0;
				} else if (next->MayHaveSideEffects()) {
					break;
				} else {
					++gap;
				}
			}
			if (group.members.size() < 2u) continue;
			for (const auto offset: offsets) {
				group.deltas.push_back(offset - group.c_min);
			}
			for (size_t member = 1; member < group.members.size(); member++) {
				state.scalar_group_members.insert(group.members[member]);
			}
			state.scalar_group_leader.emplace(&*it, static_cast<uint32_t>(state.scalar_groups.size()));
			state.scalar_groups.push_back(std::move(group));
		}
	}
}

void DeclareBdaEntryShare(EmitterState& state) {
	state.bda_memo_page = {};
	state.bda_memo_base = {};
	if (state.bda_pointer_function == 0u || BufferCache::BdaEntryShareMode() == 0) {
		return;
	}
	// KYTY_PERF_BDA_ENTRY_SHARE_HASHES=<hex,...> | all: only these shaders (a test without recompiling the hundreds of
	// shaders that load through GPU-built V#s). Default: the cloth simulation fc6f8c56eb7e168f.
	static const std::string only = [] {
		const char*       value = Common::ShaderEnv::Read("KYTY_PERF_BDA_ENTRY_SHARE_HASHES");
		const std::string v     = value != nullptr ? std::string(value) : std::string("fc6f8c56eb7e168f");
		return v == "all" ? std::string() : v;
	}();
	if (!only.empty()) {
		char text[17];
		std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(state.program.shader_hash));
		if (only.find(text) == std::string::npos) return;
	}
	const auto pointer = TypePointer(state, spv::StorageClassFunction, TypeU64(state));
	for (uint32_t half = 0; half < state.lane_count && half < 2u; half++) {
		state.bda_memo_page[half] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpVariable, pointer, state.bda_memo_page[half], spv::StorageClassFunction,
		                          ConstantU64(state, ~uint64_t {0}));
		state.bda_memo_base[half] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpVariable, pointer, state.bda_memo_base[half], spv::StorageClassFunction,
		                          ConstantU64(state, 0));
	}
}

void DefineGetBdaPointer(EmitterState& state) {
	if (!state.program.info.uses_dma) {
		return;
	}
	const auto type            = TypeU64(state);
	const auto function_type   = state.builder.Type(spv::OpTypeFunction, type, type);
	state.bda_pointer_function = state.builder.AllocateId();
	const auto address         = state.builder.AllocateId();
	const auto entry_label     = state.builder.AllocateId();
	state.builder.AddName(state.bda_pointer_function, "get_bda_pointer");
	state.builder.AddFunction(spv::OpFunction, type, state.bda_pointer_function,
	                          spv::FunctionControlMaskNone, function_type);
	state.builder.AddFunction(spv::OpFunctionParameter, type, address);
	EmitLabel(state, entry_label);

	const auto extended = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), address,
	                             ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase));
	const auto packed   = Select(
	    state, type, extended,
	    Binary(state, spv::OpISub, type, address,
	           ConstantU64(state, LibKernel::Memory::kExtendedMemoryBase - LOWER_ADDRESS_SIZE)),
	    address);
	const auto page64 = Binary(state, spv::OpShiftRightLogical, type, packed,
	                           ConstantU32(state, BufferCache::CACHING_PAGEBITS));
	// An address past the guest page table must not index it. BDA has no robustness, so an
	// out-of-range page is a missing page (page 0 is only used as a safe index).
	const auto in_range = Binary(state, spv::OpULessThan, TypeBool(state), page64,
	                             ConstantU64(state, BufferCache::CACHING_NUMPAGES));
	const auto page_raw = Unary(state, spv::OpUConvert, TypeU32(state), page64);
	const auto page     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), page, in_range, page_raw,
	                          ConstantU32(state, 0));
	const auto entry_pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state, 64),
	                          entry_pointer, state.bda_pagetable_variable, ConstantU32(state, 0),
	                          page);
	const auto loaded = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, loaded, entry_pointer);
	// The BDA fallback (M06): bit 0 marks an entry into the imported guest backing: the access uses it (the guest's real
	// bytes) and the page is still reported, so the host gives it a device-local buffer.
	const auto base = Binary(state, spv::OpBitwiseAnd, type, loaded, ConstantU64(state, ~uint64_t {1}));
	const auto null_base =
	    Binary(state, spv::OpIEqual, TypeBool(state), base, ConstantU64(state, 0));
	const auto out_of_range = Unary(state, spv::OpLogicalNot, TypeBool(state), in_range);
	const auto missing =
	    Binary(state, spv::OpLogicalOr, TypeBool(state), null_base, out_of_range);
	// One selection, both arms labelled, as below (a selection whose false target was its own merge block crashed
	// the NVIDIA compiler): report a missing or fallback page; the value is 0 for a missing one.
	const auto marked = Binary(state, spv::OpINotEqual, TypeBool(state),
	                           Binary(state, spv::OpBitwiseAnd, type, loaded, ConstantU64(state, 1)), ConstantU64(state, 0));
	const auto report = Binary(state, spv::OpLogicalOr, TypeBool(state), missing, marked);
	const auto offset = Binary(state, spv::OpBitwiseAnd, type, address,
	                           ConstantU64(state, BufferCache::CACHING_PAGESIZE - 1));
	const auto value  = Select(state, type, missing, ConstantU64(state, 0), Binary(state, spv::OpIAdd, type, base, offset));
	const auto report_label = state.builder.AllocateId();
	const auto skip_label   = state.builder.AllocateId();
	const auto done_label   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, done_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, report, report_label, skip_label);
	EmitLabel(state, report_label);
	RecordBdaFault(state, page);
	state.builder.AddFunction(spv::OpBranch, done_label);
	EmitLabel(state, skip_label);
	state.builder.AddFunction(spv::OpBranch, done_label);
	EmitLabel(state, done_label);
	state.builder.AddFunction(spv::OpReturnValue, value);
	state.builder.AddFunction(spv::OpFunctionEnd);
}


uint32_t EmitAtomic32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto condition = ctx.Arg(inst, inst.NumArgs() - 1);
	const auto& mem = ctx.Memory(inst);
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		// record-keyed (D6): an atomic through a GPU-selected descriptor operates on
		// the BDA mapping of its address; a page without a mapping is skipped (returns 0).
		auto& state = ctx.state;
		return EmitValueOrZeroIfCondition(state, condition, [&]() {
			const auto buffer       = PrepareIndirectBuffer(ctx, inst);
			const auto valid_format = Binary(state, spv::OpINotEqual, TypeBool(state), buffer.format,
			                                 ConstantU32(state, 0));
			const auto address      = Binary(state, spv::OpBitwiseAnd, TypeU64(state), buffer.address,
			                                 ConstantU64(state, ~uint64_t {3}));
			return EmitValueOrZeroIfCondition(
			    state, AndCondition(state, valid_format, IndirectBufferInBounds(state, buffer, 0u, 4u, false)),
			    [&]() {
				    const auto bda = GetBdaPointer(state, address);
				    RecordBdaTarget(state, address); // KYTY_RECORD_KEYED_VERIFY
				    ReportBdaStoreMiss(state, address, bda); // KYTY_DIAG_RT_TRIPS
				    return EmitValueOrZeroIfCondition(
				        state, Binary(state, spv::OpINotEqual, TypeBool(state), bda, ConstantU64(state, 0)),
				        [&]() {
					        const auto pointer = state.builder.AllocateId();
					        state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state),
					                                  pointer, bda);
					        const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
					        RecordBdaWrite(state, address);
					        return old;
				        });
			    });
		});
	}
	// DS_ADD_RTN_U32 / DS_SUB_RTN_U32 return their old values in lane order, as the hardware does, for the lanes that
	// address the same dword (M110: the RTGI BVH builder's tree rotations, f672d6a1, depend on it; with one atomic per
	// invocation the host's order made parent loops and the ray traversal spun to the loop watchdog): a loop over the distinct addresses of the subgroup's active
	// invocations, where the lowest invocation of an address adds (subtracts) the group's total and every invocation gets
	// the old value plus (minus) the exclusive prefix of the lower lanes' values. Every active invocation runs every
	// iteration and takes part in every subgroup operation (the exit test is a subgroup minimum); only the atomic itself
	// is conditional. A loop whose invocations leave it at different iterations would rely on reconvergence the host
	// does not guarantee (NVIDIA mixed iterations: duplicate ranks in 32acc45f). One-half shaders and both halves of a
	// lane_count 2 shader (emitted one after the other, so lanes 0..31 come before 32..63); per-lane LDS slots
	// (compact_lds) need no order.
	const auto opcode  = inst.GetOpcode();
	const bool ordered = mem.kind == IR::ResourceKind::Lds &&
	                     (opcode == IR::ValueOpcode::SharedAtomicIAdd32 || opcode == IR::ValueOpcode::SharedAtomicISub32) &&
	                     ctx.state.program.info.ordered_lds_atomics && inst.HasUses() && !ctx.state.compact_lds;
	if (ordered && ctx.state.wave_pair) {
		// The two host subgroups of a paired guest wave are not ordered against each other: no lane order here.
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 8u)
			std::fprintf(stderr, "LDS_ATOMIC_ORDER: wave64 pair shader 0x%016llx keeps per-invocation LDS atomics\n",
			             static_cast<unsigned long long>(ctx.state.program.shader_hash));
	}
	const bool lane_order = ordered && !ctx.state.wave_pair;
	EmitPairLdsWriteSync(ctx.state, mem, true);
	const auto result = EmitAtomicAccessIf(ctx, inst, mem, condition, [&](uint32_t pointer) {
		const auto scope =
		    mem.kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
		if (!lane_order) {
			const auto old = EmitAtomicOperation(ctx, inst, pointer, scope);
			EmitAtomicMemoryBarrier(ctx.state, mem.kind);
			return old;
		}
		auto&      s        = ctx.state;
		const bool sub      = opcode == IR::ValueOpcode::SharedAtomicISub32;
		const auto u32      = TypeU32(s);
		const auto subgroup = ConstantU32(s, spv::ScopeSubgroup);
		const auto value    = ctx.Arg(inst, inst.NumArgs() - 2);
		const auto index    = DwordIndex(ctx, inst, mem);
		const auto boolean  = TypeBool(s);
		const auto none     = ConstantU32(s, 0xffffffffu); // no LDS dword index reaches it
		const auto entry    = s.current_label;
		const auto header = s.builder.AllocateId(), body = s.builder.AllocateId(), work = s.builder.AllocateId();
		const auto cont = s.builder.AllocateId(), merge = s.builder.AllocateId();
		const auto pending = s.builder.AllocateId(), old = s.builder.AllocateId();
		const auto next_pending = s.builder.AllocateId(), next_old = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpBranch, header);
		EmitLabel(s, header);
		s.builder.AddFunction(spv::OpPhi, boolean, pending, ConstantBool(s, true), entry, next_pending, cont);
		s.builder.AddFunction(spv::OpPhi, u32, old, ConstantU32(s, 0), entry, next_old, cont);
		s.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
		s.builder.AddFunction(spv::OpBranch, body);
		EmitLabel(s, body);
		// The smallest address still pending in the subgroup (the same value in every invocation); none left: done.
		const auto chosen = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformUMin, u32, chosen, subgroup, spv::GroupOperationReduce,
		                      Select(s, u32, pending, index, none));
		s.builder.AddFunction(spv::OpBranchConditional, Binary(s, spv::OpIEqual, boolean, chosen, none), merge, work);
		EmitLabel(s, work);
		const auto mine = AndCondition(s, pending, Binary(s, spv::OpIEqual, boolean, index, chosen));
		const auto part = Select(s, u32, mine, value, ConstantU32(s, 0));
		const auto total = s.builder.AllocateId(), prefix = s.builder.AllocateId(), rank = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, total, subgroup, spv::GroupOperationReduce, part);
		s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, prefix, subgroup, spv::GroupOperationExclusiveScan, part);
		s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, rank, subgroup, spv::GroupOperationExclusiveScan,
		                      Select(s, u32, mine, ConstantU32(s, 1), ConstantU32(s, 0)));
		const auto leader = AndCondition(s, mine, Binary(s, spv::OpIEqual, boolean, rank, ConstantU32(s, 0)));
		const auto leader_old = EmitValueOrZeroIfCondition(s, leader, [&]() {
			const auto result = s.builder.AllocateId();
			s.builder.AddFunction(sub ? spv::OpAtomicISub : spv::OpAtomicIAdd, u32, result, pointer, ConstantU32(s, scope),
			                      ConstantU32(s, 0), total);
			return result;
		});
		// One leader adds a non-zero term: the sum is its old value.
		const auto base = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, base, subgroup, spv::GroupOperationReduce, leader_old);
		const auto lane_old = Binary(s, sub ? spv::OpISub : spv::OpIAdd, u32, base, prefix);
		s.builder.AddFunction(spv::OpSelect, u32, next_old, mine, lane_old, old);
		s.builder.AddFunction(spv::OpLogicalAnd, boolean, next_pending, pending,
		                      Unary(s, spv::OpLogicalNot, boolean, mine));
		s.builder.AddFunction(spv::OpBranch, cont);
		EmitLabel(s, cont);
		s.builder.AddFunction(spv::OpBranch, header);
		EmitLabel(s, merge);
		EmitAtomicMemoryBarrier(s, mem.kind);
		return old;
	}, !lane_order);
	EmitPairLdsWriteSync(ctx.state, mem, false);
	return result;
}

uint32_t EmitBufferAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem   = ctx.Memory(inst);
	auto&       state = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    state, ctx.Arg(inst, inst.NumArgs() - 1), TypeU64(state), ConstantU64(state, 0), [&]() {
		    const auto resource =
		        PrepareStorageBufferResourceAccess(state, mem, state.storage_buffers[3]);
		    const auto byte_address = ByteAddress(ctx, inst, mem);
		    const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), byte_address,
		                              ConstantU32(state, 3u));
		    return EmitValueOrDefaultIfCondition(
		        state, EmitMemoryElementInBounds(state, resource, index), TypeU64(state),
		        ConstantU64(state, 0), [&]() {
			        const auto pointer =
			            EmitStorageBufferElementPointer(state, resource, index, TypeStorageBufferElementPointer(state, 64));
			        const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			        EmitAtomicMemoryBarrier(state, mem.kind);
			        return old;
		        });
	    });
}

// Returning 64-bit LDS atomics and DS_CMPST_RTN_B64 (upstream 9e41d8320): the old value is defined when the instruction
// has uses (0 for an inactive or out-of-range lane).
void EmitSharedAtomic64(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	EnsureLdsStorage(state);
	const auto address = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                            ByteAddress(ctx, inst, mem), ConstantU32(state, 0xfff8u));
	const auto index = Binary(state, spv::OpShiftRightLogical, TypeU32(state), address,
	                          ConstantU32(state, 3u));
	const auto in_bounds = Binary(state, spv::OpULessThan, TypeBool(state), index,
	                              ConstantU32(state, LdsDwordCount(state) / 2u));
	const auto active = EmitLogicalAndBool(state, ctx.Arg(inst, inst.NumArgs() - 1), in_bounds);
	const auto emit = [&]() {
		auto native_index = index;
		auto semantics = spv::MemorySemanticsWorkgroupMemoryMask;
		if (state.lds_storage_class == spv::StorageClassStorageBuffer) {
			native_index = EmitAddU32(state, index, EmitBinaryU32(
			    state, spv::OpShiftRightLogical, state.lds_base_dwords, ConstantU32(state, 1)));
			semantics = spv::MemorySemanticsUniformMemoryMask;
		}
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, state.lds_storage_class, TypeU64(state)),
		    pointer, state.lds_u64_variable, ConstantU32(state, 0), native_index);
		const auto old = state.builder.AllocateId();
		const auto opcode = SpirvAtomicOpcode(inst.GetOpcode());
		const auto scope = ConstantU32(state, spv::ScopeWorkgroup);
		const auto acquire_release =
		    ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask | semantics);
		if (opcode == spv::OpAtomicCompareExchange) {
			// RDNA2 DS compares DATA and stores DATA2, opposite to BUFFER atomics.
			state.builder.AddFunction(
			    opcode, TypeU64(state), old, pointer, scope, acquire_release,
			    ConstantU32(state, spv::MemorySemanticsAcquireMask | semantics),
			    ctx.Arg(inst, 2), ctx.Arg(inst, 1));
		} else {
			state.builder.AddFunction(opcode, TypeU64(state), old, pointer, scope,
			                          acquire_release, ctx.Arg(inst, 1));
		}
		return old;
	};
	EmitPairLdsWriteSync(state, mem, true);
	if (inst.HasUses()) {
		ctx.Define(inst, EmitValueOrDefaultIfCondition(
		    state, active, TypeU64(state), ConstantU64(state, 0), emit));
	} else {
		EmitIfCondition(state, active, emit);
	}
	EmitPairLdsWriteSync(state, mem, false);
}

uint32_t EmitBufferFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::BufferAtomicFMax32;
	return EmitAtomicUpdate(ctx, inst, mem,
	                        [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                        return EmitFloatAtomicReplacement(state, old, value, max_value);
	                        });
}

void EmitSharedFloatAtomic(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& mem       = ctx.Memory(inst);
	const bool  max_value = inst.GetOpcode() == IR::ValueOpcode::SharedAtomicFMax32;
	EmitAtomicUpdate(ctx, inst, mem,
	                 [max_value](EmitterState& state, uint32_t old, uint32_t value) {
		                 return EmitDsFloatAtomicReplacement(state, old, value, max_value);
	                 });
}

uint32_t EmitAppendConsume(ValueEmitContext& ctx, const IR::Inst& inst) {
	const bool append = inst.GetOpcode() == IR::ValueOpcode::DataAppend;
	auto&      state  = ctx.state;
	if (ctx.half == 1) {
		return ctx.other_half->Def(IR::Value(const_cast<IR::Inst*>(&inst)));
	}
	const auto mem           = ctx.Memory(inst);
	const auto offset        = mem.offset & 0xfffcu;
	auto       address       = ConstantU32(state, offset);
	uint32_t   region_bounds = 0;
	if (mem.kind == IR::ResourceKind::Gds) {
		const auto m0 = ctx.Arg(inst, 0);
		const auto base =
		    Binary(state, spv::OpShiftRightLogical, TypeU32(state), m0, ConstantU32(state, 16));
		const auto size =
		    Binary(state, spv::OpBitwiseAnd, TypeU32(state), m0, ConstantU32(state, 0xffffu));
		address = Binary(state, spv::OpIAdd, TypeU32(state), base, address);
		// The entire M0 region must fit the PS5's 48 KiB GDS partition.
		region_bounds = AndCondition(
		    state, Binary(state, spv::OpULessThan, TypeBool(state),
		                  ConstantU32(state, offset + 3u), size),
		    Binary(state, spv::OpULessThanEqual, TypeBool(state),
		           Binary(state, spv::OpIAdd, TypeU32(state), base, size),
		           ConstantU32(state, 0xc000u)));
	}
	const auto index =
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), address, ConstantU32(state, 2));
	const auto access = PrepareMemoryResourceAccess(state, mem);
	// Upstream 5429888d4: elect and count from the guest EXEC words, which V_MBCNT also ranks lanes by.
	const auto low   = ctx.Arg(inst, 1);
	const auto high  = ctx.Arg(inst, 2);
	const auto count = Binary(state, spv::OpIAdd, TypeU32(state),
	                          Unary(state, spv::OpBitCount, TypeU32(state), low),
	                          Unary(state, spv::OpBitCount, TypeU32(state), high));
	const auto mask  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), mask, low, high,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	const auto first = ctx.FirstLane(mask);
	const auto source_lane =
	    state.lane_count == 2
	        ? Binary(state, spv::OpBitwiseAnd, TypeU32(state), first, ConstantU32(state, 31))
	        : first;
	const auto is_first       = Binary(state, spv::OpIEqual, TypeBool(state),
	                                   EmitSubgroupLocalInvocationId(state), source_lane);
	auto bounds = EmitMemoryElementInBounds(state, access, index);
	if (mem.kind == IR::ResourceKind::Gds) {
		bounds = AndCondition(state, bounds, region_bounds);
	}
	const auto condition = AndCondition(
	    state, is_first,
	    AndCondition(state,
	                 Binary(state, spv::OpINotEqual, TypeBool(state), count, ConstantU32(state, 0)),
	                 bounds));
	const auto atomic = EmitValueOrZeroIfCondition(state, condition, [&]() {
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(append ? spv::OpAtomicIAdd : spv::OpAtomicISub, TypeU32(state),
		                          value, EmitMemoryElementPointer(state, access, index),
		                          ConstantU32(state, mem.kind == IR::ResourceKind::Gds
		                                                 ? spv::ScopeDevice
		                                                 : spv::ScopeWorkgroup),
		                          ConstantU32(state, spv::MemorySemanticsMaskNone), count);
		return value;
	});
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), atomic, source_lane);
	return result;
}

uint32_t EmitReadConst(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (state.flattened_srt_uniform_variable != 0) {
		// Uniform const buffers (ShaderIR.h): the slot is an immediate below srt_reads.size() (ValidateNativeProgram),
		// and the host's flat buffer holds at least srt_reads.size() words.
		return LoadUniformDword(state, state.flattened_srt_uniform_variable, inst.Arg(1).Resolve().U32());
	}
	if (state.flattened_srt_variable == 0) {
		ctx.Fail(inst, "requires the flattened SRT descriptor");
	}
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.flattened_srt_variable, ConstantU32(state, 0),
	                          ctx.Arg(inst, 1));
	return EmitNative<spv::OpLoad, IR::Type::U32>(state, pointer);
}

void EmitReadConstBuffer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto mem = ctx.Memory(inst);
	if (mem.planning_only) return;
	if (mem.kind == IR::ResourceKind::IndirectBuffer) {
		// A member takes the value its leader loaded for this lane half (a half the
		// leader was not emitted for loads it here as before).
		auto& state = ctx.state;
		if (!state.scalar_groups.empty()) {
			const uint32_t half = ctx.half < 2u ? ctx.half : 0u;
			if (state.scalar_group_members.contains(&inst)) {
				const auto value = state.scalar_group_values[half].find(&inst);
				if (value != state.scalar_group_values[half].end()) {
					ctx.Define(inst, value->second);
					return;
				}
			} else if (const auto leader = state.scalar_group_leader.find(&inst); leader != state.scalar_group_leader.end()) {
				EmitScalarLoadGroup(ctx, inst, state.scalar_groups[leader->second]);
				return;
			}
		}
		ctx.Define(inst, LoadIndirectScalarBuffer(ctx, inst));
		return;
	}
	auto& state        = ctx.state;
	mem.kind           = IR::ResourceKind::ScalarBuffer;
	// An immediate offset folds into a constant dword index (upstream 0f7d11254).
	const auto offset = inst.Arg(1).Resolve();
	auto       index  = offset.IsImmediate()
	                        ? ConstantU32(state, (offset.U32() >> 2u) + (mem.offset >> 2u))
	                        : Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                 ctx.Arg(inst, 1), ConstantU32(state, 2));
	if (!offset.IsImmediate() && mem.offset >= 4u) {
		index = Binary(state, spv::OpIAdd, TypeU32(state), index, ConstantU32(state, mem.offset >> 2u));
	}
	const auto access    = PrepareMemoryResourceAccess(state, mem);
	// Scalar buffer addressing aligns the base and each offset independently.
	const auto element = Binary(
	    state, spv::OpIAdd, TypeU32(state), index,
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), access.byte_offset, ConstantU32(state, 2)));
	if (RobustBuffer(state, access)) {
		const auto safe    = RobustElement(state, access, 0u, element);
		const auto pointer = EmitMemoryElementPointer(state, access, safe);
		ctx.Define(inst, LoadResourceElement(state, TypeU32(state), state.builder.AllocateId(), pointer, access, safe, false));
		return;
	}
	const auto condition = EmitMemoryElementInBounds(state, access, element);
	ctx.Define(inst, EmitValueOrZeroIfCondition(state, condition, [&]() {
		           // (EmitNative's order: the pointer, then the value; no memory-access operand.)
		           const auto pointer = EmitMemoryElementPointer(state, access, element);
		           return LoadResourceElement(state, TypeU32(state), state.builder.AllocateId(), pointer, access, element,
		                                      false);
	           }));
}

// The dense resource a typed buffer selector picks (upstream fd66f580b); 0 without candidates. Our resource tracking does
// not produce GetBufferResourceIndex: the emitter exists because every IR opcode has one.
uint32_t EmitGetBufferResourceIndex(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto& buffer = ctx.state.program.info.buffers.at(inst.Flags<uint32_t>());
	if (buffer.indirect_resources.empty()) return ConstantU32(ctx.state, 0u);
	return EmitIndirectResourceIndex(ctx.state, ctx.Arg(inst, 0), buffer.indirect_mapping_offset,
	                                 buffer.indirect_search_iterations, UINT32_MAX);
}

void EmitLoadMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op  = inst.GetOpcode();
	const auto& mem = ctx.Memory(inst);
	if (op == IR::ValueOpcode::LoadAddressU32 && mem.planning_only) return;
	const auto buffer_components = IR::BufferComponentCount(op);
	const auto shared_components = IR::SharedComponentCount(op);
	const auto address_info      = IR::AddressOpcodeInfoOf(op);
	uint32_t   value;
	if (mem.kind == IR::ResourceKind::FlatLocal)
		value = LoadLocalFlat(ctx, inst);
	else if (buffer_components > 1u || (buffer_components == 1u && mem.formatted && mem.data_bits == 16u) ||
	         (buffer_components == 1u && mem.kind == IR::ResourceKind::IndirectBuffer && !mem.formatted &&
	          inst.GetType() != IR::Type::U8 && inst.GetType() != IR::Type::U16))
		value = LoadBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		value = LoadWideShared(ctx, inst, shared_components);
	else if (mem.kind == IR::ResourceKind::ScalarAddress)
		value = LoadBdaDword(ctx, GuestAddress(ctx, inst, mem));
	else if (address_info.access == IR::AddressAccess::Read &&
	         mem.kind != IR::ResourceKind::Scratch)
		value = LoadBda(ctx, GuestAddress(ctx, inst, mem), ctx.Arg(inst, inst.NumArgs() - 1),
		                address_info.data_bits, mem.coherent);
	else if (op == IR::ValueOpcode::LoadBufferU32 && mem.formatted)
		value = mem.kind == IR::ResourceKind::IndirectBuffer ? LoadIndirectFormattedX(ctx, inst)
		                                                     : FormattedLoad(ctx, inst, mem);
	else if (mem.kind == IR::ResourceKind::IndirectBuffer &&
	         (inst.GetType() == IR::Type::U8 || inst.GetType() == IR::Type::U16))
		value = LoadIndirectSubword(ctx, inst, inst.GetType() == IR::Type::U8 ? 8u : 16u);
	else if (inst.GetType() == IR::Type::U8)
		value = LoadSubword(ctx, inst, mem, 8, false);
	else if (inst.GetType() == IR::Type::U16)
		value = LoadSubword(ctx, inst, mem, 16, false);
	else
		value = LoadWord(ctx, inst, mem);
	ctx.Define(inst, value);
	if (mem.kind == IR::ResourceKind::Lds) ctx.state.lds_pair_synced = false; // (EmitPairLdsWriteSync)
}

void EmitStoreMemory(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto  op                = inst.GetOpcode();
	const auto& mem               = ctx.Memory(inst);
	const auto  buffer_components = IR::BufferComponentCount(op);
	const auto  shared_components = IR::SharedComponentCount(op);
	const auto  type              = inst.Arg(inst.NumArgs() - 2).GetType();
	EmitPairLdsWriteSync(ctx.state, mem, true);
	if (mem.kind == IR::ResourceKind::FlatLocal)
		StoreLocalFlat(ctx, inst);
	else if (const auto formatted = IndirectFormattedStoreComponents(inst, mem); formatted != 0u)
		StoreIndirectFormatted(ctx, inst, formatted); // record-keyed: record-keyed formatted stores
	else if (mem.kind == IR::ResourceKind::IndirectBuffer && mem.formatted)
		ctx.Fail(inst, "formatted store through an unresolved descriptor"); // tracking stops such a store first
	else if (mem.kind == IR::ResourceKind::IndirectBuffer && (type == IR::Type::U8 || type == IR::Type::U16))
		StoreIndirectSubword(ctx, inst, type == IR::Type::U8 ? 8u : 16u);
	else if (mem.kind == IR::ResourceKind::IndirectBuffer)
		StoreIndirectBuffer(ctx, inst, std::max(buffer_components, 1u));
	else if (buffer_components > 1u)
		StoreWideBuffer(ctx, inst, buffer_components);
	else if (shared_components > 1u)
		StoreWideShared(ctx, inst, shared_components);
	else if (op == IR::ValueOpcode::StoreBufferU32 && mem.formatted)
		FormattedStore(ctx, inst, mem);
	else if (type == IR::Type::U8)
		StoreSubword(ctx, inst, mem, 8);
	else if (type == IR::Type::U16)
		StoreSubword(ctx, inst, mem, 16);
	else
		StoreWord(ctx, inst, mem);
	EmitPairLdsWriteSync(ctx.state, mem, false);
}

uint32_t EmitSharedIncDec(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto replacement =
	    inst.GetOpcode() == IR::ValueOpcode::SharedAtomicInc32 ? AtomicIncrement : AtomicDecrement;
	return EmitAtomicUpdate(ctx, inst, ctx.Memory(inst), replacement);
}

void EmitSharedAtomicMaskedOr32(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto keep = Unary(ctx.state, spv::OpNot, TypeU32(ctx.state), ctx.Arg(inst, 1));
	EmitAtomicUpdate(ctx, inst, ctx.Memory(inst),
	                 [keep](EmitterState& state, uint32_t old, uint32_t value) {
		                 return Binary(state, spv::OpBitwiseOr, TypeU32(state),
		                               Binary(state, spv::OpBitwiseAnd, TypeU32(state), old, keep), value);
	                 });
}

uint32_t EmitSwizzleU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	state.builder.AddFunction(spv::OpStore, ctx.scratch_u32_variable, ctx.Arg(inst, 0));
	const auto source = EmitNative<spv::OpLoad, IR::Type::U32>(state, ctx.scratch_u32_variable);
	const auto target = EmitDsSwizzleTargetLane(state, EmitSubgroupLocalInvocationId(state),
	                                            inst.Arg(1).IsImmediate() ? inst.Arg(1).U32() : 0);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

uint32_t EmitPermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state   = ctx.state;
	auto       lane    = EmitSubgroupLocalInvocationId(state);
	if (state.lane_count == 2) {
		lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane, ConstantU32(state, 31));
	}
	const auto address = ctx.Arg(inst, 1);
	const auto word    = Binary(state, spv::OpShiftRightLogical, TypeU32(state), lane,
	                            ConstantU32(state, 5));
	const auto ballot_word = [&](uint32_t predicate) {
		const auto ballot = state.builder.AllocateId();
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
		                          ConstantU32(state, spv::ScopeSubgroup), predicate);
		state.builder.AddFunction(spv::OpVectorExtractDynamic, TypeU32(state), result, ballot, word);
		return result;
	};
	// RDNA2 permutes independently within each 32-lane half. Intersect the source
	// address bit ballots to find this destination's enabled writers without LDS.
	auto writers = ballot_word(ctx.Arg(inst, 2));
	for (uint32_t bit = 0; bit < 5; ++bit) {
		const auto address_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), address,
		                                ConstantU32(state, 1u << (bit + 2)));
		const auto mask = ballot_word(Binary(state, spv::OpINotEqual, TypeBool(state),
		                                      address_bit, ConstantU32(state, 0)));
		const auto lane_bit = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
		                             ConstantU32(state, 1u << bit));
		const auto selected = Select(
		    state, TypeU32(state),
		    Binary(state, spv::OpINotEqual, TypeBool(state), lane_bit, ConstantU32(state, 0)),
		    mask, Unary(state, spv::OpNot, TypeU32(state), mask));
		writers = Binary(state, spv::OpBitwiseAnd, TypeU32(state), writers, selected);
	}
	const auto active = Binary(state, spv::OpINotEqual, TypeBool(state), writers,
	                           ConstantU32(state, 0));
	const auto base = Binary(state, spv::OpBitwiseAnd, TypeU32(state), lane,
	                         ConstantU32(state, ~31u));
	const auto source = Select(state, TypeU32(state), active,
	                           Binary(state, spv::OpBitwiseOr, TypeU32(state), base,
	                                  EmitFindUMsb32(state, writers)), lane);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, TypeU32(state), result,
	                          ConstantU32(state, spv::ScopeSubgroup), ctx.Arg(inst, 0), source);
	return Select(state, TypeU32(state), active, result, ConstantU32(state, 0));
}

uint32_t EmitBpermuteU32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state  = ctx.state;
	const auto source = ctx.Arg(inst, 0);
	const auto index  = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                                  ctx.Arg(inst, 1), ConstantU32(state, 2)),
	                           ConstantU32(state, 31));
	const auto base   = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                           EmitSubgroupLocalInvocationId(state), ConstantU32(state, ~31u));
	const auto target = Binary(state, spv::OpBitwiseOr, TypeU32(state), base, index);
	return EmitDsMaskedLaneRead(state, source, target, ctx.Arg(inst, 2));
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
