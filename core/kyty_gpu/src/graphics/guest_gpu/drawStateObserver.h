#ifndef GRAPHICS_GUEST_GPU_DRAW_STATE_OBSERVER_H
#define GRAPHICS_GUEST_GPU_DRAW_STATE_OBSERVER_H

// Based on chenxiao07/KytyPS5 v20261020b (src/local/draw-state-observer.h), GPL-2.0; adapted to this tree's packet
// classes (RegistersWrittenBy in graphicsRun.cpp) and draw kinds.
//
// The draw-state observer (measurement only: KYTY_TRACE_CLEAN_DRAWS;
// graphicsRun.cpp DrawStateObserverOn): whether the graphics state changed between two consecutive draws of one command
// processor. A draw is clean when nothing but the graphics user-data SGPRs (PS 0x0c-0x2b, GS 0x8c-0xab, HS 0x10c-0x12b), the
// compute SH space (0x200 and up), the index state and the instance count changed since the previous direct draw: the
// previous draw's register-derived state (programs' source, pipeline, targets, dynamic state) still applies; what its
// user data points at (resources, constants) may differ.
//
// A register write dirties the gap only when it changes a value (shadow copies of the CX/SH/UC spaces, each value with
// the generation it was written in, so a reset is one increment). Packets that may write registers in other ways
// (RegistersWrittenBy says so: everything but the register packets, direct draws, index/instance setup, SET_BASE, buffer
// jumps and plain/marker NOPs) dirty it as unknown writes; context clears/pushes/pops and register resets invalidate the
// shadows. Indirect draws are draws that end a series (their handlers patch SGPRs from GPU data).
//
// Each gap's first cause of dirt is kept (Cause): KYTY_TRACE_CLEAN_DRAWS reports it for the draws the observer called
// dirty although no register group the renderer reads changed (diagnostic, active project step K1).
#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics::DrawStateObserver {

// A cause of dirt: its kind (high 16 bits) and detail (register offset, opcode, NOP code, invalidating site).
enum CauseKind : uint32_t {
	CauseNone, CauseCx, CauseSh, CauseUc, CauseBeyond, CausePacket, CauseNop, CausePredicated, CauseInvalidate, CauseNopReset,
	CauseIndirectDraw, CauseNullBlock, CauseShortPacket
};
[[nodiscard]] constexpr uint32_t Cause(uint32_t kind, uint32_t detail) {
	return (kind << 16u) | (detail & 0xffffu);
}
// Invalidate's detail: a command run on the GPU thread, a processor's slice start.
constexpr uint32_t InvalidateCommand = 1u;
constexpr uint32_t InvalidateSlice   = 2u;

struct Shadow {
	std::array<uint64_t, Pm4::CX_NUM> cx {};
	std::array<uint64_t, Pm4::SH_NUM> sh {};
	std::array<uint64_t, Pm4::UC_NUM> uc {};
	uint64_t generation = uint64_t {1} << 32u;
	bool     chain      = false; // the previous draw was a direct draw (a series can go on)
	bool     dirty      = true;  // graphics state changed since that draw
	bool     last_clean = false; // the latest draw was clean
	uint32_t cause      = Cause(CauseInvalidate, 0); // the first cause of dirt since the previous draw
	uint32_t last_cause = 0;                         // the latest draw's (0 when clean)
};
inline thread_local Shadow t_shadow;

inline void Dirty(uint32_t cause) {
	auto& s = t_shadow;
	if (!s.dirty) s.cause = cause;
	s.dirty = true;
}

// Also free: the per-draw tags the game writes before most draws (SPI_SHADER_PACE_ID_PS changed its value at ~half of
// the gate's draws, active project step K1), which the command processor ignores (pm4Handlers.cpp
// HwShIgnoreShaderRegister: no state reads them) - the PS and GS pace ids and the HS checksum tag.
[[nodiscard]] inline bool ShFree(uint32_t offset) {
	return (offset >= 0x0cu && offset <= 0x2bu) || (offset >= 0x8cu && offset <= 0xabu) ||
	       (offset >= 0x10cu && offset <= 0x12bu) || offset >= 0x200u || offset == Pm4::SPI_SHADER_PACE_ID_PS ||
	       offset == Pm4::SPI_SHADER_PACE_ID_GS || offset == Pm4::SPI_SHADER_PGM_CHKSUM_HS;
}
// The user config registers that are a draw's data, not its state: the object id (each draw's commit pushes its own
// copy of the register, DrawRegs::object_id).
[[nodiscard]] inline bool UcFree(uint32_t offset) {
	return offset == Pm4::VGT_OBJECT_ID;
}

// space: 0 CX, 1 SH, 2 UC.
inline void WriteRegister(uint32_t space, uint32_t raw_offset, uint32_t value) {
	auto&      s      = t_shadow;
	const auto offset = raw_offset & 0xffffu;
	const auto update = [&](auto& values) {
		if (offset >= values.size()) {
			Dirty(Cause(CauseBeyond, space));
			return;
		}
		const uint64_t entry = s.generation | value;
		if (values[offset] != entry) {
			values[offset] = entry;
			Dirty(Cause(space == 0 ? CauseCx : space == 1 ? CauseSh : CauseUc, offset));
		}
	};
	if (space == 1) {
		if (offset < Pm4::SH_NUM && ShFree(offset)) return;
		update(s.sh);
	} else if (space == 0) {
		update(s.cx);
	} else {
		if (UcFree(offset)) return;
		update(s.uc);
	}
}

inline void WriteIndirectBlock(uint32_t space, const uint32_t* packet) {
	const auto* block = reinterpret_cast<const uint32_t*>((uint64_t(packet[1]) & 0xfffffffcu) | (uint64_t(packet[2]) << 32u));
	const uint32_t pairs = packet[4] & 0x3fffu;
	if (block == nullptr) {
		Dirty(Cause(CauseNullBlock, space));
		return;
	}
	for (uint32_t i = 0; i < pairs; ++i) {
		if (block[2 * i] == 0xffffffffu) continue;
		WriteRegister(space, block[2 * i], block[2 * i + 1]);
	}
}

// The registers changed in bulk (context clear/push/pop, a register reset, a new command buffer): nothing the shadow holds
// is known any more.
inline void Invalidate(uint32_t cause) {
	auto& s = t_shadow;
	s.generation += uint64_t {1} << 32u;
	if (s.generation == 0) { // wrapped: an entry of the first generation would be taken as known again
		s.cx.fill(0);
		s.sh.fill(0);
		s.uc.fill(0);
		s.generation = uint64_t {1} << 32u;
	}
	s.chain = false;
	s.dirty = true;
	s.cause = cause; // a broken series outranks a changed register
}

// Every packet the command processor is about to execute (packet: the header; packet_dw: the whole packet's dwords;
// writes: RegistersWrittenBy of the packet).
inline void ObservePacket(uint32_t opcode, const uint32_t* packet, uint32_t packet_dw, uint32_t writes) {
	auto& s = t_shadow;
	switch (opcode) {
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
			s.last_clean = s.chain && !s.dirty;
			s.last_cause = s.last_clean ? 0u : s.cause;
			s.chain      = true;
			s.dirty      = false;
			s.cause      = 0;
			return;
		case Pm4::IT_DRAW_INDIRECT:
		case Pm4::IT_DRAW_INDIRECT_MULTI:
		case Pm4::IT_DRAW_INDEX_INDIRECT:
		case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI:
			s.last_clean = false;
			s.last_cause = Cause(CauseIndirectDraw, opcode);
			s.chain      = false;
			s.dirty      = true;
			s.cause      = Cause(CauseIndirectDraw, opcode);
			return;
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX: {
			const uint32_t space = opcode == Pm4::IT_SET_CONTEXT_REG ? 0u : opcode == Pm4::IT_SET_SH_REG ? 1u : 2u;
			// SET_UCONFIG_REG_INDEX's index bits share the offset dword; the offset is its low 16 bits either way.
			for (uint32_t i = 2; i < packet_dw; ++i) WriteRegister(space, packet[1] + (i - 2), packet[i]);
			return;
		}
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
		case Pm4::IT_SET_SH_REG_INDIRECT:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
			if (packet_dw < 5) {
				Dirty(Cause(CauseShortPacket, opcode));
				return;
			}
			WriteIndirectBlock(opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT ? 0u : opcode == Pm4::IT_SET_SH_REG_INDIRECT ? 1u : 2u,
			                   packet);
			return;
		// No graphics register is written by these (synchronization, events, dispatches - their registers are the compute
		// SH space -, DMA into memory, index and instance setup, buffer jumps).
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_INDIRECT_BUFFER:
		case Pm4::IT_DISPATCH_DIRECT:
		case Pm4::IT_DISPATCH_INDIRECT:
		case Pm4::IT_ACQUIRE_MEM:
		case Pm4::IT_RELEASE_MEM:
		case Pm4::IT_WAIT_REG_MEM:
		case Pm4::IT_WAIT_REG_MEM_64:
		case Pm4::IT_EVENT_WRITE:
		case Pm4::IT_EVENT_WRITE_EOP:
		case Pm4::IT_EVENT_WRITE_EOS:
		case Pm4::IT_PFP_SYNC_ME:
		case Pm4::IT_DMA_DATA: return;
		case Pm4::IT_NOP: {
			const auto r = KYTY_PM4_R(packet[0]);
			if (r == Pm4::R_CONTEXT_STATE || r == Pm4::R_DISPATCH_RESET || r == Pm4::R_DRAW_RESET) {
				Invalidate(Cause(CauseNopReset, r));
				return;
			}
			if (r == Pm4::R_RELEASE_MEM || r == Pm4::R_DMA_DATA || r == Pm4::R_WAIT_FLIP_DONE) return; // acquire: IT_ACQUIRE_MEM (a5ceb4818)
			if (writes != 0u) Dirty(Cause(CauseNop, r)); // other NOP-encoded commands (shader setup, writes) may write registers
			return;
		}
		default:
			if (writes != 0u) Dirty(Cause(CausePacket, opcode)); // a packet that may write registers some other way
			return;
	}
}

// The latest draw's answer (read by the draw's handler, after ObservePacket ran for it).
[[nodiscard]] inline bool LastClean() {
	return t_shadow.last_clean;
}
[[nodiscard]] inline uint32_t LastCause() {
	return t_shadow.last_cause;
}

} // namespace Libs::Graphics::DrawStateObserver

#endif
