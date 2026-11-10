#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTLOADFAST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTLOADFAST_H_

#include <cstdint>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;

// KYTY_PERF_INDIRECT_LOAD_FAST (default 1; 0 off, 2 verify, 3 live) with KYTY_PERF_INDIRECT_LOAD_FAST_HASHES=<hex,...>|all
// (default: the five "Deferred Ambient>Eval non-RT Probes" compute shaders): a BUFFER_LOAD_FORMAT_D16_XYZW through a
// GPU-selected V# (the per-shape function, an OpSwitch over all 78 buffer formats,
// ~260 ALU and two page-table lookups per call) first tests the V# for the one layout these shaders read,
// k16_16_16_16Float with DST_SEL X,Y,Z,W; then it reads the element's two dwords inline (the same bounds check, one
// page-table lookup when both dwords are in one 16 KiB page) and returns them with only the signaling-NaN quieting
// that the half -> float -> half round trip of the generic path applies. Any other V# calls the function as before.
// =2 also calls the function on the fast path and returns its result, counting checks / mismatches / other V#s in
// fault-buffer diag words DiagWord + 0/1/2 (PERF indirect_load_fast verify line).
// =3 compiles both paths and takes the fast one while bit 0 of the live shader flags is set (SpirvLive below): by
// default always; KYTY_LIVE_AB=KYTY_PERF_INDIRECT_LOAD_FAST toggles it per phase.
namespace IndirectLoadFast {

// After the watchdog's miss slots (BufferCache::FAULT_DIAG_MISS_FIRST + 4 * FAULT_DIAG_MISS_SLOTS = 20544; words 12.. are
// KYTY_FIX_CLAMP_NAN's, 12-13 free since KYTY_PERF_TABLE_BUFFERS left); checked in faultManager.cpp.
inline constexpr uint32_t DiagWord = 20544;

[[nodiscard]] int  Mode();
[[nodiscard]] bool Listed(uint64_t shader_hash);

} // namespace IndirectLoadFast

// Live shader flags: one word of the fault buffer, after the diagnostic tail (BufferCache::FAULT_DIAG_BASE +
// FAULT_DIAG_WORDS; the fault buffer has LiveWordsBytes more for it), which shaders compiled with a switch's live mode
// read before taking the switch's path. The GPU thread rewrites it at frame boundaries when a live A/B phase changes
// it (vkCmdUpdateBuffer between full barriers), so either value runs whole frames. Both paths of such a switch give
// the same results, so a dispatch recorded before the change that runs after it is still exact.
namespace SpirvLive {

inline constexpr uint32_t LiveWordsBytes      = 256;
inline constexpr uint32_t FlagIndirectLoadFast = 1u << 0u;

// Whether any switch is in its live mode (the fault buffer then has the flags word and Update writes it).
[[nodiscard]] bool Enabled();
// The flags for the current live A/B phase.
[[nodiscard]] uint32_t Flags();

// GPU thread, at a frame boundary: writes Flags() into `fault_buffer` when it differs from the last written value.
void Update(CommandScheduler& scheduler, Buffer& fault_buffer, uint32_t word_index);

} // namespace SpirvLive

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_INDIRECTLOADFAST_H_ */
