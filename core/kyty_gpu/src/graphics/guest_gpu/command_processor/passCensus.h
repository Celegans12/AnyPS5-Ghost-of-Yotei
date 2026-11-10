#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSCENSUS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSCENSUS_H_

#include <cstdint>

// Parallel-translation project, step S0 (diagnostic only,
// default off, removed with the project's decision):
// KYTY_TRACE_PASS_CENSUS=1: the GPU thread's time between PM4 events is attributed to the game's marker pass of the
// command processor that runs (its marker stack, two levels: "outer>inner"), less the time it blocked on the card
// (MasterSemaphore::Wait); with the pass's draw and dispatch packets. PASS_CENSUS lines every 5 s: the passes by GPU-thread
// ms per frame, and the share of the frame's translation time they hold.
// KYTY_DIAG_CEILING_FREE_PASSES=<substring,...> (ceiling, APPROXIMATION - the image breaks; KYTY_LIVE_AB toggles it as
// KYTY_DIAG_CEILING_FREE_PASSES): draw and dispatch packets inside a pass whose name holds a listed substring return
// before the renderer (no translation, no card work); register writes and every other packet run as before.
namespace Libs::Graphics::PassCensus {

// The GPU thread's blocked time on the card so far (MasterSemaphore::Wait adds to it; the census subtracts it).
inline thread_local uint64_t t_wait_ns = 0;

[[nodiscard]] bool Enabled();
// A marker push or pop of the command processor `cp` (its name for a push, null for a pop).
void Marker(const void* cp, const char* push_name);
// A draw (or dispatch) packet of `cp`; true when the ceiling leaves it out.
[[nodiscard]] bool Packet(const void* cp, bool dispatch);
// A dispatch's compute shader (the code hash; the census lists each pass's shaders: PASS_CENSUS shaders lines).
void NoteShader(uint64_t hash);

} // namespace Libs::Graphics::PassCensus

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSCENSUS_H_
