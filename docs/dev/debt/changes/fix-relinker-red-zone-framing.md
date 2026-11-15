# relinker

## change `--frame-red-zone` frames only leaf functions

- `--frame-red-zone` frames a function that keeps values below RSP by the first of three forms that fits ([framer](../../../core/relinker/codegen/src/x86/RedZoneFramer.cpp)): in place, with stubs at its prologue and exits; through a return thunk, whose entry lowers RSP and calls the rest of the function; or relocated whole into a stub that runs it with RSP 128 bytes lower. A function none fits keeps its red zone, is counted by reason in the conversion log and leaves its image without the framing mark. The three forms were not compiled or run in this tree: their author ran them on four titles, none built with Unity, and no title of ours was converted with them. A value below RSP reached through a register other than RSP or RBP is not seen, and a function that calls and keeps values below RSP is framed only when it has a frame pointer; without one it is counted as skipped.

## change An RSP operand that `--frame-red-zone` cannot move in place

- An RSP operand that `--frame-red-zone` cannot move in place, an 8-bit displacement the shift takes past 0x7f or one without a displacement, moves to a stub with the instructions after it and is re-encoded there with the moved displacement ([framer](../../../core/relinker/codegen/src/x86/RedZoneFramer.cpp)). An EVEX operand, whose 8-bit displacement is scaled, is not moved: its function is not framed in place. A game patch over a framed site or a moved displacement is refused.

## change function bounds come from the executable's `.eh_frame` FDEs

- In `--frame-red-zone`, code outside every `.eh_frame` FDE is cut into functions at int3 padding, undecodable bytes, zero fill, direct call targets and, after an exit, at code no earlier branch reaches ([framer](../../../core/relinker/codegen/src/x86/RedZoneFramer.cpp)). The cut is a guess: a function cut in two is framed as two functions, and data inside a code segment is decoded as code. A code range left outside every function that reaches below RSP is counted and leaves its image without the framing mark.

## add Functional

- The [red-zone framing mark](../../../core/relinker/relinker/src/pipeline/RedZoneFraming.cpp) is eight bytes at offset 0x28 of the DOS header of a Windows image. It says that the conversion left no function with values below RSP unframed, not that the framed code was run: no test executes a marked title. Linux and macOS images are neither framed nor marked. `--frame-red-zone` is off unless given, since framing rewrites the code of every image and no title of ours has run framed.

# libc

## add Functional

- The [red-zone gate](../../../core/libs/prx/libc/include/RedZoneGate.hpp) (libc) on Windows refuses page tracking and read traps unless the executable and every loaded guest module carry the framing mark; it was written without a build and has not run. It takes an image for guest code by its `.guest.prx` name or by a PE header at 0x80 with no DOS stub before it, as the relinker writes, so a host executable linked without a DOS stub counts as an unframed guest. Later loads are seen through `LdrRegisterDllNotification` of ntdll, whose notice layout is declared in the header; if registering fails the module list is read again at every query. A block armed or a page trapped before an unmarked module loads stays so until it is written or released. The gate does not cover the faults taken whatever the mark: write protection of shared views, SSE4a emulation and exceptions delivered to the guest.
- The [fault storm](../../../core/libs/prx/libc/src/specifics/windows/FaultStorm.cpp) (libc) (`APS5_FAULT_STORM`) was not compiled or run in this tree. It lands only on threads it finds running guest code, so a function that runs rarely may never be stepped, and it proves nothing about a host library built with a red zone unless `APS5_FAULT_STORM_GUEST_CLOBBER` is left unset.
