# relinker

## add Functional

- An Intel conversion stub that took in the instruction after the one it lowers, such as the 4-byte VRCPPS or VRSQRTPS the conversion widens over the next instruction, leaves that instruction outside the reach of `--frame-red-zone`; the framer moves the displacement of an RSP operand it holds inside the stub body ([framer](../../../core/relinker/codegen/src/x86/RedZoneFramer.cpp)) only when the moved displacement keeps its width. A displacement that must grow, a stub that took in a push, a pop or an instruction without a displacement, an operand of the lowered instruction itself and a leaf that has to be relocated whole over any stub are not framed: the leaf goes into the list of unframed ranges. Of three titles converted with `--to-intel`, 8 of 9, 6 of 7 and 8 of 19 leaves that overlapped a stub are framed now; none runs framed.
- The list of unframed ranges of an image ([marking](../../../core/relinker/relinker/src/pipeline/RedZoneFraming.cpp)) is a `.rzgap` section appended to the PE after it is written, and the mark of the image becomes `APS5RZ02` with the table's RVA at 0x30 and its entry count at 0x34. It has one entry per range, merged where they touch: an image without room for another section header, with a mark or table bytes already in use or with 96 sections is left unmarked. Each range is a whole function, since the faults it is meant for come from the code of that function; a leaf entered from another function's range, or whose red zone a callee reaches, is not covered more finely.

# libc

## add Functional

- The [red-zone gate](../../../core/libs/prx/libc/include/RedZoneGate.hpp) (libc) on Windows keeps page tracking on for an image marked `APS5RZ02` and stops the title with a message when a fault that page tracking served has its RIP inside a range of the image's list ([crash handler](../../../core/libs/prx/libc/src/specifics/windows/CrashReport.cpp)); by then Windows has written the exception record over the leaf's red zone, so the stop replaces a silent corruption, it does not prevent one. It reads at most 32 tables, one per image, and treats an image with more, or with a table that is not sorted, overlaps itself or leaves the image, as unframed. It was written without a Windows build: its table logic ran on Linux against a shim of the Windows types, and the fault path has not run.
