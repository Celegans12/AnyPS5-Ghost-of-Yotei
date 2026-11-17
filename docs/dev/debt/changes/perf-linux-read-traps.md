# libc

## add Functional

- [Read traps on Linux](../../../core/libs/prx/libc/src/GuestWriteWatch.cpp) (libc) - a thread holding the GPU lock or resolving the range reopens a trapped page without waiting, so a guest read of that page afterwards can see bytes the GPU has not written; a wait over 5 s also opens the page (`[read-trap]` line). A fault whose SIGSEGV is blocked, or that reaches a handler other than libc's crash report and libkernel's exception delivery and guest signal dispatch, kills the process. Only ranges registered for the write watch can be trapped. Kernel accesses wait only through the host write and read guards, the checked copies and libkernel's protection and mapping entry points; the paths under libkernel's locks open the pages without waiting, and any other system call on a trapped page fails with `EFAULT`.

# libSceAgcDriver

## add Opt-in draw paths

- [Early end-of-pipe behind read traps](../../../core/libs/prx/libSceAgcDriver/Graphics/src/Recorder.cpp) (`APS5_EARLY_EOP_READ_TRAP`) - opt-in, Linux on x86-64. A CPU write-back made by a completion without noting its range as a pending write is not trapped; `APS5_VERIFY_EARLY_EOP_READ_TRAP` checks only the copied buffers written back through `Recorder::NoteWrittenBack`. A label whose trapped page has no direct memory backing holds its frame. In-place reads of changing ranges still hold the frame (rule 2), except GPU copy sources, which `APS5_EARLY_EOP_INTERRUPT` copies at record time. `Recorder::ReadTrapEnabled` stays Windows only.
