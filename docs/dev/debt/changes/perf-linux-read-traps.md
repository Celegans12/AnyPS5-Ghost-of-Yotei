# libc

## add Functional

- [Read traps on Linux](../../../core/libs/prx/libc/src/GuestWriteWatch.cpp) (libc) - a thread holding the GPU lock or resolving the range reopens a trapped page without waiting, so a guest read of that page afterwards can see bytes the GPU has not written; a wait over 5 s also opens the page (`[read-trap]` line). A fault whose SIGSEGV is blocked, or that reaches a handler other than libc's crash report and libkernel's exception delivery and guest signal dispatch, kills the process. Only ranges registered for the write watch can be trapped. Kernel accesses wait only through the host write and read guards, the checked copies and libkernel's protection and mapping entry points; the paths under libkernel's locks open the pages without waiting, and any other system call on a trapped page fails with `EFAULT`.
