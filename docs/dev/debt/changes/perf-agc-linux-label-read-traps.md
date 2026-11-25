# libc

## add Functional

- In standalone Linux builds without Kyty, the [write watch](../../../core/libs/prx/libc/src/GuestWriteWatch.cpp) read traps (`GuestWriteWatchTrapReads`) make pages `PROT_NONE`, unlike its `userfaultfd` write recording, so they share the gaps of the macOS read traps above: a system call that reads a trapped page fails with `EFAULT`, another thread can read a range the resolver has partly stored with the CPU, and a thread that holds what the resolver needs and touches a trapped range waits forever. They are resolved from a SIGSEGV/SIGBUS handler installed with the first resolver, which passes other faults to the handler installed before it; libkernel's fault delivery also asks the watch first. In addition:
  - a trap is set only on pages known to be read-write. The protection is read from `/proc/self/maps` at the first trap in a registered range and then follows `GuestWriteWatchProtect`, which libkernel's `mprotect` and the guest arena use. A plain `mprotect` of a watched page after that is not seen, and a trap released there gives the page back read-write;
  - registering memory drops its traps without restoring their pages, since every caller registers a range it has just mapped; a range registered again without a new mapping keeps inaccessible pages. Unregistering restores them without running the resolver.
