# libc

## add Functional

- [Read traps on shared views](../../../core/libs/prx/libc/include/WindowsMappings.hpp) (libc) - `GuestArenaTrapReads` removes access from the guest view of shared pages until a resolver registered by the AGC driver released them. A fault raised while the faulting thread holds the mapping lock deadlocks, as a write fault on an armed shared page does. A view unmapped while its resolver refused is dropped trapped. Windows only; `GuestWriteWatchTrapReads` serves the macOS label traps.
